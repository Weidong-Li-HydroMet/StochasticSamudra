"""Fair-CRPS fine-tuning of the SamudraV2 checkpoint (AIFS-CRPS style).

=== InMOS_1degree copy (2026-10-12) -- STAGE 1 "IC noise": deterministic fair-CRPS fine-tune ===
(file renamed from train_finetune_samudra_crps.py -> 1-IC_noise.py; launcher 1-IC_noise.sh)
Minimal-diff copy of Sampling/Derecho src/CRPS/train_finetune_samudra_crps.py for the
1deg checkpoint. Every changed line is marked `1deg:`. Summary:
  * model/data via samudra_1deg (OM4.zarr, num_out=77 from the checkpoint, datasets
    re-windowed to one 5-day step); defaults point at the 1deg ckpt/data and write to
    /data/weidong/InMOS_1deg/checkpoints/SamudraV2_1deg_CRPS;
  * ONE-STEP rollout: the state is [x(t-1), x(t)] (154 ch), the model returns x(t+1)
    (77 ch), state <- cat(state[:, 77:], y); loss/val/dynamic weights act on the 77
    output channels;
  * same DAYS as the 0.25deg run: --rollout-steps 8 (40 d), --val-steps 16 (80 d);
  * --bn-mode eval|train (default eval): the 1deg net uses BatchNorm. "eval" keeps the
    checkpoint running stats (what the inference scripts use); "train" = PyTorch default.
Time units below the original docstring say 10 days per step: for 1deg it is 5.
=======================================================================================

The deterministic SamudraV2 ConvNeXt U-Net is turned into a calibrated
ensemble system: each training iteration builds an M-member ensemble by
adding 3-octave Perlin noise to the normalized IC (the same `perlin_3octaves`
mechanism used by `samudra_om4_long_ensemble_rollout.py`), rolls all members
forward `--rollout-steps` autoregressive steps, and minimizes the *fair*
(unbiased) ensemble CRPS against the OM4 truth, latitude-weighted and
land-masked, averaged over steps.

Ensemble-parallel layout (one member — or `--members-per-rank` — per GPU):
  * every rank runs the same IC with its own Perlin members,
  * per step, member predictions are all-gathered (detached, bf16) so each
    rank can form the CRPS spread term against the full ensemble,
  * the per-rank surrogate loss (`crps_loss.fair_crps_surrogate`) has exactly
    the full fair-CRPS gradient w.r.t. the local members,
  * gradients are manually all-reduced (no DDP wrapper, so the U-Net's
    activation checkpointing needs no static-graph workarounds).

Checkpoints are saved as {"model": state_dict} so the existing rollout
scripts load them unchanged (point --ckpt-path at ckpt_latest.pt and keep
--config-path at the original SamudraV2 config.yaml).

Time units: 1 model step = (hist+1) = 2 data timesteps = 10 days.

Usage (single node, 4 GPUs = 4 ensemble members):
    torchrun --standalone --nproc_per_node=4 \
        src/CRPS/1-IC_noise.py \
        --iters 1000 --rollout-steps 4 --perturbation-level 0.025

Debug (1 GPU, 2 members in-process):
    python src/CRPS/1-IC_noise.py \
        --members-per-rank 2 --iters 3 --rollout-steps 1 --val-every 0
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]                    # 1deg: .../InMOS_1degree
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "CRPS"))

from ocean_emulators.config import EvalConfig, TimeConfig
from samudra_1deg import (  # 1deg: replaces samudra_om4_multi_ic_rollout / _ensemble_rollout
    BN_MODES, DEFAULT_CKPT_PATH, DEFAULT_CONFIG_PATH, DEFAULT_DATA_ROOT, DEFAULT_OUT_ROOT,
    DEFAULT_SURFACE_VARS, InferenceDataset, load_model_and_data, perlin_3octaves, set_bn_mode,
)

from crps_loss import (
    fair_crps, fair_crps_decomposed, fair_crps_surrogate,
    make_loss_weights, weighted_mean, weighted_mean_per_channel,
)

logger = logging.getLogger("crps_finetune")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--ckpt-path",   type=Path, default=DEFAULT_CKPT_PATH,
                   help="Initial weights (default: the deterministic SamudraV2 EMA ckpt).")
    p.add_argument("--data-root",   type=Path, default=DEFAULT_DATA_ROOT)
    # Windows default to the SamudraV2 training config's train/val split.
    p.add_argument("--train-start", default="1975-01-03")
    p.add_argument("--train-end",   default="2013-10-05")
    p.add_argument("--val-start",   default="2013-10-05")
    p.add_argument("--val-end",     default="2014-10-05")
    # Ensemble / rollout geometry
    p.add_argument("--members-per-rank", type=int, default=1,
                   help="Ensemble members simulated per GPU. Global ensemble size "
                        "M = world_size * members_per_rank (M >= 2 required).")
    p.add_argument("--ics-per-iter", type=int, default=4,  # 1deg
                   help="Batch size: distinct ICs per iteration (each with M members, one forward "
                        "pass of ics_per_iter*members_per_rank samples); loss = mean over ICs. "
                        "1 = the original one-IC-per-iteration run.")
    p.add_argument("--rollout-steps", type=int, default=8,  # 1deg: 8 x 5 d = 40 d
                   help="Autoregressive steps per training iteration; CRPS is "
                        "averaged over all steps (1deg: 1 step = 5 days).")
    p.add_argument("--perturbation-level", type=float, default=0.025,
                   help="Perlin sigma on the normalized IC (matches the "
                        "long_perturb inference ensembles).")
    p.add_argument("--per-step-noise", type=float, default=0.0,
                   help="Perlin sigma added to each member's prognostic state "
                        "at every rollout step AFTER the first (model-error "
                        "representation, SPPT-like). 0 disables. Inference must "
                        "then inject the same noise.")
    p.add_argument("--dynamic-loss", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="DynamicLoss-style per-variable weights (rolling 1/CRPS, "
                        "window 25) so tracers are not drowned out by uo/vo. "
                        "Disable with --no-dynamic-loss.")
    p.add_argument("--dynamic-limit", type=float, default=20.0,
                   help="Max/min ratio allowed across the dynamic per-variable "
                        "weights (same semantics as the SamudraV2 train config).")
    p.add_argument("--lat-weight", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="cos(lat)-weight the CRPS loss and val metrics. Default "
                        "OFF: grid-uniform over wet points, matching the "
                        "original SamudraV2 training loss.")
    p.add_argument("--bn-mode", choices=BN_MODES, default="sample",  # 1deg
                   help="BatchNorm handling: sample (default) = per-sample statistics, as the net "
                        "was trained and as the inference scripts run; eval = frozen checkpoint "
                        "running stats (unstable in long rollouts); train = batch stats + updates.")
    p.add_argument("--detach-interval", type=int, default=0,
                   help="Detach the autoregressive state every N steps "
                        "(0 = full backprop through all rollout steps).")
    # Optimization
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-iters", type=int, default=50)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--trainable-regex", default=None,
                   help="Optional regex; only matching parameter names are trained "
                        "(e.g. 'decoder' for the final conv only). Default: all.")
    p.add_argument("--ema-decay", type=float, default=0.99,
                   help="EMA decay for a shadow copy of the trainable weights "
                        "(0 disables). Validation and the *_ema.pt checkpoints "
                        "use the EMA weights. Keep the effective window "
                        "1/(1-decay) well below --iters (0.99 -> ~100 iters).")
    # Validation / checkpointing / logging
    p.add_argument("--prefetch-workers", type=int, default=4,
                   help="Thread-pool size for prefetching the NEXT iteration's "
                        "zarr reads (IC + per-step targets/boundaries) while the "
                        "GPU computes the current one. 0 = synchronous reads.")
    p.add_argument("--val-every",  type=int, default=100,
                   help="Run ensemble validation every N iters (0 = never).")
    p.add_argument("--rmse-tolerance", type=float, default=0.05,
                   help="ckpt_best gate: validation CRPS must improve on the "
                        "pre-finetune baseline AND ensemble-mean RMSE must stay "
                        "within (1 + this) * baseline RMSE.")
    p.add_argument("--val-ics",    type=int, default=2)
    p.add_argument("--val-steps",  type=int, default=16)  # 1deg: 16 x 5 d = 80 d
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--log-every",  type=int, default=1)
    p.add_argument("--out-dir", type=Path,
                   default=DEFAULT_OUT_ROOT)  # 1deg: ONE folder for all CRPS stages
    p.add_argument("--ckpt-prefix", default="1-IC_noise",  # 1deg
                   help="File-name prefix of everything this stage writes into --out-dir.")
    p.add_argument("--resume", type=Path, default=None,
                   help="Resume from a ckpt saved by this script (model+optimizer+iter).")
    p.add_argument("--seed", type=int, default=2025)
    return p.parse_args()


# =====================================================================
# Distributed helpers
# =====================================================================
def setup_distributed() -> tuple[int, int, torch.device]:
    """Returns (rank, world_size, device). Uses torchrun env vars if present."""
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, world, torch.device(f"cuda:{local_rank}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, device


def gather_ensemble(y: torch.Tensor, world: int) -> torch.Tensor:
    """All-gather member predictions (detached, bf16) -> (M, C, H, W).

    Rank order defines the global member order; single-process mode returns
    the local members unchanged.
    """
    y_det = y.detach().to(torch.bfloat16)
    if world == 1:
        return y_det
    bufs = [torch.empty_like(y_det) for _ in range(world)]
    dist.all_gather(bufs, y_det)
    return torch.cat(bufs, dim=0)


def allreduce_grads(params: list[torch.nn.Parameter], world: int) -> None:
    if world == 1:
        return
    handles = []
    for p in params:
        if p.grad is not None:
            handles.append(dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True))
    for h in handles:
        h.wait()
    for p in params:
        if p.grad is not None:
            p.grad.div_(world)


# =====================================================================
# Ensemble construction and rollout pieces
# =====================================================================
def perturbed_members(current: torch.Tensor, global_members: list[int],
                      sigma: float, seed0: int) -> torch.Tensor:
    """Perlin-perturbed copies of the (1, C, H, W) IC -> (m_local, C, H, W).

    Unlike inference (member 0 = control), *all* training members are
    perturbed so the ensemble is exchangeable, as fair CRPS assumes.
    """
    H, W = current.shape[-2:]
    members = []
    for g in global_members:
        noise = perlin_3octaves((H, W), seeds=[seed0 + g * 3 + k for k in range(3)])
        noise_t = torch.from_numpy(noise).to(current.device, dtype=current.dtype)
        members.append(current[0] + sigma * noise_t)
    return torch.stack(members, dim=0)


def get_boundary(ds: InferenceDataset, step: int) -> torch.Tensor:
    """Just the (1, (hist+1)*n_bound, H, W) boundary channels for one step.

    `merge_prognostic_and_boundary` cannot broadcast over members (and would
    materialize a full merged copy), so pull only the boundary channels.
    """
    return ds._get_boundary(ds._get_x_index(step))


def build_input(state: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
    """(m, num_out, H, W) prognostic + (1, Cb, H, W) boundary -> model input."""
    boundary = boundary.to(state.device)
    return torch.cat([state, boundary.expand(state.shape[0], -1, -1, -1)], dim=1)


class Prefetcher:
    """Thread-pool prefetch of one training iteration's zarr reads.

    The IC sequence is deterministic (seed + iter), so while the GPU computes
    iteration N we read iteration N+1's IC prognostic, per-step targets and
    boundaries in background threads. zarr I/O and blosc decompression release
    the GIL, so the overlap is real. Holds at most ~2 iterations of data in
    host RAM (~6 GB/rank at 0.25 deg, K=4).
    """

    def __init__(self, ds: InferenceDataset, rollout_steps: int, workers: int):
        self.ds = ds
        self.K = rollout_steps
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        self.pending: dict[int, dict] = {}

    def _submit(self, ic: int) -> dict:
        ds = self.ds
        return {
            # _get_prognostic skips the label read that ds[ic] would also do.
            "ic":  self.pool.submit(lambda: ds._get_prognostic(ds._get_x_index(ic))),
            "tgt": [self.pool.submit(lambda k=k: ds.inference_target(ic + k))
                    for k in range(self.K)],
            "bnd": [self.pool.submit(lambda k=k: get_boundary(ds, ic + k))
                    for k in range(self.K)],
        }

    def schedule(self, ic: int) -> None:
        if ic not in self.pending:
            self.pending[ic] = self._submit(ic)

    def get(self, ic: int) -> dict:
        futs = self.pending.pop(ic, None) or self._submit(ic)
        return {
            "ic":  futs["ic"].result(),
            "tgt": [f.result() for f in futs["tgt"]],
            "bnd": [f.result() for f in futs["bnd"]],
        }

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


def forward_members(model, x: torch.Tensor, ctx, num_out: int) -> torch.Tensor:
    y = model.forward_once(x, ctx)
    if model.pred_residuals:  # 1deg: config has pred_residuals: false; residual base would
        raise NotImplementedError("pred_residuals with a one-step model")  # be the latest slot
    return y


def advance_state(state: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """1deg: drop the oldest slot(s), append the prediction (== y for a 2-slot model)."""
    return torch.cat([state, y], dim=1)[:, -state.shape[1]:]


def add_state_noise(state: torch.Tensor, global_members: list[int],
                    sigma: float, seed0: int, step_k: int) -> torch.Tensor:
    """Per-member Perlin noise on the prognostic state at rollout step k > 0.

    Represents model error (SPPT-like): the same generator and per-member
    seeding as the IC perturbation, but re-drawn every step. Additive, so
    gradients pass through unchanged.
    """
    H, W = state.shape[-2:]
    noises = []
    for i, g in enumerate(global_members):
        seeds = [seed0 + g * 97 + step_k * 9973 + k for k in range(3)]
        noise = perlin_3octaves((H, W), seeds=seeds)
        noises.append(torch.from_numpy(noise))
    noise_t = torch.stack(noises).unsqueeze(1).to(state.device, dtype=state.dtype)
    return state + sigma * noise_t


class DynamicChannelScale:
    """Port of ocean_emulators.utils.loss.DynamicLoss weighting to fair CRPS.

    Keeps a rolling (window 25) estimate of 1/CRPS per *variable* (hist slots
    folded, like the original), clamps the max/min weight ratio to `limit`,
    and exposes the scale expanded back to the (hist+1)*n_vars channel layout.
    No all-reduce is needed: every rank computes the update from the same
    gathered full-ensemble CRPS, so the scales stay bit-identical.
    """

    N_WINDOW = 25

    def __init__(self, n_vars: int, hist: int, limit: float,
                 device: torch.device):
        self.n_vars = n_vars
        self.hist = hist
        self.limit = limit
        self.scale = torch.ones(n_vars, device=device)

    def channel_scale(self) -> torch.Tensor:
        """Scale in time-major channel order: (hist+1)*n_vars."""
        return self.scale.repeat(self.hist + 1)

    @torch.no_grad()
    def update(self, crps_per_channel: torch.Tensor) -> None:
        """crps_per_channel: ((hist+1)*n_vars,), detached, unscaled."""
        loss = torch.where(crps_per_channel <= 0, 1e-8, crps_per_channel)
        new = (1.0 / loss).reshape(self.hist + 1, self.n_vars).mean(dim=0)
        if self.limit is not None and self.limit > 0:
            lo = new.min()
            new = new.clamp(lo, lo * self.limit)
        self.scale = (self.scale * (self.N_WINDOW - 1) + new) / self.N_WINDOW

    def state_dict(self) -> dict:
        return {"scale": self.scale.detach().cpu()}

    def load_state_dict(self, state: dict) -> None:
        self.scale = state["scale"].to(self.scale.device)


# =====================================================================
# EMA of the trainable weights
# =====================================================================
@torch.no_grad()
def ema_update(ema: dict[str, torch.Tensor],
               named_trainable: list[tuple[str, torch.nn.Parameter]],
               decay: float) -> None:
    for name, p in named_trainable:
        ema[name].mul_(decay).add_(p.detach(), alpha=1.0 - decay)


@contextlib.contextmanager
def ema_weights(named_trainable, ema: dict[str, torch.Tensor] | None):
    """Temporarily load the EMA weights into the model (used for validation)."""
    if ema is None:
        yield
        return
    backup = {n: p.detach().clone() for n, p in named_trainable}
    with torch.no_grad():
        for n, p in named_trainable:
            p.copy_(ema[n])
    try:
        yield
    finally:
        with torch.no_grad():
            for n, p in named_trainable:
                p.copy_(backup[n])


# =====================================================================
# Validation: CRPS / SSR on held-out ICs
# =====================================================================
@torch.no_grad()
def validate(model, vds, args, rank, world, num_out, n_prog, hist,
             weights, prog_names, device, it, tag="") -> tuple[float, float]:
    """Ensemble validation. Returns (crps, rmse) for all channels, averaged
    over ICs and reported leads — identical on every rank (the gathered
    ensembles are identical), so gating decisions stay in sync."""
    model.eval()
    m_local = args.members_per_rank
    local_members = list(range(rank * m_local, (rank + 1) * m_local))
    M = world * m_local

    # Per-variable weights for the latest history slot of each surface var.
    var_channels = {}
    prog_index = {n: i for i, n in enumerate(prog_names)}
    out_hist = num_out // n_prog - 1  # 1deg: 0 (output = one slot)
    for v in DEFAULT_SURFACE_VARS:
        if v in prog_index:
            var_channels[v] = out_hist * n_prog + prog_index[v]

    n_windows = len(vds)
    if n_windows <= args.val_steps:
        logger.warning("val window too short for val-steps=%d — skipping", args.val_steps)
        model.train()
        return float("nan"), float("nan")
    ic_list = np.unique(np.linspace(0, n_windows - args.val_steps - 1,
                                    args.val_ics).astype(int))
    report_steps = sorted({1, max(1, args.val_steps // 2), args.val_steps})

    # accum[(var, step)] = [crps, rmse, spread_std] summed over ICs
    accum: dict[tuple[str, int], np.ndarray] = {}
    for ic in ic_list:
        seed0 = args.seed + 777_000 + int(ic) * 1000
        current = vds[int(ic)][0][:, :(hist + 1) * n_prog].to(device)  # 1deg: 154-ch input state
        state = perturbed_members(current, local_members,
                                  args.perturbation_level, seed0)
        for k in range(args.val_steps):
            if args.per_step_noise > 0 and k > 0:
                state = add_state_noise(state, local_members,
                                        args.per_step_noise, seed0, k)
            x = build_input(state, get_boundary(vds, int(ic) + k))
            y = forward_members(model, x, vds.ctx, num_out)
            state = advance_state(state, y)  # 1deg
            lead = k + 1
            if lead not in report_steps:
                continue
            y_true = vds.inference_target(int(ic) + k)[0].to(device)
            ens = gather_ensemble(y, world).to(torch.float32)  # 1deg: the 77-ch prediction
            crps_all, _, _ = fair_crps(ens, y_true, weights)
            ens_mean = ens.mean(dim=0)
            rmse_all = weighted_mean((ens_mean - y_true) ** 2, weights).sqrt()
            spread_all = weighted_mean(ens.var(dim=0, unbiased=True), weights).sqrt()
            key = ("all", lead)
            vals = np.array([crps_all.item(), rmse_all.item(), spread_all.item()])
            accum[key] = accum.get(key, 0) + vals
            for v, c in var_channels.items():
                w_c = weights[c:c + 1]
                crps_v, _, _ = fair_crps(ens[:, c:c + 1], y_true[c:c + 1], w_c)
                rmse_v = weighted_mean((ens_mean[c:c + 1] - y_true[c:c + 1]) ** 2,
                                       w_c).sqrt()
                spread_v = weighted_mean(ens[:, c:c + 1].var(dim=0, unbiased=True),
                                         w_c).sqrt()
                key = (v, lead)
                vals = np.array([crps_v.item(), rmse_v.item(), spread_v.item()])
                accum[key] = accum.get(key, 0) + vals

    n_ic = len(ic_list)
    metrics = {k: v / n_ic for k, v in accum.items()}
    if rank == 0:
        ssr_corr = np.sqrt((M + 1) / M)  # finite-ensemble SSR correction
        logger.info("[val%s @ iter %d] %d ICs, M=%d, normalized units",
                    f" {tag}" if tag else "", it, n_ic, M)
        for (v, lead), vals in sorted(metrics.items()):
            crps_m, rmse_m, spread_m = vals
            ssr = ssr_corr * spread_m / max(rmse_m, 1e-12)
            logger.info("  %-9s lead=%2d steps (%3dd)  CRPS=%.4f  RMSE=%.4f  "
                        "spread=%.4f  SSR=%.3f",
                        v, lead, lead * (num_out // n_prog) * 5, crps_m, rmse_m, spread_m, ssr)  # 1deg
    model.train()
    all_crps = float(np.mean([metrics[("all", ld)][0] for ld in report_steps]))
    all_rmse = float(np.mean([metrics[("all", ld)][1] for ld in report_steps]))
    return all_crps, all_rmse


# =====================================================================
# Main
# =====================================================================
def main() -> None:
    args = parse_args()
    rank, world, device = setup_distributed()
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    m_local = args.members_per_rank
    M = world * m_local
    if M < 2:
        raise SystemExit("Fair CRPS needs M >= 2: use torchrun with >= 2 ranks "
                         "or --members-per-rank >= 2.")
    local_members = list(range(rank * m_local, (rank + 1) * m_local))
    logger.info("world=%d members_per_rank=%d -> M=%d  device=%s",
                world, m_local, M, device)

    # ---- model + data (same loading path as the rollout scripts) ----------
    with open(args.config_path) as f:
        train_yaml = yaml.safe_load(f)
    cfg = EvalConfig(
        ckpt_path               = str(args.ckpt_path),
        num_model_steps_forward = args.rollout_steps,
        inference_time          = {"start": args.train_start, "end": args.train_end},
        experiment              = {
            **train_yaml["experiment"],
            "data_root": str(args.data_root),
            "name":      "samudra_crps_finetune",
            "wandb":     {"mode": "disabled", "project": "default"},
        },
        data  = train_yaml["data"],
        model = train_yaml["model"],
    )
    src, normalize, model, prog_names, bound_names, n_prog, num_out = \
        load_model_and_data(args, cfg, device)
    hist = cfg.data.hist
    set_bn_mode(model, args.bn_mode)  # 1deg

    def make_ds(start: str, end: str) -> InferenceDataset:
        sliced = src.slice(TimeConfig(start=start, end=end))
        return InferenceDataset(
            src=sliced, prognostic_var_names=prog_names,
            boundary_var_names=bound_names, hist=hist,
            normalize_before_mask=cfg.data.normalize_before_mask,
            masked_fill_value=cfg.data.masked_fill_value, long_rollout=False,
        ).to(device)

    tds = make_ds(args.train_start, args.train_end)
    vds = make_ds(args.val_start, args.val_end)
    logger.info("train windows=%d  val windows=%d  rollout_steps=%d (%d days)",
                len(tds), len(vds), args.rollout_steps,
                args.rollout_steps * (num_out // n_prog) * 5)  # 1deg

    lat = (torch.from_numpy(src.data.lat.values.astype(np.float32)).to(device)
           if args.lat_weight else None)
    weights = make_loss_weights(tds.ctx.label_mask, lat)
    logger.info("loss/val weighting: %s over wet points",
                "cos(lat)" if args.lat_weight else "grid-uniform")

    dyn = (DynamicChannelScale(n_prog, num_out // n_prog - 1, args.dynamic_limit, device)  # 1deg
           if args.dynamic_loss else None)
    if dyn is not None:
        logger.info("dynamic per-variable loss weights enabled (limit=%.0f)",
                    args.dynamic_limit)

    # ---- optimizer ---------------------------------------------------------
    if args.trainable_regex:
        import re
        pat = re.compile(args.trainable_regex)
        for name, p in model.named_parameters():
            p.requires_grad = bool(pat.search(name))
    named_trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    trainable = [p for _, p in named_trainable]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info("trainable params: %.1fM / %.1fM", n_train / 1e6, n_total / 1e6)
    if not trainable:
        raise SystemExit(f"--trainable-regex {args.trainable_regex!r} matched nothing.")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr,
                                  weight_decay=args.weight_decay)

    # EMA shadow starts at the pretrained weights (no zero-init bias).
    # Weights are identical on all ranks (same init + all-reduced grads),
    # so the shadow stays in sync without communication.
    ema = ({n: p.detach().clone() for n, p in named_trainable}
           if args.ema_decay > 0 else None)
    if ema is not None:
        logger.info("EMA enabled: decay=%.4f (~%d-iter window)",
                    args.ema_decay, round(1.0 / (1.0 - args.ema_decay)))

    start_iter = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt["iter"] + 1
        if ema is not None and ckpt.get("ema") is not None:
            for n in ema:
                ema[n].copy_(ckpt["ema"][n].to(device))
        if dyn is not None and ckpt.get("dynamic_scale") is not None:
            dyn.load_state_dict(ckpt["dynamic_scale"])
        logger.info("resumed from %s at iter %d", args.resume, start_iter)

    # Pre-finetune baseline for the ckpt_best gate (reused across --resume so
    # the reference stays the original pretrained model).
    baseline = ckpt.get("val_baseline") if args.resume is not None else None
    best_crps = ckpt.get("best_crps", float("inf")) if args.resume is not None \
        else float("inf")

    if rank == 0:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        # Keep a copy of the architecture config next to the new weights so
        # rollout scripts can be pointed at this directory alone.
        (args.out_dir / f"{args.ckpt_prefix}_config.yaml").write_text(Path(args.config_path).read_text())  # 1deg

    def save_ckpt(it: int, name: str) -> None:
        if rank != 0:
            return
        payload = {
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter":      it,
            "ema":       ema,
            "dynamic_scale": dyn.state_dict() if dyn is not None else None,
            "val_baseline":  baseline,
            "best_crps":     best_crps,
            "crps_finetune_args": {k: str(v) for k, v in vars(args).items()},
        }
        fname = name.replace("ckpt_", f"{args.ckpt_prefix}_", 1)  # 1deg: e.g. 1-IC_noise_best.pt
        torch.save(payload, args.out_dir / fname)
        logger.info("saved %s", args.out_dir / fname)
        if ema is not None:
            # Companion EMA-weights file in the plain {"model": ...} layout so
            # the rollout scripts can load it directly (no optimizer, ~1/3 size).
            ema_sd = {k: ema.get(k, v) for k, v in model.state_dict().items()}
            # 1deg: every gate-PASS keeps its own EMA file, never overwritten
            # (<prefix>_it<NNNN>_ema.pt); <prefix>_best_ema.pt is a symlink to the newest one.
            ema_name = f"{args.ckpt_prefix}_it{it:04d}_ema.pt" if name == "ckpt_best.pt" \
                else name.replace(".pt", "_ema.pt")
            torch.save({"model": ema_sd, "iter": it}, args.out_dir / ema_name)
            logger.info("saved %s", args.out_dir / ema_name)
            if name == "ckpt_best.pt":  # 1deg
                link = args.out_dir / f"{args.ckpt_prefix}_best_ema.pt"
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(ema_name)

    def run_val_and_gate(it: int) -> None:
        """Validate with EMA weights; update ckpt_best if the gate passes:
        CRPS better than both the baseline and the previous best, AND
        ensemble-mean RMSE within (1 + rmse_tolerance) * baseline RMSE."""
        nonlocal best_crps
        with ema_weights(named_trainable, ema):
            v_crps, v_rmse = validate(model, vds, args, rank, world, num_out,
                                      n_prog, hist, weights, prog_names,
                                      device, it)
        if baseline is None or np.isnan(v_crps):
            return
        rmse_limit = baseline["rmse"] * (1.0 + args.rmse_tolerance)
        if v_crps < baseline["crps"] and v_crps < best_crps and v_rmse <= rmse_limit:
            best_crps = v_crps
            save_ckpt(it, "ckpt_best.pt")
            if rank == 0:
                logger.info("gate PASS: CRPS=%.5f (baseline %.5f)  RMSE=%.5f "
                            "(limit %.5f) -> ckpt_best updated",
                            v_crps, baseline["crps"], v_rmse, rmse_limit)
        elif rank == 0:
            logger.info("gate SKIP: CRPS=%.5f (best %.5f, baseline %.5f)  "
                        "RMSE=%.5f (limit %.5f)",
                        v_crps, best_crps, baseline["crps"], v_rmse, rmse_limit)

    # ---- training loop -----------------------------------------------------
    model.train()
    max_ic = len(tds) - args.rollout_steps
    if max_ic < 1:
        raise SystemExit("train window shorter than rollout-steps")

    if args.val_every > 0 and baseline is None:
        # EMA == raw pretrained weights at iter 0, so this scores the
        # unmodified SamudraV2 under the exact ensemble protocol used later.
        with ema_weights(named_trainable, ema):
            b_crps, b_rmse = validate(model, vds, args, rank, world, num_out,
                                      n_prog, hist, weights, prog_names,
                                      device, start_iter, tag="baseline")
        if not np.isnan(b_crps):
            baseline = {"crps": b_crps, "rmse": b_rmse}
            if rank == 0:
                logger.info("baseline set: CRPS=%.5f RMSE=%.5f  (gate: CRPS < "
                            "baseline and RMSE <= %.2fx baseline)",
                            b_crps, b_rmse, 1.0 + args.rmse_tolerance)

    # Same IC on every rank (identical rng sequence) -> deterministic and
    # therefore prefetchable one iteration ahead.
    B = args.ics_per_iter  # 1deg: ICs per iteration (batch size)

    def ics_of(it: int) -> list[int]:
        """B distinct ICs, identical on every rank; B=1 reproduces the original sequence."""
        rng = np.random.default_rng(args.seed + it)
        if B == 1:
            return [int(rng.integers(0, max_ic + 1))]
        return [int(v) for v in rng.choice(max_ic + 1, size=B, replace=False)]

    pre = (Prefetcher(tds, args.rollout_steps, args.prefetch_workers)
           if args.prefetch_workers > 0 else None)
    if pre is not None:
        for _ic in ics_of(start_iter):
            pre.schedule(_ic)

    t_start = time.perf_counter()

    for it in range(start_iter, args.iters):
        ics = ics_of(it)  # 1deg: batch of B ICs
        seed0 = args.seed + it * 7919
        seed0s = [seed0 + b * 104729 for b in range(B)]  # b=0 -> the original seed  # distinct Perlin seeds per global member

        if pre is not None:
            if it + 1 < args.iters:
                for _ic in ics_of(it + 1):  # overlaps with this iter's compute
                    pre.schedule(_ic)
            datas = [pre.get(ic) for ic in ics]
        else:
            datas = [{"ic":  tds._get_prognostic(tds._get_x_index(ic)),
                      "tgt": [tds.inference_target(ic + k)
                              for k in range(args.rollout_steps)],
                      "bnd": [get_boundary(tds, ic + k)
                              for k in range(args.rollout_steps)]} for ic in ics]

        for g in optimizer.param_groups:  # linear warmup, then constant
            g["lr"] = args.lr * min(1.0, (it + 1) / max(1, args.warmup_iters))

        # 1deg: (B*m_local, ...) IC-major; each IC gets its own Perlin seeds
        state = torch.cat([perturbed_members(d["ic"].to(device), local_members,
                                             args.perturbation_level, seed0s[b])
                           for b, d in enumerate(datas)])

        # Segmented backward: at every --detach-interval boundary the segment
        # loss is backpropagated immediately and the autoregressive state is
        # detached, so peak memory holds only one segment's graph (gradients
        # accumulate across segments; the objective is truncated BPTT).
        # detach_interval=0 keeps one segment = full BPTT over all steps.
        optimizer.zero_grad(set_to_none=True)
        seg_loss = 0.0
        log_crps = log_skill = log_spread = 0.0
        crps_c_acc = torch.zeros(num_out, device=device)
        for k in range(args.rollout_steps):
            if args.per_step_noise > 0 and k > 0:  # 1deg: per IC
                state = torch.cat([add_state_noise(state[b * m_local:(b + 1) * m_local], local_members,
                                                   args.per_step_noise, seed0s[b], k) for b in range(B)])
            # 1deg: B ICs x m_local members in ONE batch (IC-major), each IC with its own forcing
            bnd = torch.cat([datas[b]["bnd"][k].to(device).expand(m_local, -1, -1, -1) for b in range(B)])
            x = torch.cat([state, bnd], dim=1)
            del bnd
            y = forward_members(model, x, tds.ctx, num_out)
            del x
            full_all = gather_ensemble(y, world)  # (world*B*m_local, ...), rank-major
            loss_k = 0.0
            for b in range(B):  # 1deg: fair CRPS per IC ensemble, averaged over the B ICs
                y_b = y[b * m_local:(b + 1) * m_local]
                fb = full_all.view(world, B, m_local, *y.shape[1:])[:, b]
                full = fb.reshape(world * m_local, *y.shape[1:])
                remote = (torch.cat([fb[r] for r in range(world) if r != rank])
                          if world > 1 else None)
                y_true = datas[b]["tgt"][k][0].to(device)
                loss_k = loss_k + fair_crps_surrogate(
                    y_b, remote, y_true, M, weights,
                    channel_scale=dyn.channel_scale() if dyn is not None else None,
                ) / B

                with torch.no_grad():
                    # Unscaled diagnostics from the gathered ensemble; also feeds the
                    # dynamic weight update (identical on all ranks).
                    skill_f, spread_f = fair_crps_decomposed(full.to(torch.float32), y_true)
                    crps_f = skill_f - spread_f
                    nrm = args.rollout_steps * B
                    crps_c_acc += weighted_mean_per_channel(crps_f, weights) / nrm
                    log_crps += weighted_mean(crps_f, weights).item() / nrm
                    log_skill += weighted_mean(skill_f, weights).item() / nrm
                    log_spread += weighted_mean(spread_f, weights).item() / nrm
                    del skill_f, spread_f, crps_f
                del full, remote, y_true
            seg_loss = seg_loss + loss_k / args.rollout_steps
            del full_all

            last = k == args.rollout_steps - 1
            boundary = (args.detach_interval > 0
                        and (k + 1) % args.detach_interval == 0)
            if last or boundary:
                seg_loss.backward()
                seg_loss = 0.0
                state = advance_state(state, y).detach()  # 1deg
            else:
                state = advance_state(state, y)  # 1deg

        allreduce_grads(trainable, world)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        if ema is not None:
            ema_update(ema, named_trainable, args.ema_decay)
        if dyn is not None:
            dyn.update(crps_c_acc)

        if rank == 0 and (it % args.log_every == 0 or it == args.iters - 1):
            elapsed = time.perf_counter() - t_start
            dyn_str = ("  dynw=[%.2f,%.2f]" % (dyn.scale.min(), dyn.scale.max())
                       if dyn is not None else "")
            logger.info(
                "iter %5d  ic=%-5d  CRPS=%.5f  skill=%.5f  spread=%.5f  "
                "lr=%.2e  %.1fs/it%s",
                it, ics[0], log_crps, log_skill, log_spread,
                optimizer.param_groups[0]["lr"],
                elapsed / max(1, it - start_iter + 1), dyn_str,
            )

        if args.val_every > 0 and (it + 1) % args.val_every == 0:
            run_val_and_gate(it)
        # 1deg: no unconditional ckpt_latest saves -- only gate-PASS checkpoints are
        # written (full resume state in ckpt_best.pt = the last PASS). --save-every is unused.

    if pre is not None:
        pre.shutdown()
    if args.val_every > 0 and args.iters % args.val_every != 0:
        run_val_and_gate(args.iters - 1)
    logger.info("done: %d iters in %.1f min",
                args.iters - start_iter, (time.perf_counter() - t_start) / 60)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
