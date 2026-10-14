"""Fair-CRPS fine-tuning of a *stochastic* SamudraV2 (IC noise + FiLM noise).

=== InMOS_1degree copy (2026-10-12) -- STAGE 2: FiLM fine-tune ===========================
Minimal-diff copy of Sampling/Derecho src/CRPS/train_finetune_samudra_crps_film.py (renamed 2-FiLM_noise.py) for
the 1deg checkpoint; every changed line is marked `1deg:`. Internally two phases, as in
the original: 2a = iters < --film-warmup-iters (backbone frozen, FiLM heads only),
2b = afterwards (backbone at --lr joins). Starts from the STAGE-1 output
(/data/weidong/InMOS_1deg/checkpoints/SamudraV2_1deg_CRPS/ckpt_best_ema.pt), writes to
.../SamudraV2_1deg_CRPS_FiLM. 1deg changes as in stage 1 (one-step state update,
77-ch loss/val, --rollout-steps 8, --val-steps 16, --bn-mode eval|train), plus
--rmse-abs-max default 0 (0.29225 was an absolute RMSE of the 0.25deg pretrained model).
FiLM heads sit after every norm layer (_NORM_TYPES includes BatchNorm2d); with the 1deg
BatchNorm they follow the BN affine instead of filling an empty affine slot, and the
zero-init identity still holds. --z-field-hw keeps the 0.25deg default (45, 90).
==========================================================================================

Sibling of `1-IC_noise.py` (stage 1, formerly train_finetune_samudra_crps.py). Everything about the CRPS
objective, the ensemble-parallel layout, the dynamic per-variable weights,
EMA and the gated checkpoints is imported from that script unchanged; what
this one adds is a second, *learned* source of ensemble spread.

    deterministic run : y_i = f(x + eta_i)                 eta = Perlin IC noise
    this run          : y_i = f(x + eta_i, z_i)            z   = latent noise
                                                                 consumed by FiLM
                                                                 on the InstanceNorm
                                                                 slots (see
                                                                 stochastic_samudra.py)

Both sources are active. The IC term keeps representing initial-condition
uncertainty; the FiLM term gives the model a channel through which fair CRPS
can learn *model* uncertainty — a flow-dependent, spatially structured
perturbation whose amplitude is calibrated by the loss instead of hand-tuned.

`z` is redrawn at every rollout step, i.e. white in time: the standard
choice for a model-error term. (A single draw held fixed for the whole
rollout would instead represent perturbed-parameter uncertainty; that is a
different experiment, not a flag.)

Zero-initialized FiLM means iteration 0 is bitwise the pretrained
deterministic model, so the validation baseline and the `ckpt_best` gate
remain directly comparable with the non-FiLM run. `assert_zero_init_identity`
enforces this before training starts; pass `--no-self-test` to skip it.

Training schedule: FiLM heads start from scratch while the backbone is
already converged, so they get their own (much larger) learning rate, and
for the first `--film-warmup-iters` iterations the backbone is frozen and
only the heads move. This gets spread off the ground before the backbone is
allowed to react to it.

IMPORTANT — inference: checkpoints from this script contain FiLM weights.
A rollout script must call `make_stochastic(model, **ckpt["film_config"])`
*before* `load_state_dict`, otherwise the FiLM weights are silently dropped
and you get the deterministic model back. The config is stored in every
checkpoint payload under "film_config".

Usage (single node, 4 GPUs = 4 ensemble members):
    torchrun --standalone --nproc_per_node=4 \
        src/CRPS/2-FiLM_noise.py \
        --iters 1500 --rollout-steps 4 --perturbation-level 0.025

Debug (1 GPU, 2 members in-process):
    python src/CRPS/2-FiLM_noise.py \
        --members-per-rank 2 --iters 3 --rollout-steps 1 --val-every 0
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]                    # 1deg: .../InMOS_1degree
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "CRPS"))

from ocean_emulators.config import EvalConfig, TimeConfig
from samudra_1deg import (  # 1deg: replaces samudra_om4_multi_ic_rollout
    BN_MODES, DEFAULT_CONFIG_PATH, DEFAULT_DATA_ROOT, DEFAULT_OUT_ROOT, DEFAULT_SURFACE_VARS,
    InferenceDataset, load_model_and_data, set_bn_mode,
)

from crps_loss import (
    fair_crps, fair_crps_decomposed, fair_crps_surrogate,
    make_loss_weights, weighted_mean, weighted_mean_per_channel,
)

# Reused verbatim from the deterministic CRPS fine-tuning script (stage 1).
# 1deg: stage 1 lives in 1-IC_noise.py (renamed 2026-10-12; not an importable module
# name), so load it by path. Same objects as the former `train_finetune_samudra_crps`.
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location("ic_noise_stage1", Path(__file__).with_name("1-IC_noise.py"))
_stage1 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_stage1)
(DynamicChannelScale, Prefetcher, add_state_noise, advance_state, allreduce_grads,  # 1deg: advance_state
 build_input, ema_update, ema_weights, gather_ensemble, get_boundary,
 perturbed_members, setup_distributed) = (
    _stage1.DynamicChannelScale, _stage1.Prefetcher, _stage1.add_state_noise, _stage1.advance_state,
    _stage1.allreduce_grads, _stage1.build_input, _stage1.ema_update, _stage1.ema_weights,
    _stage1.gather_ensemble, _stage1.get_boundary, _stage1.perturbed_members, _stage1.setup_distributed)

from stochastic_samudra import (
    WHERE_CHOICES, assert_zero_init_identity, film_parameters,
    film_weight_norm, make_stochastic, sample_z,
)

logger = logging.getLogger("crps_finetune_film")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--ckpt-path",   type=Path,
                   default=DEFAULT_OUT_ROOT / "1-IC_noise_best_ema.pt",  # 1deg: stage-1 output (newest PASS)
                   help="Initial weights. Default: the CRPS-tuned deterministic "
                        "model, so the IC-uncertainty term is already calibrated "
                        "and FiLM learns the residual (= model uncertainty). "
                        "Pass the plain SamudraV2 EMA ckpt to instead learn both "
                        "at once — faster to state, but the two spread sources "
                        "are then not separately identifiable.")
    p.add_argument("--data-root",   type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--train-start", default="1975-01-03")
    p.add_argument("--train-end",   default="2013-10-05")
    p.add_argument("--val-start",   default="2013-10-05")
    p.add_argument("--val-end",     default="2014-10-05")
    # ---- ensemble / rollout geometry ----
    p.add_argument("--members-per-rank", type=int, default=1,
                   help="Members simulated per GPU; M = world_size * this (>= 2).")
    p.add_argument("--ics-per-iter", type=int, default=4,  # 1deg
                   help="Batch size: distinct ICs per iteration (each with M members, one forward "
                        "pass of ics_per_iter*members_per_rank samples); loss = mean over ICs. "
                        "1 = the original one-IC-per-iteration run.")
    p.add_argument("--rollout-steps", type=int, default=8,  # 1deg: 8 x 5 d = 40 d
                   help="Autoregressive steps per iteration (1 step = 10 days).")
    p.add_argument("--perturbation-level", type=float, default=0.025,
                   help="Perlin sigma on the normalized IC (source #1: IC noise).")
    p.add_argument("--per-step-noise", type=float, default=0.0,
                   help="Additive Perlin sigma on the state at every step after "
                        "the first. Independent of the FiLM noise; leave at 0 "
                        "unless you deliberately want a hand-tuned SPPT term too.")
    p.add_argument("--bn-mode", choices=BN_MODES, default="sample",  # 1deg
                   help="BatchNorm handling: sample (default) = per-sample statistics, as the net "
                        "was trained and as the inference scripts run; eval = frozen checkpoint "
                        "running stats (unstable in long rollouts); train = batch stats + updates.")
    p.add_argument("--detach-interval", type=int, default=1,
                   help="Backward + detach the state every N steps (truncated "
                        "BPTT). 0 = full backprop through all rollout steps.")
    # ---- FiLM / latent noise (source #2: model noise) ----
    p.add_argument("--z-dim", type=int, default=64,
                   help="Dimension of the global latent vector driving FiLM.")
    p.add_argument("--z-field-ch", type=int, default=4,
                   help="Channels of the coarse random field giving FiLM its "
                        "spatial structure. 0 = per-channel (globally uniform) "
                        "modulation only, which is far too weak on its own.")
    p.add_argument("--z-field-hw", type=int, nargs=2, default=(45, 90),
                   metavar=("H", "W"),
                   help="Resolution of that field. The grid spacing sets the "
                        "correlation length of the perturbation; 45x90 on the "
                        "0.25-deg grid matches the U-Net bottleneck.")
    p.add_argument("--film-hidden", type=int, default=128,
                   help="Hidden width of each FiLM head's MLP.")
    p.add_argument("--film-where", choices=WHERE_CHOICES, default="bottleneck",
                   help="Which ConvNeXt blocks get FiLM. 'bottleneck' (middle + "
                        "decoder) injects large-scale coherent perturbations; "
                        "'all' also perturbs the encoder's fine scales.")
    p.add_argument("--film-lr", type=float, default=1e-3,
                   help="Learning rate for the (from-scratch) FiLM heads.")
    p.add_argument("--film-warmup-iters", type=int, default=800,
                   help="Iterations with the backbone frozen, training the FiLM "
                        "heads only. This is the primary RMSE protection: with "
                        "the backbone fixed the deterministic map is preserved "
                        "exactly, so ensemble-mean RMSE can only move by the "
                        "second-order effect of the perturbation itself. Set it "
                        ">= --iters to never unfreeze. 0 trains everything from "
                        "the start.")
    p.add_argument("--self-test", action=argparse.BooleanOptionalAction, default=True,
                   help="Assert that zero-initialized FiLM reproduces the "
                        "deterministic model bitwise before training.")
    # ---- loss ----
    p.add_argument("--crps-alpha", type=float, default=0.95,
                   help="almost-fair CRPS: alpha*fCRPS + (1-alpha)*CRPS, as in "
                        "AIFS-CRPS. The standard term rewards spread slightly "
                        "less, trading a little calibration for ensemble-mean "
                        "RMSE and removing the small-M degeneracy of pure fair "
                        "CRPS in bf16. 1.0 = the pure fair CRPS of the previous "
                        "run. Affects the LOSS only; validation metrics stay "
                        "pure fair CRPS so they remain comparable.")
    p.add_argument("--dynamic-loss", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Rolling 1/CRPS per-variable weights (window 25).")
    p.add_argument("--dynamic-limit", type=float, default=20.0)
    p.add_argument("--lat-weight", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="cos(lat)-weight the loss and val metrics. Default OFF: "
                        "grid-uniform over wet points, as in SamudraV2 training.")
    # ---- optimization ----
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--lr", type=float, default=5e-6,
                   help="Backbone learning rate. Half the deterministic run's "
                        "1e-5: that run drifted RMSE +3.6%% at 1e-5 and this one "
                        "starts from its endpoint, so the drift budgets compound.")
    p.add_argument("--warmup-iters", type=int, default=50)
    p.add_argument("--reset-base-lr", action="store_true",
                   help="On --resume, re-apply --lr/--film-lr instead of "
                        "inheriting the base_lr stored in the checkpoint's "
                        "optimizer state. Needed after --rmse-brake has cut the "
                        "backbone lr: the brake writes base_lr into the "
                        "optimizer, so a plain resume silently continues at the "
                        "braked rate and the command-line --lr is ignored.")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.99)
    # ---- validation / checkpointing / logging ----
    p.add_argument("--prefetch-workers", type=int, default=4)
    p.add_argument("--val-every",  type=int, default=100)
    # ---- ckpt_best gate: four thresholds, all must pass ----
    p.add_argument("--rmse-tolerance", type=float, default=0.03,
                   help="Ensemble-mean RMSE must stay within (1 + this) x the "
                        "baseline. Tighter than the deterministic run's 0.05 "
                        "because the baseline is already +3.6%% over the "
                        "pretrained model — the budgets compound.")
    p.add_argument("--rmse-abs-max", type=float, default=0.0,  # 1deg: 0.29225 was a 0.25deg anchor
                   help="ABSOLUTE ens-mean RMSE ceiling, in normalized units; "
                        "the effective limit is min(this, relative limit). A "
                        "purely relative tolerance compounds: the baseline here "
                        "is already +3.6%% over the pretrained SamudraV2, so "
                        "+3%% on top of it (0.29703) is a LOOSER absolute "
                        "ceiling than the previous run's +5%% on the pretrained "
                        "model (0.27833 * 1.05 = 0.29225). Anchoring to that "
                        "same absolute number keeps successive fine-tunes from "
                        "ratcheting deterministic skill away. 0 disables.")
    p.add_argument("--min-crps-gain", type=float, default=0.005,
                   help="CRPS must beat the baseline by at least this *relative* "
                        "margin (0.005 = 0.5%%). Stops ckpt_best churning on "
                        "noise-level improvements.")
    p.add_argument("--rmse-brake", type=float, default=0.5,
                   help="When a validation fails the RMSE threshold, multiply "
                        "the BACKBONE learning rate by this and carry on — the "
                        "closed-loop version of the manual rule in the original CRPS/HANDOFF.md "
                        "('RMSE gate SKIP -> drop lr, resume from ckpt_best'). "
                        "The FiLM lr is untouched: FiLM is not what damages "
                        "RMSE, backbone drift is. 1.0 disables.")
    p.add_argument("--ssr-max", type=float, default=1.25,
                   help="Reject if the all-channel SSR at ANY reported lead "
                        "exceeds this: over-dispersion is the new failure mode "
                        "once a second noise source is stacked on IC noise. "
                        "0 disables. (Under-dispersion needs no threshold — "
                        "CRPS already penalizes it.)")
    p.add_argument("--val-ics",    type=int, default=2)
    p.add_argument("--val-steps",  type=int, default=16)  # 1deg: 16 x 5 d = 80 d
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--log-every",  type=int, default=1)
    p.add_argument("--out-dir", type=Path,
                   default=DEFAULT_OUT_ROOT)  # 1deg: ONE folder for all CRPS stages
    p.add_argument("--ckpt-prefix", default="2-FiLM_noise",  # 1deg
                   help="File-name prefix of everything this stage writes into --out-dir.")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--seed", type=int, default=2025)
    return p.parse_args()


def forward_members(model, x: torch.Tensor, ctx, num_out: int, z) -> torch.Tensor:
    """Ensemble forward with an explicit latent draw."""
    y = model.forward_once(x, ctx, z=z)
    if model.pred_residuals:  # 1deg: config has pred_residuals: false
        raise NotImplementedError("pred_residuals with a one-step model")
    return y


@torch.no_grad()
def validate(model, vds, args, rank, world, num_out, n_prog, hist,
             weights, prog_names, device, it, tag="") -> tuple[float, float, float]:
    """Ensemble validation -> (crps, rmse, ssr_max), identical on every rank.

    Same protocol as `train_finetune_samudra_crps.validate` (same ICs, same
    Perlin seeds, same reported leads), so the baseline of a run started from
    the deterministic CRPS checkpoint must reproduce that run's final numbers.
    It additionally returns the worst-case SSR over the reported leads, which
    the gate uses to reject over-dispersed models — a failure mode the
    deterministic run could not have.

    `model.forward_once(x, ctx)` is called without a z, so the model's own
    auto-z generator draws one per member per step.
    """
    model.eval()
    m_local = args.members_per_rank
    local_members = list(range(rank * m_local, (rank + 1) * m_local))
    M = world * m_local

    var_channels = {}
    prog_index = {n: i for i, n in enumerate(prog_names)}
    for v in DEFAULT_SURFACE_VARS:
        if v in prog_index:
            var_channels[v] = (num_out // n_prog - 1) * n_prog + prog_index[v]  # 1deg: output slot

    n_windows = len(vds)
    if n_windows <= args.val_steps:
        logger.warning("val window too short for val-steps=%d — skipping", args.val_steps)
        model.train()
        return float("nan"), float("nan"), float("nan")
    ic_list = np.unique(np.linspace(0, n_windows - args.val_steps - 1,
                                    args.val_ics).astype(int))
    report_steps = sorted({1, max(1, args.val_steps // 2), args.val_steps})

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
            y = forward_members(model, x, vds.ctx, num_out, z=None)
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
    ssr_corr = np.sqrt((M + 1) / M)  # finite-ensemble SSR correction
    if rank == 0:
        logger.info("[val%s @ iter %d] %d ICs, M=%d, normalized units",
                    f" {tag}" if tag else "", it, n_ic, M)
        for (v, lead), vals in sorted(metrics.items()):
            crps_m, rmse_m, spread_m = vals
            logger.info("  %-9s lead=%2d steps (%3dd)  CRPS=%.4f  RMSE=%.4f  "
                        "spread=%.4f  SSR=%.3f",
                        v, lead, lead * (num_out // n_prog) * 5, crps_m, rmse_m, spread_m,  # 1deg
                        ssr_corr * spread_m / max(rmse_m, 1e-12))
    model.train()
    all_crps = float(np.mean([metrics[("all", ld)][0] for ld in report_steps]))
    all_rmse = float(np.mean([metrics[("all", ld)][1] for ld in report_steps]))
    ssr_max = float(max(
        ssr_corr * metrics[("all", ld)][2] / max(metrics[("all", ld)][1], 1e-12)
        for ld in report_steps
    ))
    return all_crps, all_rmse, ssr_max


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

    # ---- model + data ------------------------------------------------------
    with open(args.config_path) as f:
        train_yaml = yaml.safe_load(f)
    cfg = EvalConfig(
        ckpt_path               = str(args.ckpt_path),
        num_model_steps_forward = args.rollout_steps,
        inference_time          = {"start": args.train_start, "end": args.train_end},
        experiment              = {
            **train_yaml["experiment"],
            "data_root": str(args.data_root),
            "name":      "samudra_crps_film_finetune",
            "wandb":     {"mode": "disabled", "project": "default"},
        },
        data  = train_yaml["data"],
        model = train_yaml["model"],
    )
    src, normalize, model, prog_names, bound_names, n_prog, num_out = \
        load_model_and_data(args, cfg, device)
    hist = cfg.data.hist
    set_bn_mode(model, args.bn_mode)  # 1deg

    # ---- FiLM conversion (after the pretrained weights are in) -------------
    film_config = {
        "z_dim":     args.z_dim,
        "field_ch":  args.z_field_ch,
        "hidden":    args.film_hidden,
        "where":     args.film_where,
    }
    n_before = sum(p.numel() for p in model.parameters())
    model = make_stochastic(model, **film_config).to(device)
    model.enable_auto_z(args.z_dim, args.z_field_ch, tuple(args.z_field_hw),
                        seed=args.seed + 10_000 * (rank + 1))
    film_params = film_parameters(model)
    n_film = sum(p.numel() for p in film_params)
    logger.info("FiLM: where=%s z_dim=%d field=%dx%dx%d tensors=%d params=%.2fM "
                "(+%.1f%% on %.1fM)",
                args.film_where, args.z_dim, args.z_field_ch, *args.z_field_hw,
                len(film_params), n_film / 1e6, 100.0 * n_film / n_before,
                n_before / 1e6)

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

    def draw_z(seed0: int, step: int):
        return sample_z(local_members, args.z_dim, args.z_field_ch,
                        tuple(args.z_field_hw), device, seed0, step)

    # ---- acceptance test: zero-init FiLM == deterministic model ------------
    # Must run before --resume loads trained FiLM weights.
    if args.self_test:
        cur0 = tds._get_prognostic(tds._get_x_index(0)).to(device)
        st0 = perturbed_members(cur0, local_members, args.perturbation_level,
                               args.seed)
        x0 = build_input(st0, get_boundary(tds, 0))
        assert_zero_init_identity(model, x0, tds.ctx, draw_z(args.seed, 0))
        del cur0, st0, x0
        torch.cuda.empty_cache()
        logger.info("self-test OK: zero-init FiLM reproduces the deterministic "
                    "model bitwise (val baseline stays comparable)")

    dyn = (DynamicChannelScale(n_prog, num_out // n_prog - 1, args.dynamic_limit, device)  # 1deg
           if args.dynamic_loss else None)
    if dyn is not None:
        logger.info("dynamic per-variable loss weights enabled (limit=%.0f)",
                    args.dynamic_limit)

    # ---- optimizer: FiLM heads and backbone get separate learning rates ----
    film_ids = {id(p) for p in film_params}
    backbone_named = [(n, p) for n, p in model.named_parameters()
                      if id(p) not in film_ids]
    backbone_params = [p for _, p in backbone_named]
    named_trainable = list(model.named_parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "base_lr": args.lr,      "name": "backbone"},
            {"params": film_params,     "base_lr": args.film_lr, "name": "film"},
        ],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    trainable = backbone_params + film_params
    logger.info("params: backbone %.1fM @ lr=%.1e, FiLM %.2fM @ lr=%.1e",
                sum(p.numel() for p in backbone_params) / 1e6, args.lr,
                n_film / 1e6, args.film_lr)

    ema = ({n: p.detach().clone() for n, p in named_trainable}
           if args.ema_decay > 0 else None)
    if ema is not None:
        logger.info("EMA enabled: decay=%.4f (~%d-iter window)",
                    args.ema_decay, round(1.0 / (1.0 - args.ema_decay)))

    start_iter = 0
    ckpt = None
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
        if args.reset_base_lr:
            for g in optimizer.param_groups:
                was = g["base_lr"]
                g["base_lr"] = args.film_lr if g["name"] == "film" else args.lr
                logger.info("reset base_lr [%s]: %.2e -> %.2e",
                            g["name"], was, g["base_lr"])
        logger.info("resumed from %s at iter %d", args.resume, start_iter)

    baseline = ckpt.get("val_baseline") if ckpt is not None else None
    best_crps = ckpt.get("best_crps", float("inf")) if ckpt is not None \
        else float("inf")

    # This run writes ckpt_best/ckpt_latest under --out-dir. Refuse to point it
    # at the directory the checkpoint was loaded from: the deterministic
    # SamudraV2_CRPS run is finished and its ckpt_best_ema.pt is the only copy
    # of the iter-799 model.
    # 1deg: all stages share one folder now; refuse only if this stage's files could
    # overwrite the checkpoint it starts from (same prefix in the same folder).
    if (args.out_dir.resolve() == args.ckpt_path.resolve().parent
            and args.ckpt_path.name.startswith(args.ckpt_prefix + "_")):
        raise SystemExit(
            f"--out-dir {args.out_dir} is the directory holding --ckpt-path; "
            "this would overwrite the checkpoint being fine-tuned from. "
            "Use a separate output directory (default: SamudraV2_CRPS_FiLM)."
        )
    if rank == 0:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / f"{args.ckpt_prefix}_config.yaml").write_text(Path(args.config_path).read_text())  # 1deg

    def save_ckpt(it: int, name: str) -> None:
        if rank != 0:
            return
        payload = {
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "iter":          it,
            "ema":           ema,
            "dynamic_scale": dyn.state_dict() if dyn is not None else None,
            "val_baseline":  baseline,
            "best_crps":     best_crps,
            # Inference MUST call make_stochastic(model, **film_config) before
            # load_state_dict, or the FiLM weights are silently dropped.
            "film_config":   film_config,
            "z_field_hw":    tuple(args.z_field_hw),
            "crps_finetune_args": {k: str(v) for k, v in vars(args).items()},
        }
        fname = name.replace("ckpt_", f"{args.ckpt_prefix}_", 1)  # 1deg: e.g. 2-FiLM_noise_best.pt
        torch.save(payload, args.out_dir / fname)
        logger.info("saved %s", args.out_dir / fname)
        if ema is not None:
            ema_sd = {k: ema.get(k, v) for k, v in model.state_dict().items()}
            # 1deg: every gate-PASS keeps its own EMA file, never overwritten
            # (<prefix>_it<NNNN>_ema.pt); <prefix>_best_ema.pt is a symlink to the newest one.
            ema_name = f"{args.ckpt_prefix}_it{it:04d}_ema.pt" if name == "ckpt_best.pt" \
                else name.replace(".pt", "_ema.pt")
            torch.save({"model": ema_sd, "iter": it, "film_config": film_config,
                        "z_field_hw": tuple(args.z_field_hw)},
                       args.out_dir / ema_name)
            logger.info("saved %s", args.out_dir / ema_name)
            if name == "ckpt_best.pt":  # 1deg
                link = args.out_dir / f"{args.ckpt_prefix}_best_ema.pt"
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(ema_name)

    def run_validation(it: int, tag: str = ""):
        """Validate with EMA weights. `validate` calls model.forward_once(x, ctx)
        without a z, so the model's auto-z generator supplies one per member per
        step; pinning its seed keeps successive validations comparable."""
        model.set_z_seed(args.seed + 555_000 + rank)
        with ema_weights(named_trainable, ema):
            return validate(model, vds, args, rank, world, num_out, n_prog, hist,
                            weights, prog_names, device, it, tag=tag)

    def run_val_and_gate(it: int) -> None:
        """Validate, then overwrite ckpt_best only if EVERY threshold passes.

        Thresholds, all relative to the pre-finetune baseline (which is the
        starting checkpoint scored under this exact ensemble protocol):

          1. CRPS beats the baseline by >= --min-crps-gain (relative);
          2. CRPS beats the previous best;
          3. ensemble-mean RMSE within (1 + --rmse-tolerance) x baseline;
          4. SSR at every reported lead <= --ssr-max.

        (3) stops the run trading away deterministic skill for spread; (4)
        stops it overshooting into an over-dispersed ensemble, which stacking
        FiLM noise on top of already-calibrated IC noise can do. A single
        failed threshold leaves ckpt_best untouched, so the best model can
        never be poisoned by a later, worse iteration.
        """
        nonlocal best_crps
        v_crps, v_rmse, v_ssr = run_validation(it)
        if baseline is None or not np.isfinite(v_crps) or not np.isfinite(v_rmse):
            if rank == 0:
                logger.warning("gate SKIP: non-finite validation metrics "
                               "(CRPS=%.5f RMSE=%.5f)", v_crps, v_rmse)
            return
        crps_limit = baseline["crps"] * (1.0 - args.min_crps_gain)
        rmse_limit = baseline["rmse"] * (1.0 + args.rmse_tolerance)
        if args.rmse_abs_max > 0:
            # Whichever binds. The relative term alone compounds across
            # successive fine-tunes; the absolute one is anchored to the
            # original pretrained model and does not move.
            rmse_limit = min(rmse_limit, args.rmse_abs_max)
        checks = {
            "crps<baseline": v_crps < crps_limit,
            "crps<best":     v_crps < best_crps,
            "rmse":          v_rmse <= rmse_limit,
            "ssr":           args.ssr_max <= 0 or v_ssr <= args.ssr_max,
        }
        summary = ("CRPS=%.5f (need <%.5f, best %.5f)  RMSE=%.5f (limit %.5f)  "
                   "SSR_max=%.3f (limit %s)" % (
                       v_crps, crps_limit, best_crps, v_rmse, rmse_limit, v_ssr,
                       "off" if args.ssr_max <= 0 else "%.2f" % args.ssr_max))
        if all(checks.values()):
            best_crps = v_crps
            save_ckpt(it, "ckpt_best.pt")
            if rank == 0:
                logger.info("gate PASS: %s -> ckpt_best updated", summary)
        else:
            failed = ",".join(k for k, ok in checks.items() if not ok)
            if rank == 0:
                logger.info("gate SKIP [%s]: %s", failed, summary)
            # Closed-loop RMSE protection: the gate alone is passive, it only
            # refuses to save. Backing off the backbone lr is what actually
            # stops the drift, and backbone drift is the only thing that moves
            # the ensemble mean (FiLM perturbs members around it).
            if not checks["rmse"] and 0 < args.rmse_brake < 1.0:
                for g in optimizer.param_groups:
                    if g["name"] == "backbone":
                        g["base_lr"] *= args.rmse_brake
                        if rank == 0:
                            logger.warning("RMSE brake: backbone base_lr -> %.2e",
                                           g["base_lr"])

    # ---- training loop -----------------------------------------------------
    model.train()
    max_ic = len(tds) - args.rollout_steps
    if max_ic < 1:
        raise SystemExit("train window shorter than rollout-steps")

    if args.val_every > 0 and baseline is None:
        # FiLM is still zero -> this scores the unmodified deterministic model
        # under the exact ensemble protocol used later.
        b_crps, b_rmse, b_ssr = run_validation(start_iter, tag="baseline")
        if np.isfinite(b_crps):
            baseline = {"crps": b_crps, "rmse": b_rmse, "ssr_max": b_ssr}
            if rank == 0:
                logger.info("baseline set: CRPS=%.5f RMSE=%.5f SSR_max=%.3f",
                            b_crps, b_rmse, b_ssr)
                rel = b_rmse * (1.0 + args.rmse_tolerance)
                lim = min(rel, args.rmse_abs_max) if args.rmse_abs_max > 0 else rel
                logger.info("gate thresholds: CRPS < %.5f (-%.1f%%), RMSE <= "
                            "%.5f (%s; relative %.5f, absolute %s), SSR_max <= %s",
                            b_crps * (1.0 - args.min_crps_gain),
                            100 * args.min_crps_gain, lim,
                            "absolute binds" if lim < rel else "relative binds",
                            rel,
                            "off" if args.rmse_abs_max <= 0
                            else "%.5f" % args.rmse_abs_max,
                            "off" if args.ssr_max <= 0 else "%.2f" % args.ssr_max)
                if args.val_ics == 2:  # 1deg: reference is the 1deg stage-1 run
                    logger.info("NOTE: --val-ics 2 is the stage-1 protocol, so this "
                                "baseline should reproduce the stage-1 run's final "
                                "val CRPS/RMSE for the starting checkpoint; a "
                                "mismatch means the load or the protocol differs.")
                else:
                    logger.info("NOTE: --val-ics %d, so this baseline is NOT "
                                "comparable to stage-1's 2-IC numbers. It is "
                                "measured under the same protocol as every "
                                "other number in THIS run, which is what the "
                                "gate needs.", args.val_ics)

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

    frozen = False
    t_start = time.perf_counter()

    for it in range(start_iter, args.iters):
        ics = ics_of(it)  # 1deg: batch of B ICs
        seed0 = args.seed + it * 7919
        seed0s = [seed0 + b * 104729 for b in range(B)]  # b=0 -> the original seed

        # Phase 1: FiLM heads only. The backbone is converged; letting it react
        # to a still-zero noise channel just relearns the deterministic map.
        want_frozen = it < args.film_warmup_iters
        if want_frozen != frozen:
            for p in backbone_params:
                p.requires_grad = not want_frozen
            frozen = want_frozen
            logger.info("iter %d: backbone %s", it,
                        "FROZEN (FiLM-only phase)" if frozen else "unfrozen")

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

        warm = min(1.0, (it + 1) / max(1, args.warmup_iters))
        for g in optimizer.param_groups:
            g["lr"] = g["base_lr"] * warm

        # 1deg: (B*m_local, ...) IC-major; each IC gets its own Perlin seeds
        state = torch.cat([perturbed_members(d["ic"].to(device), local_members,
                                             args.perturbation_level, seed0s[b])
                           for b, d in enumerate(datas)])

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
            z_parts = [draw_z(seed0s[b], k) for b in range(B)]  # fresh z per step, per IC
            z = (torch.cat([zp[0] for zp in z_parts]),
                 torch.cat([zp[1] for zp in z_parts]) if args.z_field_ch > 0 else None)
            # Fresh latent draw per step: white-in-time model error.
            y = forward_members(model, x, tds.ctx, num_out, z)
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
                    alpha=args.crps_alpha,
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
            # film|w| stuck near 0 == the network is ignoring z: the failure
            # mode this whole design exists to avoid. Watch it with `spread`.
            # Peak allocation since the last log: the spatially-varying FiLM
            # modulation is the largest new tensor (2C x H x W at the block's
            # own resolution), so this is the number to watch for OOM headroom.
            peak_gb = (torch.cuda.max_memory_allocated() / 2**30
                       if torch.cuda.is_available() else 0.0)
            torch.cuda.reset_peak_memory_stats()
            logger.info(
                "iter %5d  ic=%-5d  CRPS=%.5f  skill=%.5f  spread=%.5f  "
                "film|w|=%.2e  lr=%.2e/%.2e  peak=%.1fGB  %.1fs/it%s",
                it, ics[0], log_crps, log_skill, log_spread, film_weight_norm(model),
                optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"],
                peak_gb, elapsed / max(1, it - start_iter + 1), dyn_str,
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
    import torch.distributed as dist
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
