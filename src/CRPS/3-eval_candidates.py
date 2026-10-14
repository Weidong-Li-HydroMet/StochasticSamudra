"""Score FiLM checkpoints against each other with the training-identical val protocol.

=== InMOS_1degree copy (2026-10-12) -- STAGE 3a: honest re-scoring / strength sweep ===
Minimal-diff copy; changed lines marked `1deg:`. Imports the 1deg stage-2 `validate`;
--ckpt-path defaults to the 1deg stage-1 output; --val-steps 16 (80 d). Validation runs
in eval mode, i.e. BatchNorm with the candidate's running stats (as the inference scripts).
=======================================================================================

Why this exists: the `ckpt_best` gate in `train_finetune_samudra_crps_film.py`
decides on a 6-IC estimate drawn from a single validation year, and the rollouts
of neighbouring ICs overlap (each spans 8 windows = 80 days out of ~36 windows),
so its effective sample size is ~4. Measured over run 6/7 the iteration-to-
iteration scatter of that estimate is sigma ~ 3.7e-4 while the gate's decision
margin is 4.4e-4 — i.e. the gate cannot tell a better model from a lucky one.

This script re-scores a *list* of checkpoints:

  * over the whole held-out period (default 2013-10-05 .. 2022-12-29) instead of
    one year, so ICs can be spaced ~80 days apart and are near-independent;
  * PAIRED — every candidate sees the identical IC list, the identical Perlin IC
    noise (seed0 = seed + 777000 + ic*1000) and the identical pinned FiLM z seed,
    so a difference between two candidates is far better resolved than each
    candidate's own absolute error bar;
  * with `validate()` imported from the training script, not reimplemented, so
    the numbers are on the same scale as the gate's.

`--candidate` takes `LABEL PATH [FILM_SCALE]`. PATH may be the literal `BASE`,
meaning "the --ckpt-path weights with a zero-init FiLM" — that is exactly how a
run measures its own pre-finetune baseline. FILM_SCALE multiplies the FiLM
modulation: the heads' output layers are zero-initialized and `mod` is linear in
them, so scaling those four tensors scales gamma/beta EXACTLY, and scale 0
recovers the candidate's own deterministic map (its drifted backbone, no learned
noise). Sweeping it answers the question the gate never asked: does the learned
model noise buy anything on top of the IC ensemble, or is the backbone drift
doing the work?

Usage (4 GPUs = 4 ensemble members, same layout as training):
    torchrun --standalone --nproc_per_node=4 3-eval_candidates.py \
        --candidate baseline BASE \
        --candidate iter1599 /.../SamudraV2_CRPS_FiLM/ckpt_best_ema.pt \
        --candidate iter2199 /.../SamudraV2_CRPS_FiLM/ckpt_latest_ema.pt \
        --val-ics 40
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]                    # 1deg: .../InMOS_1degree
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "CRPS"))

from ocean_emulators.config import EvalConfig, TimeConfig
from samudra_1deg import (  # 1deg: replaces samudra_om4_multi_ic_rollout
    BN_MODES, DEFAULT_CONFIG_PATH, DEFAULT_DATA_ROOT, DEFAULT_OUT_ROOT, InferenceDataset, load_model_and_data,
    set_bn_mode,
)
from crps_loss import make_loss_weights
# 1deg: stage 1 lives in 1-IC_noise.py (renamed 2026-10-12; not an importable module
# name), so load it by path. Same objects as the former `train_finetune_samudra_crps`.
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location("ic_noise_stage1", Path(__file__).with_name("1-IC_noise.py"))
_stage1 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_stage1)
build_input, get_boundary, setup_distributed = (  # 1deg
    _stage1.build_input, _stage1.get_boundary, _stage1.setup_distributed)
from stochastic_samudra import make_stochastic, scale_film

# 1deg: stage 2 lives in 2-FiLM_noise.py (renamed; not an importable module name).
_spec2 = _ilu.spec_from_file_location("film_noise_stage2", Path(__file__).with_name("2-FiLM_noise.py"))
_stage2 = _ilu.module_from_spec(_spec2)
_spec2.loader.exec_module(_stage2)
validate = _stage2.validate

logger = logging.getLogger("eval_film_candidates")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidate", action="append", nargs="+", metavar="LABEL PATH [SCALE]",
                   required=True, help="repeatable; PATH may be the literal BASE")
    p.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--ckpt-path", type=Path,
                   default=DEFAULT_OUT_ROOT / "1-IC_noise_best_ema.pt",  # 1deg: stage-1 newest PASS
                   help="weights the model is built from; also the BASE candidate")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    # The train slice is only used for the loss/val weights (its label mask), so
    # that they are bitwise the weights the gate used.
    p.add_argument("--train-start", default="1975-01-03")
    p.add_argument("--train-end",   default="2013-10-05")
    p.add_argument("--val-start",   default="2013-10-05")
    p.add_argument("--val-end",     default="2022-12-29",
                   help="default is ALL held-out data, not the 1-year training val slice")
    p.add_argument("--val-ics",   type=int, default=40)
    p.add_argument("--val-steps", type=int, default=16)  # 1deg: 16 x 5 d = 80 d
    p.add_argument("--members-per-rank", type=int, default=1)
    p.add_argument("--rollout-steps", type=int, default=8)  # cfg only; unused here  # 1deg
    p.add_argument("--perturbation-level", type=float, default=0.025)
    p.add_argument("--per-step-noise", type=float, default=0.0)
    p.add_argument("--z-dim", type=int, default=64)
    p.add_argument("--z-field-ch", type=int, default=4)
    p.add_argument("--z-field-hw", type=int, nargs=2, default=(45, 90))
    p.add_argument("--film-hidden", type=int, default=128)
    p.add_argument("--film-where", default="bottleneck")
    p.add_argument("--lat-weight", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--bn-mode", choices=BN_MODES, default="sample")  # 1deg: same as training/inference
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


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

    cands = []
    for spec in args.candidate:
        if len(spec) not in (2, 3):
            raise SystemExit(f"--candidate takes LABEL PATH [SCALE], got {spec}")
        label, path = spec[0], spec[1]
        scale = float(spec[2]) if len(spec) == 3 else 1.0
        cands.append((label, path, scale))

    M = world * args.members_per_rank
    if M < 2:
        raise SystemExit("Fair CRPS needs M >= 2.")
    logger.info("world=%d members_per_rank=%d -> M=%d", world, args.members_per_rank, M)

    with open(args.config_path) as f:
        train_yaml = yaml.safe_load(f)
    cfg = EvalConfig(
        ckpt_path               = str(args.ckpt_path),
        num_model_steps_forward = args.rollout_steps,
        inference_time          = {"start": args.train_start, "end": args.train_end},
        experiment              = {
            **train_yaml["experiment"],
            "data_root": str(args.data_root),
            "name":      "crps_film_candidate_eval",
            "wandb":     {"mode": "disabled", "project": "default"},
        },
        data  = train_yaml["data"],
        model = train_yaml["model"],
    )
    src, normalize, model, prog_names, bound_names, n_prog, num_out = \
        load_model_and_data(args, cfg, device)
    hist = cfg.data.hist
    set_bn_mode(model, args.bn_mode)  # 1deg

    film_config = {"z_dim": args.z_dim, "field_ch": args.z_field_ch,
                   "hidden": args.film_hidden, "where": args.film_where}
    model = make_stochastic(model, **film_config).to(device)
    model.enable_auto_z(args.z_dim, args.z_field_ch, tuple(args.z_field_hw),
                        seed=args.seed + 10_000 * (rank + 1))
    # Deterministic weights + zero-init FiLM: the run's own baseline state.
    base_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

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
    lat = (torch.from_numpy(src.data.lat.values.astype(np.float32)).to(device)
           if args.lat_weight else None)
    weights = make_loss_weights(tds.ctx.label_mask, lat)

    n_windows = len(vds)
    ic_list = np.unique(np.linspace(0, n_windows - args.val_steps - 1,
                                    args.val_ics).astype(int))
    gap = float(np.mean(np.diff(ic_list))) if len(ic_list) > 1 else float("inf")
    logger.info("val %s..%s: %d windows, %d ICs, mean IC gap %.1f windows "
                "(rollout spans %d -> %s)", args.val_start, args.val_end,
                n_windows, len(ic_list), gap, args.val_steps,
                "near-independent" if gap >= args.val_steps else "OVERLAPPING")
    logger.info("weighting: %s over wet points; M=%d, IC sigma=%.3f, leads up to %d steps",
                "cos(lat)" if args.lat_weight else "grid-uniform",
                M, args.perturbation_level, args.val_steps)

    results = []
    for label, path, scale in cands:
        if path == "BASE":
            sd = base_sd
        else:
            payload = torch.load(path, map_location=device, weights_only=False)
            fc = payload.get("film_config")
            if fc is not None and fc != film_config:
                raise SystemExit(f"{label}: film_config {fc} != this run's {film_config}; "
                                 "FiLM weights would not load onto this architecture")
            sd = payload["model"]
            if payload.get("iter") is not None:
                label = f"{label}(it{payload['iter']})"
        model.load_state_dict(sd)  # strict: a silent FiLM drop must be an error
        if scale != 1.0:
            n = scale_film(model, scale)
            logger.info("%s: FiLM modulation scaled by %.3g (%d tensors)", label, scale, n)
        tag = f"{label} scale={scale:g}" if scale != 1.0 else label
        model.set_z_seed(args.seed + 555_000 + rank)  # same pinning as training val
        crps, rmse, ssr = validate(model, vds, args, rank, world, num_out, n_prog,
                                   hist, weights, prog_names, device, 0, tag=tag)
        results.append({"label": label, "path": str(path), "film_scale": scale,
                        "crps": crps, "rmse": rmse, "ssr_max": ssr})
        logger.info("[%s] CRPS=%.5f  RMSE=%.5f  SSR_max=%.3f", tag, crps, rmse, ssr)

    if rank == 0:
        ref = results[0]
        logger.info("=" * 78)
        logger.info("%-26s %9s %8s %9s %8s %7s", "candidate", "CRPS", "dCRPS%",
                    "RMSE", "dRMSE%", "SSRmax")
        for r in results:
            logger.info("%-26s %9.5f %+8.2f %9.5f %+8.2f %7.3f",
                        f"{r['label']}"
                        + (f" x{r['film_scale']:g}" if r["film_scale"] != 1.0 else ""),
                        r["crps"], 100.0 * (r["crps"] / ref["crps"] - 1.0),
                        r["rmse"], 100.0 * (r["rmse"] / ref["rmse"] - 1.0),
                        r["ssr_max"])
        logger.info("deltas are vs the first candidate (%s); paired protocol — "
                    "identical ICs, IC noise and z seeds throughout", ref["label"])
        if args.out_json is not None:
            args.out_json.write_text(json.dumps(
                {"val_start": args.val_start, "val_end": args.val_end,
                 "n_ics": len(ic_list), "ic_gap_windows": gap, "M": M,
                 "val_steps": args.val_steps,
                 "perturbation_level": args.perturbation_level,
                 "results": results}, indent=2))
            logger.info("wrote %s", args.out_json)


if __name__ == "__main__":
    main()
