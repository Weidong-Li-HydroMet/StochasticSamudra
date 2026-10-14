"""1-degree SI forecaster: single-IC long ensemble (port of Sampling 5-Long-SamudraSI_NVIDIA.py).

Model: src/SI (stochastic interpolant a_t -> a_{t+1}, one 5-day step, 77 ch, full field,
cond [a_t, (a_t - a_{t-1}) / s_inc], TRUE forcing [f(t-1), f(t)], calendar [0, doy(t)]).
Inference operator = the one the rollout stages train with: --si-steps 2 DDIM steps, the
5-day increment clamped to +-inc_clamp * s_inc, land re-zeroed, strict fp32.

Spread = two sources, as in the 0.25-degree SI ensemble:
  a. Perlin IC noise (sigma 0.025 on the normalized states a_{t-1}, a_t, land re-zeroed),
     member m gets Perlin index m+1, the same draw as CRPS member m (2-CRPS_long.py);
  b. the SI sampler noise, torch seed --si-seed + 100003 m + 7 k for call k.
--control-member-0: member 0 = unperturbed IC (still a stochastic SI sample).
--perturbation-level 0 gives the SI-noise-only ensemble.

Weights: an EMA sidecar (*_ema.pt) or a raw accelerate dir (model.safetensors); the UViT
config is read from the matching *_meta.pt. Default = the stage-1 best EMA (the only SI
checkpoint so far); after stage 2/3 pass e.g. .../SI1deg_replay_best (raw weights).
Output layout = 0-SamudraV2_long.py (frame 0-1 = IC states, k+2 = prediction of call k).

Runs in /data/weidong/wdenv, the SI training env (torch_harmonics 0.8.0; ocean_env's 0.9.2 calibrates
the spherical noise 2% larger, so samples differ from what the model was trained with).

    CUDA_VISIBLE_DEVICES=0 /data/weidong/wdenv/bin/python 4-SI_long.py \
        --n-members 2 --rollout-len 4 --output-dir /tmp/smoke/si
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import NP, PRED_ROOT, SI_DATA, SRC, Data, create_zarr, perlin_ic_noise, sst_error, write_frames

sys.path.insert(0, str(SRC / "SI"))
from ocean_mask import OceanMask  # noqa: E402
from si_common import bind_call, build_model, build_si, load_weights, rollout_step, set_precision  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--ckpt-path", type=Path, default=Path("/data/weidong/InMOS_1deg/ckpt/SI/SI1deg_step1_best_ema.pt"))
p.add_argument("--inference-start", default="2014-10-10")
p.add_argument("--inference-end", default="2022-12-24")
p.add_argument("--rollout-len", type=int, default=None, help="calls (5 d each); default = whole window")
p.add_argument("--n-members", type=int, default=32)
p.add_argument("--control-member-0", action="store_true", help="member 0: unperturbed IC")
p.add_argument("--perturbation-level", type=float, default=0.025)
p.add_argument("--base-seed", type=int, default=2025, help="Perlin seeds = base_seed + 3*index + {0,1,2}")
p.add_argument("--si-steps", type=int, default=2, help="DDIM steps per call")
p.add_argument("--inc-clamp", type=float, default=8.0, help="clamp of the 5-day increment, in s_inc (0 = off)")
p.add_argument("--si-seed", type=int, default=20260826)
p.add_argument("--output-dir", type=Path, default=PRED_ROOT / "SI_long")
p.add_argument("--chunk-frames", type=int, default=5)
p.add_argument("--log-every", type=int, default=50)
p.add_argument("--overwrite", action="store_true")
p.add_argument("--rank", type=int, default=0, help="worker index; members are split [rank::world_size]")
p.add_argument("--world-size", type=int, default=1)
args = p.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t_start = time.perf_counter()

# ---------------------------------------------------------------- data -----------
data = Data()
i0 = data.index(args.inference_start)
i_end = data.index(args.inference_end)
L = args.rollout_len if args.rollout_len is not None else i_end - i0 - 1
if i0 + L + 1 > i_end:
    sys.exit(f"rollout-len {L} exceeds the window ({i_end - i0 - 1} steps max)")
frame_dates = data.dates[i0:i0 + L + 2]

# ---------------------------------------------------------------- model ----------
set_precision("fp32")
ck = str(args.ckpt_path)
meta_path = (ck[:-len("_ema.pt")] if ck.endswith("_ema.pt") else ck.rstrip("/")) + "_meta.pt"
cfg = torch.load(meta_path, map_location="cpu", weights_only=False)["cfg"]
cfg["grad_checkpoint"] = False
ocean = OceanMask(patch_size=2 ** len(cfg["uvit_widths"])).to(device)
cell_mask = ocean.cell_mask                                                  # (1, 77, Y, X)
s_inc_f = torch.from_numpy(np.load(SI_DATA / "stats.npz")["s_inc_floored"].astype(np.float32)).to(device)
s_inc_b = s_inc_f.view(1, NP, 1, 1)
module = build_model(ocean, cfg)
load_weights(module, ck)
module = module.to(device).eval()
si, _ = build_si(cfg, cell_mask, s_inc_f, device)
print(f"{args.ckpt_path} ({sum(q.numel() for q in module.parameters()) / 1e6:.1f}M) | ddim {args.si_steps}, "
      f"clamp {args.inc_clamp:g} s_inc | init {frame_dates[0]}, {L} calls -> {frame_dates[-1]}", flush=True)

s0, s1 = data.state(i0), data.state(i0 + 1)
args.output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- members --------
for m in list(range(args.n_members))[args.rank::args.world_size]:
    index = m if args.control_member_0 else m + 1                # Perlin index; 0 = no noise
    out = args.output_dir / (f"SI_1deg_long_init{frame_dates[0]}_to{frame_dates[-1]}_steps{L}"
                             f"_sigma{args.perturbation_level}_ddim{args.si_steps}_K{args.inc_clamp:g}_m{m:02d}.zarr")
    if out.exists() and not args.overwrite:
        print(f"skip (exists) {out.name}", flush=True)
        continue
    t0 = time.perf_counter()
    create_zarr(out, data, frame_dates, chunk_frames=args.chunk_frames, attrs={
        "init_date": str(frame_dates[0]), "final_date": str(frame_dates[-1]), "rollout_len": L,
        "step_days": 5, "n_ic_frames": 2,
        "frames": "0-1 = IC states (prediction: perturbed, target: truth); k+2 = prediction of call k",
        "member": m, "n_members": args.n_members, "perturbation_level": args.perturbation_level,
        "perlin_index": index, "base_seed": args.base_seed, "si_seed": args.si_seed,
        "si_steps": args.si_steps, "inc_clamp": args.inc_clamp,
        "units": "physical, NaN on land", "checkpoint": ck, "precision": "fp32",
        "model": "1deg SI (a_t -> a_{t+1}, cond [a_t, da_t]) + Perlin IC + SI sampler noise"})

    x_prev = torch.from_numpy(s0)[None].to(device)
    x = torch.from_numpy(s1)[None].to(device)
    if index > 0 and args.perturbation_level > 0:
        noise = torch.from_numpy(perlin_ic_noise(s0.shape[1:], index, args.base_seed)).to(device)
        x_prev = (x_prev + args.perturbation_level * noise) * cell_mask
        x = (x + args.perturbation_level * noise) * cell_mask
    buf_pred, buf_tgt, written = [x_prev[0].cpu().numpy(), x[0].cpu().numpy()], [s0, s1], 0
    with torch.no_grad():
        for k in range(L):
            t = i0 + 1 + k                                                       # a_t = x
            c_grid = torch.from_numpy(np.concatenate([data.forc(t - 1), data.forc(t)]))[None].to(device)
            scalar = torch.tensor([[0.0, float(data.doy[t])]], device=device)
            torch.manual_seed(args.si_seed + 100_003 * m + 7 * k)               # per (member, call)
            call = bind_call(module, scalar, x_prev, cell_mask, 1.0 / s_inc_b)
            y = rollout_step(call, si, x, c_grid, args.si_steps, args.inc_clamp, s_inc_b, cell_mask)
            x_prev, x = x, y
            y_np, truth = y[0].cpu().numpy(), data.state(t + 1)
            buf_pred.append(y_np)
            buf_tgt.append(truth)
            if len(buf_pred) >= args.chunk_frames or k == L - 1:
                written = write_frames(out, data, buf_pred, buf_tgt, written)
                buf_pred, buf_tgt = [], []
            if (k + 1) % args.log_every == 0 or k == L - 1:
                bias, rmse = sst_error(data, y_np, truth)
                print(f"m={m:02d} call {k + 1}/{L} valid={frame_dates[k + 2]}  SST bias={bias:+.2f}C "
                      f"rmse={rmse:.2f}C  max|a|={float(y.abs().max()):.1f}  "
                      f"{(time.perf_counter() - t0) / (k + 1):.2f}s/call", flush=True)
    print(f"member {m:02d} done in {time.perf_counter() - t0:.1f}s -> {out.name}", flush=True)

print(f"done in {time.perf_counter() - t_start:.1f}s -> {args.output_dir}")
