"""1-degree SamudraV2: lagged-IC ensemble (port of Sampling 0.5-samudra_om4_lag_ensemble_rollout.py).

Member m starts from the TRUE states (x(s), x(s+1)), the later one (m+1)*lag steps
before --test-start, and rolls to --inference-end.
Spread comes only from the IC lag (deterministic model, no Perlin noise). Only frames
with valid date >= --test-start are written, so all members share one time axis:

    member 0: IC 2014-09-30, 2014-10-05 -> [2014-10-10 ... 2022-12-24]  (lag 1 step = 5 d)
    member 1: IC 2014-09-25, 2014-09-30 -> warm-up 1 step -> [2014-10-10 ... ]
    ...
1 step = 5 days at 1 degree (the 0.25-degree script lagged by 10-day steps).
Frame k = valid date test_start + 5k days; every frame is a prediction.

    CUDA_VISIBLE_DEVICES=0 /data/weidong/ocean_env/bin/python 1-SamudraV2_lag.py \
        --n-members 2 --inference-end 2014-11-14 --output-dir /tmp/smoke/lag
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from common import NP, PRED_ROOT, SAMUDRA_CKPT, Data, build_samudra, create_zarr, sst_error, write_frames

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--ckpt-path", type=Path, default=SAMUDRA_CKPT)
p.add_argument("--test-start", default="2014-10-10", help="first saved (valid) date")
p.add_argument("--inference-end", default="2022-12-24")
p.add_argument("--n-members", type=int, default=32)
p.add_argument("--lag-steps", type=int, default=1, help="IC spacing between members, in 5-day steps")
p.add_argument("--output-dir", type=Path, default=PRED_ROOT / "SamudraV2_lag")
p.add_argument("--chunk-frames", type=int, default=5)
p.add_argument("--log-every", type=int, default=50)
p.add_argument("--overwrite", action="store_true")
p.add_argument("--rank", type=int, default=0, help="worker index; members are split [rank::world_size]")
p.add_argument("--world-size", type=int, default=1)
args = p.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t_start = time.perf_counter()

data = Data()
anchor, end = data.index(args.test_start), data.index(args.inference_end)
frame_dates = data.dates[anchor:end + 1]
if anchor - 1 - args.n_members * args.lag_steps < 0:
    raise SystemExit("too many members x lag steps for the start of the data")
model, ctx, _ = build_samudra(args.ckpt_path, device)
print(f"{args.n_members} members, lag {args.lag_steps} step(s); deepest IC "
      f"{data.dates[anchor - 1 - args.n_members * args.lag_steps]}; save {frame_dates[0]}..{frame_dates[-1]} "
      f"({len(frame_dates)} frames)", flush=True)
args.output_dir.mkdir(parents=True, exist_ok=True)

for m in list(range(args.n_members))[args.rank::args.world_size]:
    lag = (m + 1) * args.lag_steps                     # IC x(s+1) is `lag` steps before test_start
    s = anchor - 1 - lag                               # IC = true (x(s), x(s+1))
    out = args.output_dir / (f"SamudraV2_1deg_lag_test{frame_dates[0]}_to{frame_dates[-1]}"
                             f"_lag{lag}_m{m:02d}.zarr")
    if out.exists() and not args.overwrite:
        print(f"skip (exists) {out.name}", flush=True)
        continue
    t0 = time.perf_counter()
    create_zarr(out, data, frame_dates, chunk_frames=args.chunk_frames, attrs={
        "init_date": str(frame_dates[0]), "member_init_date": str(data.dates[s + 1]),
        "final_date": str(frame_dates[-1]), "step_days": 5, "lag_steps": lag, "lag_days": 5 * lag,
        "frames": "frame k = prediction valid at test_start + 5k d (warm-up steps not saved)",
        "member": m, "n_members": args.n_members, "units": "physical, NaN on land",
        "checkpoint": str(args.ckpt_path), "bn_stats": "per-sample",
        "model": "SamudraV2 1deg (one-step), lagged-IC ensemble"})

    state = torch.from_numpy(np.concatenate([data.state(s), data.state(s + 1)]))[None].to(device)
    buf_pred, buf_tgt, written = [], [], 0
    n_steps = end - (s + 1)
    with torch.no_grad():
        for k in range(n_steps):
            j = s + 2 + k                                                # valid index of this step
            f = np.concatenate([data.forc(j - 2), data.forc(j - 1)])
            x = torch.cat([state, torch.from_numpy(f)[None].to(device)], 1)
            y = model.forward_once(x, ctx)                               # (1, 77, Y, X)
            state = torch.cat([state[:, NP:], y], 1)
            if j < anchor:                                               # warm-up, not saved
                continue
            y_np, truth = y[0].float().cpu().numpy(), data.state(j)
            buf_pred.append(y_np)
            buf_tgt.append(truth)
            if len(buf_pred) >= args.chunk_frames or j == end:
                written = write_frames(out, data, buf_pred, buf_tgt, written)
                buf_pred, buf_tgt = [], []
            if (j - anchor + 1) % args.log_every == 0 or j == end:
                bias, rmse = sst_error(data, y_np, truth)
                print(f"m={m:02d} valid={data.dates[j]}  SST bias={bias:+.2f}C rmse={rmse:.2f}C  "
                      f"{(time.perf_counter() - t0) / (k + 1):.2f}s/step", flush=True)
    print(f"member {m:02d} (lag {lag}) done in {time.perf_counter() - t0:.1f}s -> {out.name}", flush=True)

print(f"done in {time.perf_counter() - t_start:.1f}s -> {args.output_dir}")
