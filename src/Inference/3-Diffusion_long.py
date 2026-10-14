"""1-degree SamudraV2 + conditional-diffusion SDEdit: single-IC long ensemble
(port of Sampling 1-Long-Diff.py / 1-samudra_om4_long_diffusion_ensemble_rollout.py).

Every step: SamudraV2 1deg predicts x(t+1); for members >= 1 that prediction is then
perturbed in pixel space by SDEdit with the 1deg diffusion model p(state | forcing, date)
of src/Diffusion: noise it to level t0 of a --sde-steps DDIM schedule, then DDIM-denoise
back to 0, conditioned on the TRUE forcing and day-of-year at the valid time t+1
(hfds clipped to +-8, as in training). The result is fed back autoregressively.

t0 counts DDIM steps from pure noise (t0 = 95 of 100 = only the last 5 steps, light noise),
piecewise linear over the rollout:
    step <  --sde-phase1-steps : t0 = sde_t0_0   -> sde_t0_mid
    step >= --sde-phase1-steps : t0 = sde_t0_mid -> sde_t0_1
Defaults 95 -> 50 -> 30 as in the 0.25-degree run; phase 1 = 40 steps (200 days, the
0.25-degree 20 steps x 10 days).

Member 0 = unperturbed control (no Perlin, no SDEdit); member m >= 1 = Perlin IC (index m,
as 0-SamudraV2_long.py) + per-step SDEdit, torch seed --seed + 1000 m + 7.
Output layout = 0-SamudraV2_long.py (frame 0-1 = IC states, k+2 = prediction of step k).

    CUDA_VISIBLE_DEVICES=0 /data/weidong/ocean_env/bin/python 3-Diffusion_long.py \
        --n-members 2 --rollout-len 4 --output-dir /tmp/smoke/diff
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from common import NP, PRED_ROOT, SAMUDRA_CKPT, SRC, Data, build_samudra, create_zarr, perlin_ic_noise, \
    sst_error, write_frames

sys.path.insert(0, str(SRC / "Diffusion"))
from model import UNet_1degV1, Diffusion_for_UNet  # noqa: E402  (src/Diffusion/model.py)

# Diffusion constants: must match src/Diffusion/train.py
DIFF_CHANNEL, DIFF_C_MULTS, DIFF_SCALE = 128, [1, 2, 2, 4, 8], [3, 2, 2]

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--ckpt-path", type=Path, default=SAMUDRA_CKPT, help="SamudraV2 1deg weights")
p.add_argument("--diff-ckpt", type=Path, default=Path("/data/weidong/InMOS_1deg/ckpt/Diffusion/condiff_1deg_date_best"),
               help="accelerate save_state dir (model.safetensors) of src/Diffusion/train.py")
p.add_argument("--inference-start", default="2014-10-10")
p.add_argument("--inference-end", default="2022-12-24")
p.add_argument("--rollout-len", type=int, default=None, help="steps (5 d each); default = whole window")
p.add_argument("--n-members", type=int, default=8, help="member 0 = control")
p.add_argument("--perturbation-level", type=float, default=0.025)
p.add_argument("--base-seed", type=int, default=2025)
p.add_argument("--sde-steps", type=int, default=100, help="DDIM steps of the SDEdit schedule")
p.add_argument("--sde-t0-0", type=int, default=95)
p.add_argument("--sde-t0-mid", type=int, default=50)
p.add_argument("--sde-t0-1", type=int, default=30)
p.add_argument("--sde-phase1-steps", type=int, default=40)
p.add_argument("--clamp-x0", type=float, default=12.0, help="x0 clamp in the first 70%% of the DDIM steps")
p.add_argument("--seed", type=int, default=15)
p.add_argument("--output-dir", type=Path, default=PRED_ROOT / "Diffusion_long")
p.add_argument("--chunk-frames", type=int, default=5)
p.add_argument("--log-every", type=int, default=50)
p.add_argument("--overwrite", action="store_true")
p.add_argument("--rank", type=int, default=0, help="worker index; members are split [rank::world_size]")
p.add_argument("--world-size", type=int, default=1)
args = p.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t_start = time.perf_counter()


def sde_t0(step, L):
    """Two-segment linear t0 schedule (piecewise_sde_t0 of the 0.25-degree script)."""
    a, mid, b, n1 = args.sde_t0_0, args.sde_t0_mid, args.sde_t0_1, args.sde_phase1_steps
    if step < n1:
        t0 = a - step / max(1, n1 - 1) * (a - mid)
    else:
        t0 = mid - (step - n1) / max(1, L - n1 - 1) * (mid - b)
    return int(np.clip(round(t0), 0, args.sde_steps - 1))


# ---------------------------------------------------------------- data / models --
data = Data()
i0 = data.index(args.inference_start)
i_end = data.index(args.inference_end)
L = args.rollout_len if args.rollout_len is not None else i_end - i0 - 1
if i0 + L + 1 > i_end:
    sys.exit(f"rollout-len {L} exceeds the window ({i_end - i0 - 1} steps max)")
frame_dates = data.dates[i0:i0 + L + 2]

model, ctx, _ = build_samudra(args.ckpt_path, device)
diff_model = UNet_1degV1(NP + 3, NP, DIFF_CHANNEL, DIFF_C_MULTS, scale=DIFF_SCALE)
diff_model.load_state_dict(load_file(str(args.diff_ckpt / "model.safetensors")), strict=True)
diff_model = diff_model.to(device).eval()
diffusion = Diffusion_for_UNet(size=[NP, *data.wet.shape[1:]], lambda_range=20, steps=1000, device=device)
u_seq = torch.cat([torch.arange(diffusion.bound, 0, -1.0 / args.sde_steps, device=device),
                   torch.tensor([0.0], device=device)])
prog_mask = torch.from_numpy(data.wet[None].astype(np.float32)).to(device)
print(f"init {frame_dates[0]}, {L} steps -> {frame_dates[-1]} | diffusion {args.diff_ckpt} | t0 "
      f"{sde_t0(0, L)} -> {sde_t0(args.sde_phase1_steps, L)} (step {args.sde_phase1_steps}) -> "
      f"{sde_t0(L - 1, L)} of {args.sde_steps}", flush=True)

s0, s1 = data.state(i0), data.state(i0 + 1)
args.output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- members --------
for m in list(range(args.n_members))[args.rank::args.world_size]:
    out = args.output_dir / (f"SamudraV2Diff_1deg_long_init{frame_dates[0]}_to{frame_dates[-1]}_steps{L}"
                             f"_sigma{args.perturbation_level}_t0{args.sde_t0_0}-{args.sde_t0_mid}-{args.sde_t0_1}"
                             f"_m{m:02d}.zarr")
    if out.exists() and not args.overwrite:
        print(f"skip (exists) {out.name}", flush=True)
        continue
    t0 = time.perf_counter()
    create_zarr(out, data, frame_dates, chunk_frames=args.chunk_frames, attrs={
        "init_date": str(frame_dates[0]), "final_date": str(frame_dates[-1]), "rollout_len": L,
        "step_days": 5, "n_ic_frames": 2,
        "frames": "0-1 = IC states (prediction: perturbed, target: truth); k+2 = prediction of step k",
        "member": m, "n_members": args.n_members, "perturbation_level": args.perturbation_level,
        "perturbation_kind": "perlin_3octaves + per-step SDEdit" if m > 0 else "none (control)",
        "base_seed": args.base_seed, "seed": args.seed, "sde_steps": args.sde_steps,
        "sde_t0": f"{args.sde_t0_0}->{args.sde_t0_mid}@{args.sde_phase1_steps}->{args.sde_t0_1}",
        "units": "physical, NaN on land", "checkpoint": str(args.ckpt_path), "diff_checkpoint": str(args.diff_ckpt),
        "bn_stats": "per-sample", "model": "SamudraV2 1deg + Perlin IC + per-step diffusion SDEdit"})

    state = torch.from_numpy(np.concatenate([s0, s1]))[None].to(device)        # (1, 154, Y, X)
    if m > 0:
        noise = perlin_ic_noise(s0.shape[1:], m, args.base_seed)
        state = state + args.perturbation_level * torch.from_numpy(noise).to(device)
    torch.manual_seed(args.seed + 1000 * m + 7)
    ic_np = state[0].cpu().numpy()
    buf_pred, buf_tgt, written = [ic_np[:NP], ic_np[NP:]], [s0, s1], 0
    with torch.no_grad():
        for k in range(L):
            j = i0 + k + 2                                                       # valid index
            f = np.concatenate([data.forc(j - 2), data.forc(j - 1)])
            x = torch.cat([state, torch.from_numpy(f)[None].to(device)], 1)
            y = model.forward_once(x, ctx)                                       # (1, 77, Y, X)

            if m > 0:                                                            # SDEdit
                cond = torch.from_numpy(data.forc(j))[None].to(device)
                cond[:, 2] = cond[:, 2].clamp(-8.0, 8.0)                         # hfds, as in training
                date_frac = torch.tensor([data.doy[j] / 365.0], device=device)
                t0k = sde_t0(k, L)
                xd, _, _ = diffusion.noise_images(y.float(), torch.ones(1, device=device) * u_seq[t0k])
                for i in range(t0k, len(u_seq) - 1):
                    u = torch.ones(1, device=device) * u_seq[i]
                    u_next = torch.ones(1, device=device) * u_seq[i + 1]
                    xd = xd * prog_mask
                    with torch.autocast("cuda", dtype=torch.bfloat16):          # as trained
                        pred_v = diff_model(torch.cat([xd, cond], 1), u, date_frac=date_frac)
                    pred_v = pred_v.float() * prog_mask
                    x0 = diffusion.predict_start_from_v(xd, u, pred_v)
                    if i < args.sde_steps * 0.7:
                        x0 = x0.clamp(-args.clamp_x0, args.clamp_x0)
                    xd = diffusion.i_posterior(x0, xd, u, u_next)
                y = (xd * prog_mask).to(y.dtype)

            state = torch.cat([state[:, NP:], y], 1)
            y_np, truth = y[0].float().cpu().numpy(), data.state(j)
            buf_pred.append(y_np)
            buf_tgt.append(truth)
            if len(buf_pred) >= args.chunk_frames or k == L - 1:
                written = write_frames(out, data, buf_pred, buf_tgt, written)
                buf_pred, buf_tgt = [], []
            if (k + 1) % args.log_every == 0 or k == L - 1:
                bias, rmse = sst_error(data, y_np, truth)
                print(f"m={m:02d} step {k + 1}/{L} valid={frame_dates[k + 2]} t0={sde_t0(k, L) if m else '-'}  "
                      f"SST bias={bias:+.2f}C rmse={rmse:.2f}C  {(time.perf_counter() - t0) / (k + 1):.2f}s/step",
                      flush=True)
    print(f"member {m:02d} done in {time.perf_counter() - t0:.1f}s -> {out.name}", flush=True)

print(f"done in {time.perf_counter() - t_start:.1f}s -> {args.output_dir}")
