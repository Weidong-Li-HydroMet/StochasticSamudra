"""1-degree CRPS SamudraV2: single-IC long ensemble (port of Sampling 3-/4-samudra_om4_crps_*long_ensemble_rollout.py).

One script for both CRPS stages (src/CRPS); the checkpoint decides which:
  * stage 1 (1-IC_noise_*_ema.pt, no film_config): deterministic CRPS model,
    spread only from the Perlin IC noise (= Sampling script 3-);
  * stage 2 (2-FiLM_noise_*_ema.pt, has film_config): FiLM heads are added before the
    strict load, and a latent z is REDRAWN EVERY STEP, as in training (= Sampling 4-).
    z for (member m, step k) = stochastic_samudra.sample_z([m], ..., --z-seed, k), so a
    member is reproducible whichever worker runs it.

Members: EVERY member is perturbed (Perlin index m+1, sigma 0.025 = the training value) and
FiLM is on, so the ensemble is exchangeable (training had no control member).
--control-member-0: member 0 = unperturbed IC and FiLM off (the deterministic backbone).
--film-scale s multiplies the FiLM modulation at run time (0 = FiLM off for everyone).

Output layout = 0-SamudraV2_long.py: one zarr per member, all 77 channels, physical units,
frame 0-1 = the (perturbed) IC states, frame k+2 = prediction of step k; `target` holds truth.

    CUDA_VISIBLE_DEVICES=0 /data/weidong/ocean_env/bin/python 2-CRPS_long.py \
        --n-members 2 --rollout-len 4 --output-dir /tmp/smoke/crps
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import NP, PRED_ROOT, SRC, Data, build_samudra, create_zarr, perlin_ic_noise, sst_error, write_frames

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--ckpt-path", type=Path, default=Path("/data/weidong/InMOS_1deg/ckpt/CRPS/2-FiLM_noise_best_ema.pt"))
p.add_argument("--inference-start", default="2014-10-10")
p.add_argument("--inference-end", default="2022-12-24")
p.add_argument("--rollout-len", type=int, default=None, help="steps (5 d each); default = whole window")
p.add_argument("--n-members", type=int, default=32)
p.add_argument("--control-member-0", action="store_true", help="member 0: unperturbed IC, FiLM off")
p.add_argument("--perturbation-level", type=float, default=0.025)
p.add_argument("--base-seed", type=int, default=2025, help="Perlin seeds = base_seed + 3*index + {0,1,2}")
p.add_argument("--z-seed", type=int, default=2025, help="FiLM latent seed offset")
p.add_argument("--film-scale", type=float, default=1.0, help="runtime FiLM strength (0 = off)")
p.add_argument("--output-dir", type=Path, default=PRED_ROOT / "CRPS_long")
p.add_argument("--chunk-frames", type=int, default=5)
p.add_argument("--log-every", type=int, default=50)
p.add_argument("--overwrite", action="store_true")
p.add_argument("--rank", type=int, default=0, help="worker index; members are split [rank::world_size]")
p.add_argument("--world-size", type=int, default=1)
args = p.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t_start = time.perf_counter()

# ---------------------------------------------------------------- data / model ---
data = Data()
i0 = data.index(args.inference_start)
i_end = data.index(args.inference_end)
L = args.rollout_len if args.rollout_len is not None else i_end - i0 - 1
if i0 + L + 1 > i_end:
    sys.exit(f"rollout-len {L} exceeds the window ({i_end - i0 - 1} steps max)")
frame_dates = data.dates[i0:i0 + L + 2]

model, ctx, ck = build_samudra(args.ckpt_path, device)
film = ck.get("film_config")
if film is not None:
    sys.path.insert(0, str(SRC / "CRPS"))
    from stochastic_samudra import film_weight_norm, sample_z, scale_film
    if film_weight_norm(model) <= 0:
        sys.exit("FiLM weights are all zero after loading: not a trained FiLM checkpoint")
    if args.film_scale != 1.0:
        scale_film(model, args.film_scale)
    z_hw = tuple(ck["z_field_hw"])
print(f"{args.ckpt_path.name} (iter {ck.get('iter')}): "
      + (f"FiLM {film} z_field {z_hw}, scale {args.film_scale:g}" if film else "no FiLM (IC noise only)")
      + f" | init {frame_dates[0]}, {L} steps -> {frame_dates[-1]}", flush=True)

s0, s1 = data.state(i0), data.state(i0 + 1)
args.output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- members --------
for m in list(range(args.n_members))[args.rank::args.world_size]:
    control = args.control_member_0 and m == 0
    index = m if args.control_member_0 else m + 1           # Perlin index; 0 = no noise
    use_film = film is not None and args.film_scale != 0 and not control
    out = args.output_dir / (f"CRPS{'FiLM' if film else ''}_1deg_long_init{frame_dates[0]}_to{frame_dates[-1]}"
                             f"_steps{L}_sigma{args.perturbation_level}_m{m:02d}.zarr")
    if out.exists() and not args.overwrite:
        print(f"skip (exists) {out.name}", flush=True)
        continue
    t0 = time.perf_counter()
    create_zarr(out, data, frame_dates, chunk_frames=args.chunk_frames, attrs={
        "init_date": str(frame_dates[0]), "final_date": str(frame_dates[-1]), "rollout_len": L,
        "step_days": 5, "n_ic_frames": 2,
        "frames": "0-1 = IC states (prediction: perturbed, target: truth); k+2 = prediction of step k",
        "member": m, "n_members": args.n_members, "control": int(control),
        "perturbation_level": args.perturbation_level, "perlin_index": index, "base_seed": args.base_seed,
        "film": str(film) if use_film else "off", "film_scale": args.film_scale, "z_seed": args.z_seed,
        "units": "physical, NaN on land", "checkpoint": str(args.ckpt_path), "bn_stats": "per-sample",
        "model": "SamudraV2 1deg fair-CRPS" + (" + FiLM (per-step z)" if use_film else "") + " + Perlin IC"})

    state = torch.from_numpy(np.concatenate([s0, s1]))[None].to(device)        # (1, 154, Y, X)
    if index > 0 and args.perturbation_level > 0:
        noise = perlin_ic_noise(s0.shape[1:], index, args.base_seed)
        state = state + args.perturbation_level * torch.from_numpy(noise).to(device)
    ic_np = state[0].cpu().numpy()
    buf_pred, buf_tgt, written = [ic_np[:NP], ic_np[NP:]], [s0, s1], 0
    with torch.no_grad():
        for k in range(L):
            f = np.concatenate([data.forc(i0 + k), data.forc(i0 + k + 1)])
            x = torch.cat([state, torch.from_numpy(f)[None].to(device)], 1)
            if use_film:
                z = sample_z([m], film["z_dim"], film["field_ch"], z_hw, device, args.z_seed, k)
                y = model.forward_once(x, ctx, z=z)
            else:
                y = model.forward_once(x, ctx)                                   # FiLM model: z=None = backbone
            state = torch.cat([state[:, NP:], y], 1)
            y_np, truth = y[0].float().cpu().numpy(), data.state(i0 + k + 2)
            buf_pred.append(y_np)
            buf_tgt.append(truth)
            if len(buf_pred) >= args.chunk_frames or k == L - 1:
                written = write_frames(out, data, buf_pred, buf_tgt, written)
                buf_pred, buf_tgt = [], []
            if (k + 1) % args.log_every == 0 or k == L - 1:
                bias, rmse = sst_error(data, y_np, truth)
                print(f"m={m:02d} step {k + 1}/{L} valid={frame_dates[k + 2]}  SST bias={bias:+.2f}C "
                      f"rmse={rmse:.2f}C  {(time.perf_counter() - t0) / (k + 1):.2f}s/step", flush=True)
    print(f"member {m:02d} done in {time.perf_counter() - t0:.1f}s -> {out.name}", flush=True)

print(f"done in {time.perf_counter() - t_start:.1f}s -> {args.output_dir}")
