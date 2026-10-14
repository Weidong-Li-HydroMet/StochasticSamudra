# 1° stochastic-interpolant (SI) ocean forecaster

This is a 1° port of `/home/nvidia/weidong/Samudra-SI`, written on 2026-10-13. The original directory was not changed.

## Formulation

| item | value |
|---|---|
| state `a_t` | 77 channels (`uo`, `vo`, `thetao`, `so` × 19 levels, plus `zos`). Samudra-normalized **full field**: `(x − mean)/std` with the scalar OM4_means/OM4_stds, land = 0. No climatology is subtracted. |
| one call | `a_t → a_{t+1}`: **one 5-day step**, 77 channels. Samudra-SI was 2-in/2-out with 154 channels and 10 days per call. |
| SI bridge | source `x = a_t`, target `y = a_{t+1}`. x-prediction, DDIM sampler, L0.5 spherical noise (lmax 90), `noise_scales = s_inc`. |
| condition | `[a_t, da_t]` with `da_t = (a_t − a_{t-1}) / s_inc` (154 channels) |
| forcing | `tauuo, tauvo, hfds` at t−1 and t (6 channels), the same as `src/CRPS`. No `hfds_anomalies`. |
| calendar | `[0, doy(t)]` (CalendarEmbedding) |
| backbone | UViT with 2 conv levels (stride 4, 180×360 → 45×90 latent, 3076/4050 tokens kept) around MaskedDiT (768, 12 blocks). 151.6M parameters. **No carry, no big skip:** the decoder head outputs the full field directly. Samudra-SI used `carry(x) + s_inc·g`. |
| loss | `cos(lat)·mask / s_inc²`, so persistence scores about 1 (`LOSS_W_MODE=inc`). `LOSS_W_MODE=full` drops the `1/s_inc²` factor. |
| LR schedule | OneCycleLR, stepped once per optimizer step: `LR/25` → `LR` over the first `PCT_START`=0.3 of the run, then cosine down to `MIN_LR`. Momentum is not cycled. Stage 1 defaults: LR 1e-4, 1500 epochs (≈108k steps); stage 2: LR 1e-5, 30 epochs. |
| resume | `RESUME_FROM=best` (default) continues from `<tag>_best` and re-trains the epochs after it (logged to a new wandb run); `RESUME_FROM=latest` continues from the last epoch. The schedule is rebuilt and fast-forwarded. |
| precision | **strict fp32, always** (no switch). Without the skip, the net has to reproduce `a_t` to about 1e-3, because the deep T/S `s_inc` is 0.0013–0.0018. Rounding the field to bf16 alone costs 1.1 × `s_inc(so_18)` of error. |

The stage-1 output starts at 0 (zero-initialized head), so the initial loss is about 2.5e4 instead of about 1. Early training is about learning the identity map, and the deep T/S channels dominate it.

## Files

| file | what it does |
|---|---|
| `0-prepare_data.py` | Run once (about 8 min). Converts OM4.zarr into `/data/weidong/InMOS_1deg/SI_data/` (`states.npy`, 95 GB; `forcing.npy`; mask; lat; times; `stats.npz` with `s_inc`). `so_13` at 1999-08-01 is NaN everywhere and is filled by time interpolation. |
| `1-SI_step1.py/.sh` | **Stage 1:** single-step SI loss (first call only). Model selection by EMA val loss. A 73-call (1 yr) rollout is logged every 20 epochs. |
| `2-SI_rollout.py/.sh` | **Stage 2:** rollout loss with K=8 calls (40 days). `L = Σ_k λ_k SI_loss(call k)`, λ ∝ k. The calls run on the live model's 2-step DDIM + 8·s_inc clamp, and the gradient flows through all calls (activation-checkpointed). Warm start from the stage-1 `_best_ema.pt`. `_best` is chosen by the rmse77 at the end of a 146-call (2 yr) hindcast. |
| `3-replay_buffer.py/.sh` | **Stage 3:** long-horizon replay, a port of Samudra-SI's 2.x series. Each batch first rolls the live raw weights m calls without gradient from truth (the same 2-step DDIM + clamp operator), then runs the stage-2 K=8 graded unroll from that model-generated state, with the true targets. It is not a stored buffer: the states are regenerated every batch. Curriculum `REPLAY_SCHEDULE=0:0-80,3:60-220,6:120-400` over 9 epochs (calls of 5 d = 0–1.1, 0.8–3.0, 1.6–5.5 yr; the last range matches Samudra-SI 2.3). Warm start from the stage-2 `_best` raw weights. `_best` is chosen by the 8-yr (584-call) hindcast rmse77 of the raw weights. `LOSS_W_SURFV_BOOST=4` optionally reproduces Samudra-SI 2.2/2.3's surface-uo/vo loss boost. |
| `si_common.py` | Shared code: dataset, `bind_call` (builds the cond `[a_t, da_t]`), `masked_si_loss`, `rollout_step`, `rollout_eval`, builders. |
| `uvit.py` | UViT with configurable depth and no carry |
| `masked_dit.py`, `ocean_noise.py`, `modules/` | Copied from Samudra-SI unchanged |
| `ocean_mask.py` | Points at the 1° mask; uses the real latitudes for the area weights |

## Run

```bash
cd /home/nvidia/weidong/InMOS_1degree/src/SI
nohup setsid bash 1-SI_step1.sh > /dev/null 2>&1 &      # GPUs 0-7 by default, log in logs/
# after stage 1:
nohup setsid bash 2-SI_rollout.sh > /dev/null 2>&1 &      # GPUs 1-7 (too big for GPU 0 next to CRPS)
# after stage 2:
nohup setsid bash 3-replay_buffer.sh > /dev/null 2>&1 &   # GPUs 1-7
```

Checkpoints go to `/data/weidong/InMOS_1deg/ckpt/SI/<CKPT_TAG>`. Running again with the same tag resumes from the latest checkpoint.
Smoke test: `GPUS=3 LIMIT_WINDOWS=16 EPOCHS=2 USE_WANDB=0 CKPT_TAG=smoke bash 1-SI_step1.sh`.

## Measured cost (one H100, 2026-10-13, GPUs shared with the SQG jobs)

- **Stage 1**, batch 8, fp32: about 2.4 s per step; 59–60 GB peak with val and diagnostics. About 73 steps per epoch on 7 GPUs.
- **Stage 2**, K=8, fp32: about 40 s per iteration at batch 4 (66.5 GB peak); about 21 s at batch 2 (59 GB).
