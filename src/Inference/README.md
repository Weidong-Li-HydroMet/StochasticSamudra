# 1° inference

This is a 1° port of the inference scripts in `weidong/Sampling/src/Inference`, written on 2026-10-14. It covers the Perlin and lag ensembles, CRPS, Diffusion and SI. The logic follows the 0.25° scripts, but in a simpler form: flat scripts with one shared `common.py`.

| file | port of | what it runs | env |
|---|---|---|---|
| `0-SamudraV2_long.py/.sh` | `0-…long_ensemble` | **Perlin** single-IC ensemble of the deterministic SamudraV2 (member 0 = control). Already existed; unchanged. | ocean_env |
| `0-SamudraV2_multiICs.py/.sh` | `0-…multi_ic` | Deterministic runs from many ICs. Already existed; unchanged. | ocean_env |
| `1-SamudraV2_lag.py/.sh` | `0.5-…lag_ensemble` | **Lagged-IC** ensemble: member m starts (m+1)·lag 5-day steps before `TEST_START` (2014-10-10). Only frames from `TEST_START` on are saved, so all members share one time axis. | ocean_env |
| `2-CRPS_long.py/.sh` | `3-` and `4-…crps…` | **CRPS** Samudra (`src/CRPS`). The checkpoint decides the mode: stage 1 gives Perlin IC noise only; stage 2 (has `film_config`) adds FiLM heads, with z redrawn every step (`sample_z(m, step)`). All members are perturbed (`--control-member-0` makes member 0 a control); `--film-scale` scales the FiLM noise. | ocean_env |
| `3-Diffusion_long.py/.sh` | `1-Long-Diff.py` | SamudraV2 + Perlin IC + **per-step diffusion SDEdit** with the 1° model from `src/Diffusion`, conditioned on the true forcing and date at the valid time. t0 goes 95 → 50 (first 40 steps = 200 d) → 30, out of 100 DDIM steps. Member 0 = control. | ocean_env |
| `4-SI_long.py/.sh` | `5-Long-SamudraSI_NVIDIA.py` | **SI** forecaster (`src/SI`): 2-step DDIM, increment clamped to 8·s_inc, fp32. Spread comes from Perlin IC noise (member m gets the same draw as CRPS member m) plus the SI sampler noise. | **wdenv** |
| `common.py` | — | Data (`SI_data` memmaps), Perlin noise, the SamudraV2 loader (strict load, per-sample BN, FiLM detected automatically), the zarr writer | |

Not ported: DiffusionNext (`2-`), the multi-IC variants of CRPS/SI, and the SI_Ocean DPS refiner (`6-`).

## Output

Every script writes one zarr (v2) per member to `/data/weidong/InMOS_1deg/predictions/<family>/`, with logs in `_logs/`.

- **Variables:** `prediction` and `target` with dimensions (time, var = 77, lat, lon), in physical units, NaN on land.
- **Frames** are 5 days apart:
  - **Long ensembles (`0-`, `2-`, `3-`, `4-`):** frames 0–1 are the IC states (`prediction` holds the perturbed IC) and frame k+2 is step k.
  - **Lag ensemble (`1-`):** frame k is valid at `TEST_START` + 5k days.
- **Default window:** 2014-10-10 .. 2022-12-24, i.e. 598 steps or 8.2 yr.

## Run

```bash
cd /home/nvidia/weidong/InMOS_1degree/src/Inference
GPU_IDS=0,1 N_MEMBERS=32 nohup setsid bash 2-CRPS_long.sh > /dev/null 2>&1 &     # one worker per GPU
GPU_IDS=0 N_MEMBERS=2 ROLLOUT_LEN=4 OUT_DIR=/tmp/smoke bash 4-SI_long.sh         # smoke test
```

Every `.sh` takes `GPU_IDS`, `N_MEMBERS`, `ROLLOUT_LEN` (empty = whole window), `OUT_DIR` and `EXTRA_ARGS` (any flag of the `.py`, e.g. `--overwrite`), plus its own knobs listed in its header. Members are split stride-wise across the workers, and existing stores are skipped.

**Checkpoint defaults:**

| script | default weights |
|---|---|
| `1-` | `ema_ckpt.pt` |
| `2-` | `CRPS/2-FiLM_noise_best_ema.pt` |
| `3-` | Samudra `ema_ckpt.pt` + `Diffusion/condiff_1deg_date_best`. The Diffusion checkpoint does not exist until `src/Diffusion` is trained. |
| `4-` | `SI/SI1deg_step1_best_ema.pt`. After stage 2/3, pass `CKPT=…/SI1deg_replay_best`, a raw accelerate dir. |

## Checks (GPU 0, 2026-10-14)

- **Shared data path:** `common.py` (SI_data) and `0-SamudraV2_long.py` (OM4.zarr) give bitwise identical predictions, targets, dates and coordinates over 6 steps for the control member and the Perlin member. Through `common.py` a SamudraV2 step takes 0.21 s instead of 0.8 s.
- **Smoke runs:** all four launchers exit 0.
  - **Lag:** member ICs and dates are correct.
  - **CRPS:** stage 1 and stage 2 both run, and the FiLM members diverge.
  - **Diffusion:** SDEdit with t0 = 99 (almost no noise) returns to the Samudra forecast. Tested with a throwaway 1-epoch checkpoint.
  - **SI:** runs with the stage-1 checkpoint.
- **Two workers on one GPU:** gives bitwise the same members as one worker.
- **SI environment:** one SI call differs by up to 0.27 between wdenv (torch_harmonics 0.8.0, used in training) and ocean_env (0.9.2, whose noise calibration is 2% larger). That is why `4-` runs in wdenv. `common.py` writes zarr v2 explicitly under wdenv's zarr 3.

Rough cost per member over the full window, single GPU:
- SamudraV2 / CRPS: about 2–4 min.
- SI: about 5–9 min.
- Diffusion: about 10–20 min (SDEdit gets more expensive as t0 drops).
- Storage: about 24 GB uncompressed per member (pred + target, 77 channels).
