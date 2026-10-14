# CRPS fine-tuning of the 1° Samudra

This is a 1° port of `/home/nvidia/weidong/CRPS` (the Sampling/Derecho code). Changed lines are marked `1deg:`.
The original 0.25° documentation stays in that folder.

| stage | what it does | files |
|---|---|---|
| 1 IC noise | Deterministic Samudra trained with fair CRPS. Spread comes only from Perlin IC noise (σ 0.025). | `1-IC_noise.py/.sh` |
| 2 FiLM noise | FiLM latent noise `z` (redrawn every step) added on top of the IC noise, which learns model error. 2a: backbone frozen for the first 800 iters; 2b: joint training. | `2-FiLM_noise.py/.sh`, `stochastic_samudra.py` |
| 3 selection | Re-score the saved checkpoints on 40 ICs over 2013–2022, sweep the FiLM strength, and bake the chosen strength into the weights. | `3-eval_candidates.py/.sh`, `3-scale_sweep.sh`, `3-analyze_scale_sweep.py`, `3-bake_film_scale.py` |

Shared modules: `crps_loss.py` (fair CRPS) and `samudra_1deg.py` (model and data for 1°). There is also
`1-check_stability.py` (long-rollout check) and `run_1_then_2.sh` (runs stage 1 then stage 2).

## Run

```bash
cd /home/nvidia/weidong/InMOS_1degree/src/CRPS
bash run_1_then_2.sh      # stage 1 then 2, detaches itself, prints the log path
```

Defaults: GPU 0, M = 4 members in one process, 4 ICs per iteration, 3000 iters for stage 1 and 6000 for stage 2.
To run one stage only: `nohup setsid bash -l 1-IC_noise.sh > log 2>&1 &` (the same for `2-FiLM_noise.sh` and `3-*.sh`).
Settings can be overridden from the environment, e.g. `ITERS=1000 ICS_PER_ITER=2 bash run_1_then_2.sh`.
Also available: `GPU_IDS`, `MEMBERS_PER_RANK` (M = number of GPUs × this), `OUT_DIR`, and `RESUME=<OUT_DIR>/<stage>_best.pt`.
Cost: about 21.6 GB and 35–40 s per iteration at M = 4 with 4 ICs per iteration.

## Rules

- **BatchNorm uses per-sample statistics** (`--bn-mode sample`, the default). The 1° net was trained this way.
  With the checkpoint's running stats, the base model blows up after about 3.5 years.
- **Training forward = inference forward.** Checked on 2026-10-12 for the same IC and 20 steps,
  with a control member and a Perlin member: dates, BN statistics, predictions, targets, and train() vs eval()
  were all bitwise identical to `src/Inference/0-SamudraV2_long.py`.
- **Checkpoints:** all stages write to one folder, `/data/weidong/InMOS_1deg/ckpt/CRPS/`, with a stage prefix:
  - each gate PASS → `<stage>_it<NNNN>_ema.pt`;
  - `<stage>_best_ema.pt` is a symlink to the newest PASS;
  - `<stage>_best.pt` is the resume state;
  - nothing is saved on a SKIP.
- **Gate thresholds:**
  - stage 1: CRPS below the baseline and below the best so far, and RMSE ≤ 1.05 × baseline;
  - stage 2: CRPS at least 0.5 % below the baseline and below the best so far, and RMSE ≤ 1.5 × baseline (a divergence guard);
  - change them with `--rmse-tolerance` and `--min-crps-gain`.
- **Loading FiLM checkpoints:** call `make_stochastic(model, **ck["film_config"])` before `load_state_dict`.
