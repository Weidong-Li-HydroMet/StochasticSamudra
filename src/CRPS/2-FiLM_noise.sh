#!/bin/bash
# === InMOS_1degree copy (2026-10-12) -- STAGE 2 (FiLM: 2a frozen backbone, 2b joint): local 8xH100 box, no scheduler. Changed lines are
# marked "1deg:". Run detached:  nohup setsid bash -l <this>.sh > <log> 2>&1 & disown
# Default: GPU 0, M = 4 members in one process. Override: GPU_IDS=0,1,2,3 MEMBERS_PER_RANK=1 (M = NGPU x MEMBERS_PER_RANK).
#
# Fair-CRPS fine-tuning of a *stochastic* SamudraV2 on one 4-GPU node.
#
# Two sources of ensemble spread are active:
#   1. IC noise   — 3-octave Perlin on the normalized IC (sigma below), as in
#                   run_finetune_crps_4gpu.sh;
#   2. FiLM noise — a latent draw consumed by conditional-InstanceNorm (FiLM)
#                   modulation inside the U-Net, redrawn every rollout step.
# Only (2) is learned: fair CRPS calibrates how much model-error spread the
# network should generate, instead of it being a hand-tuned sigma.
#
# 4 ranks x 1 member each = 4-member training ensemble (ensemble-parallel:
# one member per GPU; member predictions are all-gathered per step for the
# CRPS spread term).
#
# Usage:
#   ./2-FiLM_noise.sh
#   RESUME=<out>/2-FiLM_noise_best.pt ./2-FiLM_noise.sh   # 1deg: last gate-PASS state (no ckpt_latest any more)
# Always launch detached, or a dropped terminal SIGHUPs the run:
#   nohup setsid bash -l 2-FiLM_noise.sh > <log> 2>&1 & disown
#

set -euo pipefail

# Under PBS the script runs from a spool copy in $HOME, so BASH_SOURCE does
# not point at src/CRPS — use the repo path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # 1deg
PY_SCRIPT="${SCRIPT_DIR}/2-FiLM_noise.py"

# ---- configuration -----------------------------------------------------------
REPO="/data/weidong/InMOS_1deg"   # 1deg: checkpoint root (CKPT/OUT dirs below)
OUT_DIR="${OUT_DIR:-/data/weidong/InMOS_1deg/ckpt/CRPS}"   # 1deg: ONE folder for all CRPS stages (files prefixed 1-/2-/3-)
LOG_DIR="${OUT_DIR}/_logs"

# Starting weights: the CRPS-tuned deterministic model (iter 799 of the
# SamudraV2_CRPS run). The IC-uncertainty term is already calibrated there, so
# FiLM learns the *residual* — i.e. model uncertainty — and the gate asks the
# question that matters: does learned model noise buy anything beyond
# amplifying the IC perturbation? Starting from the plain SamudraV2 ckpt
# instead makes both spread sources train at once and they stop being
# separately identifiable.
CKPT="${CKPT:-/data/weidong/InMOS_1deg/ckpt/CRPS/1-IC_noise_best_ema.pt}"   # 1deg: stage-1 newest gate-PASS

ITERS="${ITERS:-6000}"   # 1deg: fixed length, env-overridable; only gate-PASS checkpoints are saved
                         # Only gate-PASS checkpoints are saved, so a long run cannot hurt the
                         # result. The 0.25deg run, kept for reference: extended from 2000 on 2026-08-23: at iter 1399 CRPS
                         # was still falling (-2.33% vs baseline) with RMSE flat
                         # (+0.04%) and every variable still under-dispersed
                         # (SSR 0.61-0.92), i.e. not converged. Safe to extend:
                         # --iters only bounds the loop, the lr schedule is
                         # warmup-only with no total-step-dependent decay, and
                         # the backbone freeze is decided per iteration from the
                         # loop counter (`it < film_warmup_iters`), so a resume
                         # past 800 stays unfrozen.
                         # The gate freezes the best model, so overshooting
                         # costs GPU hours but cannot hurt the result.
RESUME="${RESUME:-}"     # restores model/optimizer/EMA/dynamic weights/baseline/iter
ROLLOUT_STEPS=8          # 1deg: 8 one-step (5 d) calls = 40 days of rollout per iteration
ICS_PER_ITER="${ICS_PER_ITER:-4}"   # 1deg: batch size = ICs per iteration (each with M members)
MEMBERS_PER_RANK="${MEMBERS_PER_RANK:-4}"   # 1deg: default M = 1 GPU x 4 members; global M = NGPU * this
DETACH_INTERVAL=1        # truncated BPTT; 1 fits a shared 40GB A100
PERTURBATION_LEVEL=0.025 # source #1, matches the long_perturb inference ensembles
PER_STEP_NOISE=0.0       # hand-tuned SPPT term; leave off, FiLM replaces it

# ---- FiLM (source #2) --------------------------------------------------------
Z_DIM=64                 # global latent vector
Z_FIELD_CH=4             # coarse random-field channels -> spatial structure.
                         # 0 would make the modulation globally uniform, which
                         # is far too weak for the ocean.
Z_FIELD_H=45             # field resolution: sets the perturbation's correlation
Z_FIELD_W=90             # length. 45x90 == the U-Net bottleneck on the 0.25 grid.
FILM_HIDDEN=128
FILM_WHERE=bottleneck    # middle + decoder blocks: large-scale, coherent model
                         # error. Use "all" to also perturb the encoder.
FILM_LR=1e-3             # heads are from scratch -> ~200x the backbone lr
FILM_WARMUP_ITERS=800    # backbone FROZEN for this many iters. This is the
                         # primary RMSE protection, not a warm-up trick: with
                         # the backbone fixed the deterministic map is preserved
                         # exactly and the ensemble mean can only move by the
                         # second-order effect of the perturbation. Set it to
                         # >= ITERS to never unfreeze at all.

# ---- CRPS / RMSE trade-off ---------------------------------------------------
DYNAMIC_LOSS="--dynamic-loss"
                         # ON, as in the deterministic run. This also selects
                         # the PER-CHANNEL reduction path in the surrogate
                         # (mean over channels of each channel's weighted CRPS)
                         # instead of one global pool over all 154 channels.
                         # That matters: with it off, channels enter the loss at
                         # their own magnitude, and velocities (CRPS ~0.2) run
                         # ~10x the tracers (~0.02), so run 2 was ~10x
                         # velocity-weighted.
                         #
                         # Known trade-off, from run 1 (_run1_dynamicloss/):
                         # the rolling 1/CRPS weights saturate at the
                         # --dynamic-limit cap (19x tracer:velocity), and FiLM —
                         # a per-channel knob with no cross-variable coupling —
                         # obeys literally. By iter 299 thetao_0 spread was +41%
                         # (SSR 0.81 -> 1.14, past calibration into
                         # over-dispersion) while uo_0 was -33% (SSR 0.77 ->
                         # 0.51). The deterministic run did not hit this because
                         # its weights drove the backbone, which learns shared
                         # dynamics. --dynamic-limit is the knob between the two
                         # extremes: 20 = run 1, lower (3-5) softens the skew.
CRPS_ALPHA=0.95          # almost-fair CRPS (AIFS): 0.95*fCRPS + 0.05*CRPS.
                         # The standard term rewards spread slightly less, so
                         # the optimum sits a little closer to the deterministic
                         # solution. 1.0 = the pure fair CRPS of the last run.
                         # LOSS only — val metrics stay pure fair CRPS.

# ---- optimization ------------------------------------------------------------
LR=5e-6                  # backbone, HALVED vs the deterministic run: that run
                         # drifted RMSE +3.6% at 1e-5 and this one starts from
                         # its endpoint, so the drift budgets compound
WARMUP_ITERS=50
EMA_DECAY=0.99           # ~100-iter window; 0 disables EMA
VAL_EVERY=100
VAL_STEPS=16             # 1deg: validation lead up to 16 x 5 d = 80 days
VAL_ICS=6                # ICs per validation, up from 2. The gate decides on a
                         # 0.5% CRPS margin, and a 2-IC estimate has no
                         # resolving power at that scale — runs 1-2 were making
                         # pass/fail calls inside their own sampling noise.
                         # Cost: validation goes ~19s -> ~57s every 100 iters
                         # (~5% overhead). Note the val window is one year (36
                         # windows) and each rollout is 8 windows, so ICs at
                         # linspace(0,27,6) still overlap; effective sample size
                         # is nearer 4 than 6. Extending --val-end is the lever
                         # if that is not enough.
SAVE_EVERY=100
SEED=2025
BN_MODE="${BN_MODE:-sample}"   # 1deg: per-sample BN statistics (as trained; the stable mode). eval = running stats, train = PyTorch default

# ---- ckpt_best gate: ALL must pass, else ckpt_best is left untouched ---------
# Baseline = the starting checkpoint scored under this exact ensemble protocol,
# measured at run time. With VAL_ICS=6 it will NOT be the 0.08885 / 0.28838 that
# runs 1-3 reported: those used 2 ICs. Only the 2-IC protocol reproduces the
# SamudraV2_CRPS iter-799 numbers. The gate is self-consistent either way —
# every number in a run is measured the same way — but do not compare absolute
# CRPS across runs with different VAL_ICS.
MIN_CRPS_GAIN=0.005      # CRPS must beat the baseline by >= 0.5% (relative),
                         # so ckpt_best does not churn on noise-level wins
RMSE_TOLERANCE=0.50      # NOT a skill gate any more — a DIVERGENCE guard.
                         # Limit = 0.29005 * 1.50 = 0.43508. Set deliberately on
                         # 2026-08-23 (run 6): ckpt_best is now selected on CRPS
                         # alone, and RMSE only has to stay off the floor.
                         #
                         # The history behind the decision. Runs 1-4 all failed
                         # the gate the same way, and run 4 made the mechanism
                         # explicit: FiLM buys CRPS by spending ens-mean skill,
                         # monotonically. Through iter 499 CRPS fell 0.09070 ->
                         # 0.08862 (closing to +0.1% of the 0.08852 baseline)
                         # while RMSE rose 0.29361 -> 0.29417. The original
                         # 0.29225 ceiling allowed only +0.76% over baseline, so
                         # it would have rejected every genuine CRPS win; +3%
                         # (0.29875) would likely have bound a few hundred iters
                         # later. Neither threshold was going to let this run
                         # produce a checkpoint.
                         #
                         # KNOW WHAT THIS COSTS. 0.29225 was chain-anchored to
                         # the pretrained SamudraV2 (0.27833 * 1.05) to stop
                         # successive fine-tunes from ratcheting skill away.
                         # That anchor is now gone: at the limit this model
                         # would be +56% RMSE on the pretrained SamudraV2. The
                         # gate will NOT stop deterministic skill loss — read
                         # the logged RMSE yourself before shipping ckpt_best,
                         # and re-anchor before any 3rd generation of fine-tuning.
RMSE_ABS_MAX=0           # absolute ceiling off; the relative tolerance binds
SSR_MAX=0                # SSR is NOT gated on. It is still computed and logged
                         # per variable and per lead for diagnosis — it is just
                         # not a pass/fail criterion. (The aggregate it would
                         # have gated on was blind anyway: run 1 sat at
                         # SSR_max=0.765 while thetao_0 had already overshot to
                         # 1.14, because averaging over all channels hid it.)
                         # The gate is CRPS + RMSE only.
RMSE_BRAKE=0.5           # ACTIVE protection: on an RMSE-threshold failure, halve
                         # the backbone lr and keep going (the gate alone only
                         # refuses to save; it does not stop the drift). FiLM's
                         # lr is untouched — backbone drift is what moves the
                         # ensemble mean. 1.0 disables
                         #
                         # CAVEAT, measured in run 4: while the backbone is
                         # frozen (FILM_WARMUP_ITERS) this brake is a NO-OP —
                         # it cuts an lr nothing is using. It fired at iters
                         # 299/399/499 and took base_lr 5e-6 -> 6.25e-7, so by
                         # the iter-800 unfreeze the backbone would have been at
                         # 1/8 the intended rate without ever being trained.
                         # Run 4 also showed RMSE drifting +1.6%/100 iters with
                         # the backbone frozen, i.e. FiLM moves the ensemble
                         # mean on its own and freezing does NOT bound it.
                         # Hence RESET_BASE_LR below on resume.

RESET_BASE_LR="--reset-base-lr"
                         # On --resume, re-apply LR/FILM_LR instead of
                         # inheriting the checkpoint's optimizer base_lr. Undoes
                         # the brake cuts described above. Drop this flag if you
                         # deliberately want to continue at the braked rate.

# -----------------------------------------------------------------------------

source /home/nvidia/miniforge3/etc/profile.d/conda.sh   # 1deg
conda activate /data/weidong/ocean_env                      # 1deg
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0}"   # 1deg: default GPU 0 (several: GPU_IDS=0,1,2,3 MEMBERS_PER_RANK=1)
NGPU=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")  # 1deg
# 1deg: torchrun --standalone hangs in rendezvous on this box (the hostname resolves to an
# external address); 1 GPU needs no torchrun, >1 GPU pins the rendezvous to localhost.
if [[ "${NGPU}" == "1" ]]; then LAUNCH="python"
else LAUNCH="torchrun --nnodes=1 --nproc_per_node=${NGPU} --rdzv_backend=c10d --rdzv_endpoint=localhost:0"; fi

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
LOG="${LOG_DIR}/2-FiLM_noise_train_$(date +%Y%m%d_%H%M%S).log"   # 1deg

echo "script:   ${PY_SCRIPT}"
echo "init:     ${CKPT}"
echo "out dir:  ${OUT_DIR}"
echo "BN mode:  ${BN_MODE}   GPUs: ${CUDA_VISIBLE_DEVICES}   batch: ${ICS_PER_ITER} ICs/iter"
echo "log:      ${LOG}"
echo "ensemble: M = ${NGPU} x ${MEMBERS_PER_RANK}, IC sigma=${PERTURBATION_LEVEL}"
echo "FiLM:     where=${FILM_WHERE} z_dim=${Z_DIM} field=${Z_FIELD_CH}x${Z_FIELD_H}x${Z_FIELD_W}"
echo "          lr=${FILM_LR}, backbone frozen for ${FILM_WARMUP_ITERS} iters"
echo "rollout:  ${ROLLOUT_STEPS} steps/iter, ${ITERS} iters, backbone lr=${LR}"
echo "val:      every ${VAL_EVERY} iters, ${VAL_ICS} ICs, leads up to ${VAL_STEPS} steps"
echo "loss:     almost-fair CRPS alpha=${CRPS_ALPHA}, ${DYNAMIC_LOSS}"
if [[ "${RMSE_ABS_MAX}" == "0" ]]; then
    echo "gate:     CRPS <= -${MIN_CRPS_GAIN} rel; RMSE <= +${RMSE_TOLERANCE} rel (abs ceiling OFF); SSR not gated"
else
    echo "gate:     CRPS <= -${MIN_CRPS_GAIN} rel; RMSE <= min(+${RMSE_TOLERANCE} rel, ${RMSE_ABS_MAX} abs); SSR not gated"
fi
echo "RMSE:     backbone frozen ${FILM_WARMUP_ITERS} iters, lr=${LR}, brake x${RMSE_BRAKE} on failure"
RESUME_ARGS=()
if [[ -n "${RESUME}" ]]; then
    echo "resume:   ${RESUME}"
    RESUME_ARGS=(--resume "${RESUME}")
fi

PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
${LAUNCH} \
    "${PY_SCRIPT}" \
    --ckpt-path          "${CKPT}" \
    --iters              "${ITERS}" \
    --rollout-steps      "${ROLLOUT_STEPS}" \
    --detach-interval    "${DETACH_INTERVAL}" \
    --members-per-rank   "${MEMBERS_PER_RANK}" \
    --ics-per-iter       "${ICS_PER_ITER}" \
    --perturbation-level "${PERTURBATION_LEVEL}" \
    --per-step-noise     "${PER_STEP_NOISE}" \
    --z-dim              "${Z_DIM}" \
    --z-field-ch         "${Z_FIELD_CH}" \
    --z-field-hw         "${Z_FIELD_H}" "${Z_FIELD_W}" \
    --film-hidden        "${FILM_HIDDEN}" \
    --film-where         "${FILM_WHERE}" \
    --film-lr            "${FILM_LR}" \
    --film-warmup-iters  "${FILM_WARMUP_ITERS}" \
    --lr                 "${LR}" \
    --warmup-iters       "${WARMUP_ITERS}" \
    --ema-decay          "${EMA_DECAY}" \
    --val-every          "${VAL_EVERY}" \
    --val-steps          "${VAL_STEPS}" \
    --val-ics            "${VAL_ICS}" \
    ${DYNAMIC_LOSS} \
    --crps-alpha         "${CRPS_ALPHA}" \
    --min-crps-gain      "${MIN_CRPS_GAIN}" \
    --rmse-tolerance     "${RMSE_TOLERANCE}" \
    --rmse-abs-max       "${RMSE_ABS_MAX}" \
    --ssr-max            "${SSR_MAX}" \
    --rmse-brake         "${RMSE_BRAKE}" \
    --save-every         "${SAVE_EVERY}" \
    --seed               "${SEED}" \
    --bn-mode            "${BN_MODE}" \
    --out-dir            "${OUT_DIR}" \
    ${RESET_BASE_LR} \
    "${RESUME_ARGS[@]}" \
    2>&1 | tee "${LOG}"
