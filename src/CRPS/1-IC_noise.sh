#!/bin/bash
# === InMOS_1degree copy (2026-10-12) -- STAGE 1 (deterministic fair-CRPS): local 8xH100 box, no scheduler. Changed lines are
# marked "1deg:". Run detached:  nohup setsid bash -l <this>.sh > <log> 2>&1 & disown
# Default: GPU 0, M = 4 members in one process. Override: GPU_IDS=0,1,2,3 MEMBERS_PER_RANK=1 (M = NGPU x MEMBERS_PER_RANK).
#
# Fair-CRPS fine-tuning of SamudraV2 on one 4-GPU node.
# 4 ranks x 1 member each = 4-member training ensemble (ensemble-parallel:
# one member per GPU; member predictions are all-gathered per step for the
# CRPS spread term).
#
# Usage:
#   ./1-IC_noise.sh

set -euo pipefail

# Under PBS the script runs from a spool copy in $HOME, so BASH_SOURCE does
# not point at src/CRPS — fall back to the repo path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # 1deg
PY_SCRIPT="${SCRIPT_DIR}/1-IC_noise.py"   # 1deg: renamed
if [[ ! -f "${PY_SCRIPT}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # 1deg
    PY_SCRIPT="${SCRIPT_DIR}/1-IC_noise.py"   # 1deg: renamed
fi
# if [[ -n "${PBS_O_WORKDIR:-}" ]]; then
#     cd "${PBS_O_WORKDIR}"
# fi

# ---- configuration -----------------------------------------------------------
OUT_DIR="${OUT_DIR:-/data/weidong/InMOS_1deg/ckpt/CRPS}"   # 1deg: ONE folder for all CRPS stages (files prefixed 1-/2-/3-)
LOG_DIR="${OUT_DIR}/_logs"

ITERS="${ITERS:-3000}"   # 1deg: fixed length, env-overridable; only gate-PASS checkpoints are saved
# Resume from a ckpt saved by this script (restores optimizer, EMA, dynamic
# weights, val baseline, best_crps, and the iteration counter). Leave empty
# for a fresh run, or override at launch:
#   RESUME="${OUT_DIR}/1-IC_noise_best.pt" bash 1-IC_noise.sh   # 1deg: last gate-PASS state (no ckpt_latest any more)
RESUME="${RESUME:-}"
ROLLOUT_STEPS=8          # 1deg: 8 one-step (5 d) calls = 40 days of rollout per iteration
ICS_PER_ITER="${ICS_PER_ITER:-4}"   # 1deg: batch size = ICs per iteration (each with M members)
MEMBERS_PER_RANK="${MEMBERS_PER_RANK:-4}"   # 1deg: default M = 1 GPU x 4 members; global M = NGPU * this
# Backward + state detach every N steps (truncated BPTT). 1 fits comfortably
# on a 40GB A100 (also when the node is shared); raise to 2 or 0 (=full BPTT
# through all rollout steps) on empty 80GB nodes for cross-step spread
# gradients at the cost of ~N x the activation memory.
DETACH_INTERVAL=1
PERTURBATION_LEVEL=0.025 # matches the long_perturb inference ensembles
PER_STEP_NOISE=0.0       # >0 = SPPT-like Perlin on the state every step
                         # (inference must then inject the same noise)
LR=1e-5
WARMUP_ITERS=50
EMA_DECAY=0.99           # ~100-iter window; 0 disables EMA
VAL_EVERY=100
VAL_STEPS=16             # 1deg: validation lead up to 16 x 5 d = 80 days
SAVE_EVERY=100
SEED=2025
BN_MODE="${BN_MODE:-sample}"   # 1deg: per-sample BN statistics (as trained; the stable mode). eval = running stats, train = PyTorch default

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
LOG="${LOG_DIR}/1-IC_noise_train_$(date +%Y%m%d_%H%M%S).log"   # 1deg

echo "script:   ${PY_SCRIPT}"
echo "out dir:  ${OUT_DIR}"
echo "BN mode:  ${BN_MODE}   GPUs: ${CUDA_VISIBLE_DEVICES}   batch: ${ICS_PER_ITER} ICs/iter"
echo "log:      ${LOG}"
echo "ensemble: M = ${NGPU} x ${MEMBERS_PER_RANK}, sigma=${PERTURBATION_LEVEL}"
echo "rollout:  ${ROLLOUT_STEPS} steps/iter, ${ITERS} iters, lr=${LR}"
RESUME_ARGS=()
if [[ -n "${RESUME}" ]]; then
    echo "resume:   ${RESUME}"
    RESUME_ARGS=(--resume "${RESUME}")
fi

PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
${LAUNCH} \
    "${PY_SCRIPT}" \
    --iters              "${ITERS}" \
    --rollout-steps      "${ROLLOUT_STEPS}" \
    --detach-interval    "${DETACH_INTERVAL}" \
    --members-per-rank   "${MEMBERS_PER_RANK}" \
    --ics-per-iter       "${ICS_PER_ITER}" \
    --perturbation-level "${PERTURBATION_LEVEL}" \
    --per-step-noise     "${PER_STEP_NOISE}" \
    --lr                 "${LR}" \
    --warmup-iters       "${WARMUP_ITERS}" \
    --ema-decay          "${EMA_DECAY}" \
    --val-every          "${VAL_EVERY}" \
    --val-steps          "${VAL_STEPS}" \
    --save-every         "${SAVE_EVERY}" \
    --seed               "${SEED}" \
    --bn-mode            "${BN_MODE}" \
    --out-dir            "${OUT_DIR}" \
    "${RESUME_ARGS[@]}" \
    2>&1 | tee "${LOG}"
