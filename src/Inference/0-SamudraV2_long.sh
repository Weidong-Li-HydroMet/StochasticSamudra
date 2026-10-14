#!/bin/bash
# 1-degree SamudraV2: single-IC long Perlin ensemble on GPU 0 (driver: 0-SamudraV2_long.py).
# Member 0 = unperturbed control, members >= 1 = Perlin IC noise (sigma 0.025). Every 5-day state
# + the 2 IC frames, all 77 channels, physical units, one zarr per member.
# Per-sample BatchNorm statistics are applied inside the .py (the stable inference mode).
#
#   bash 0-SamudraV2_long.sh                                   # foreground
#   nohup setsid bash 0-SamudraV2_long.sh > /dev/null 2>&1 &   # detached (log goes to OUT_DIR/_logs)
#   N_MEMBERS=1 ROLLOUT_LEN=4 OUT_DIR=/tmp/smoke bash 0-SamudraV2_long.sh   # smoke test
#
# Size: a full-window member (598 steps) is ~24 GB uncompressed (pred + target, 77 ch).
# Every knob below can be overridden from the environment.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_ID="${GPU_ID:-0}"
CKPT="${CKPT:-/data/weidong/InMOS_1deg/ckpt/ema_ckpt.pt}"
DATA_ROOT="${DATA_ROOT:-/data/weidong/InMOS_1deg/om4_onedeg_v3}"
DATA_SOURCE="${DATA_SOURCE:-zarr}"              # zarr (1958-2022) | cache (ends 2014-10-05)
START="${START:-2014-10-10}"
END="${END:-2022-12-24}"
ROLLOUT_LEN="${ROLLOUT_LEN:-}"                  # empty = whole window (598 steps = 8.2 yr)
N_MEMBERS="${N_MEMBERS:-8}"
SIGMA="${SIGMA:-0.025}"
LOG_EVERY="${LOG_EVERY:-50}"
OUT_DIR="${OUT_DIR:-/data/weidong/InMOS_1deg/predictions/SamudraV2_long}"

source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate /data/weidong/ocean_env
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

mkdir -p "${OUT_DIR}/_logs"
LOG="${OUT_DIR}/_logs/long_$(date +%Y%m%d_%H%M%S).log"
EXTRA=()
[[ -n "${ROLLOUT_LEN}" ]] && EXTRA+=(--rollout-len "${ROLLOUT_LEN}")

echo "ckpt:     ${CKPT}"
echo "data:     ${DATA_ROOT} (${DATA_SOURCE}), window ${START}..${END}"
echo "ensemble: ${N_MEMBERS} members (m00 control), Perlin sigma ${SIGMA}, rollout ${ROLLOUT_LEN:-whole window}"
echo "GPU:      ${GPU_ID}   out: ${OUT_DIR}   log: ${LOG}"

PYTHONUNBUFFERED=1 python "${SCRIPT_DIR}/0-SamudraV2_long.py" \
    --ckpt-path          "${CKPT}" \
    --data-root          "${DATA_ROOT}" \
    --data-source        "${DATA_SOURCE}" \
    --inference-start    "${START}" \
    --inference-end      "${END}" \
    --n-members          "${N_MEMBERS}" \
    --perturbation-level "${SIGMA}" \
    --log-every          "${LOG_EVERY}" \
    --output-dir         "${OUT_DIR}" \
    "${EXTRA[@]}" \
    2>&1 | tee "${LOG}"
