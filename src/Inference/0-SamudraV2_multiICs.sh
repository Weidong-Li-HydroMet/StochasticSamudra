#!/bin/bash
# 1-degree SamudraV2: deterministic multi-IC rollouts on GPU 0 (driver: 0-SamudraV2_multiICs.py).
# Every 5-day state + the 2 IC frames, all 77 channels, physical units, one zarr per IC.
# Per-sample BatchNorm statistics are applied inside the .py (the stable inference mode).
#
#   bash 0-SamudraV2_multiICs.sh                                   # foreground
#   nohup setsid bash 0-SamudraV2_multiICs.sh > /dev/null 2>&1 &   # detached (log goes to OUT_DIR/_logs)
#   MAX_ICS=2 ROLLOUT_LEN=4 OUT_DIR=/tmp/smoke bash 0-SamudraV2_multiICs.sh   # smoke test
#
# Every knob below can be overridden from the environment.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_ID="${GPU_ID:-0}"
CKPT="${CKPT:-/data/weidong/InMOS_1deg/ckpt/ema_ckpt.pt}"
DATA_ROOT="${DATA_ROOT:-/data/weidong/InMOS_1deg/om4_onedeg_v3}"
DATA_SOURCE="${DATA_SOURCE:-zarr}"              # zarr (1958-2022) | cache (ends 2014-10-05)
START="${START:-2014-10-10}"
END="${END:-2022-12-24}"
IC_STRIDE="${IC_STRIDE:-6}"                     # steps between ICs (6 x 5 d = 30 d)
ROLLOUT_LEN="${ROLLOUT_LEN:-20}"                # steps per IC (20 x 5 d = 100 d)
MAX_ICS="${MAX_ICS:-}"                          # empty = every IC that fits in the window
OUT_DIR="${OUT_DIR:-/data/weidong/InMOS_1deg/predictions/SamudraV2_multiICs}"

source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate /data/weidong/ocean_env
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

mkdir -p "${OUT_DIR}/_logs"
LOG="${OUT_DIR}/_logs/multiICs_$(date +%Y%m%d_%H%M%S).log"
EXTRA=()
[[ -n "${MAX_ICS}" ]] && EXTRA+=(--max-ics "${MAX_ICS}")

echo "ckpt: ${CKPT}"
echo "data: ${DATA_ROOT} (${DATA_SOURCE}), window ${START}..${END}"
echo "ICs:  stride ${IC_STRIDE} steps, rollout ${ROLLOUT_LEN} steps, max ${MAX_ICS:-all}"
echo "GPU:  ${GPU_ID}   out: ${OUT_DIR}   log: ${LOG}"

PYTHONUNBUFFERED=1 python "${SCRIPT_DIR}/0-SamudraV2_multiICs.py" \
    --ckpt-path       "${CKPT}" \
    --data-root       "${DATA_ROOT}" \
    --data-source     "${DATA_SOURCE}" \
    --inference-start "${START}" \
    --inference-end   "${END}" \
    --ic-stride       "${IC_STRIDE}" \
    --rollout-len     "${ROLLOUT_LEN}" \
    --output-dir      "${OUT_DIR}" \
    "${EXTRA[@]}" \
    2>&1 | tee "${LOG}"
