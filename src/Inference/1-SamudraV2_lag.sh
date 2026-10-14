#!/bin/bash
# 1-degree SamudraV2 lagged-IC ensemble, saved from TEST_START (2014-10-10) on  (driver: 1-SamudraV2_lag.py)
#
#   bash 1-SamudraV2_lag.sh                                   # foreground
#   nohup setsid bash 1-SamudraV2_lag.sh > /dev/null 2>&1 &   # detached; logs in OUT_DIR/_logs
#   GPU_IDS=0 N_MEMBERS=2 ROLLOUT_LEN=4 OUT_DIR=/tmp/smoke bash 1-SamudraV2_lag.sh   # smoke test
#
# One worker per GPU in GPU_IDS; members are split stride-wise (worker r runs members[r::n]).
# Env: GPU_IDS N_MEMBERS ROLLOUT_LEN (empty = whole window 2014-10-10..2022-12-24) OUT_DIR
#      TEST_START LAG_STEPS (5-day steps between member ICs) CKPT
#      EXTRA_ARGS (any other flag of the .py, e.g. "--overwrite"; see python 1-SamudraV2_lag.py --help)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_IDS="${GPU_IDS:-0}"
N_MEMBERS="${N_MEMBERS:-32}"
ROLLOUT_LEN="${ROLLOUT_LEN:-}"
OUT_DIR="${OUT_DIR:-/data/weidong/InMOS_1deg/predictions/SamudraV2_lag}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
TEST_START="${TEST_START:-2014-10-10}"
LAG_STEPS="${LAG_STEPS:-1}"
CKPT="${CKPT:-/data/weidong/InMOS_1deg/ckpt/ema_ckpt.pt}"
set +u
source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate /data/weidong/ocean_env
set -u

IFS=',' read -r -a GPU_ARR <<< "${GPU_IDS}"
NW=${#GPU_ARR[@]}
mkdir -p "${OUT_DIR}/_logs"
STAMP=$(date +%Y%m%d_%H%M%S)
ARGS=(--n-members "${N_MEMBERS}" --output-dir "${OUT_DIR}")
[[ -n "${ROLLOUT_LEN}" ]] && ARGS+=(--rollout-len "${ROLLOUT_LEN}")
ARGS=(--n-members "${N_MEMBERS}" --output-dir "${OUT_DIR}" --test-start "${TEST_START}" --lag-steps "${LAG_STEPS}" --ckpt-path "${CKPT}")
[[ -n "${ROLLOUT_LEN}" ]] && echo "note: ROLLOUT_LEN is ignored here; use EXTRA_ARGS=\"--inference-end YYYY-MM-DD\""
echo "$(basename "1-SamudraV2_lag.py"): ${N_MEMBERS} members on GPUs ${GPU_IDS} -> ${OUT_DIR}  (logs ${OUT_DIR}/_logs/${STAMP}_rank*.log)"

pids=()
for ((r = 0; r < NW; r++)); do
    CUDA_VISIBLE_DEVICES="${GPU_ARR[$r]}" PYTHONUNBUFFERED=1 \
        python "${SCRIPT_DIR}/1-SamudraV2_lag.py" "${ARGS[@]}" --rank "$r" --world-size "${NW}" ${EXTRA_ARGS} \
        > "${OUT_DIR}/_logs/${STAMP}_rank${r}.log" 2>&1 &
    pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "finished (exit ${fail}): $(date '+%F %T')"
exit ${fail}
