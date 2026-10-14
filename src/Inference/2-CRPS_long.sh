#!/bin/bash
# 1-degree CRPS SamudraV2 long ensemble (Perlin IC + per-step FiLM z for stage-2 checkpoints)  (driver: 2-CRPS_long.py)
#
#   bash 2-CRPS_long.sh                                   # foreground
#   nohup setsid bash 2-CRPS_long.sh > /dev/null 2>&1 &   # detached; logs in OUT_DIR/_logs
#   GPU_IDS=0 N_MEMBERS=2 ROLLOUT_LEN=4 OUT_DIR=/tmp/smoke bash 2-CRPS_long.sh   # smoke test
#
# One worker per GPU in GPU_IDS; members are split stride-wise (worker r runs members[r::n]).
# Env: GPU_IDS N_MEMBERS ROLLOUT_LEN (empty = whole window 2014-10-10..2022-12-24) OUT_DIR
#      CKPT (stage 2 FiLM default; a 1-IC_noise_*_ema.pt runs stage 1) SIGMA
#      EXTRA_ARGS (any other flag of the .py, e.g. "--overwrite"; see python 2-CRPS_long.py --help)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_IDS="${GPU_IDS:-0}"
N_MEMBERS="${N_MEMBERS:-32}"
ROLLOUT_LEN="${ROLLOUT_LEN:-}"
OUT_DIR="${OUT_DIR:-/data/weidong/InMOS_1deg/predictions/CRPS_long}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
CKPT="${CKPT:-/data/weidong/InMOS_1deg/ckpt/CRPS/2-FiLM_noise_best_ema.pt}"
SIGMA="${SIGMA:-0.025}"
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
ARGS+=(--ckpt-path "${CKPT}" --perturbation-level "${SIGMA}")
echo "$(basename "2-CRPS_long.py"): ${N_MEMBERS} members on GPUs ${GPU_IDS} -> ${OUT_DIR}  (logs ${OUT_DIR}/_logs/${STAMP}_rank*.log)"

pids=()
for ((r = 0; r < NW; r++)); do
    CUDA_VISIBLE_DEVICES="${GPU_ARR[$r]}" PYTHONUNBUFFERED=1 \
        python "${SCRIPT_DIR}/2-CRPS_long.py" "${ARGS[@]}" --rank "$r" --world-size "${NW}" ${EXTRA_ARGS} \
        > "${OUT_DIR}/_logs/${STAMP}_rank${r}.log" 2>&1 &
    pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "finished (exit ${fail}): $(date '+%F %T')"
exit ${fail}
