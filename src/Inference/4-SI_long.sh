#!/bin/bash
# 1-degree SI forecaster long ensemble (Perlin IC + SI sampler noise; 2-step DDIM, clamp 8 s_inc)  (driver: 4-SI_long.py)
#
#   bash 4-SI_long.sh                                   # foreground
#   nohup setsid bash 4-SI_long.sh > /dev/null 2>&1 &   # detached; logs in OUT_DIR/_logs
#   GPU_IDS=0 N_MEMBERS=2 ROLLOUT_LEN=4 OUT_DIR=/tmp/smoke bash 4-SI_long.sh   # smoke test
#
# One worker per GPU in GPU_IDS; members are split stride-wise (worker r runs members[r::n]).
# Env: GPU_IDS N_MEMBERS ROLLOUT_LEN (empty = whole window 2014-10-10..2022-12-24) OUT_DIR
#      CKPT (*_ema.pt or raw dir; config from *_meta.pt) SIGMA SI_STEPS INC_CLAMP
#      EXTRA_ARGS (any other flag of the .py, e.g. "--overwrite"; see python 4-SI_long.py --help)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_IDS="${GPU_IDS:-0}"
N_MEMBERS="${N_MEMBERS:-32}"
ROLLOUT_LEN="${ROLLOUT_LEN:-}"
OUT_DIR="${OUT_DIR:-/data/weidong/InMOS_1deg/predictions/SI_long}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
CKPT="${CKPT:-/data/weidong/InMOS_1deg/ckpt/SI/SI1deg_step1_best_ema.pt}"
SIGMA="${SIGMA:-0.025}"
SI_STEPS="${SI_STEPS:-2}"
INC_CLAMP="${INC_CLAMP:-8}"
set +u
source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate /data/weidong/wdenv     # the SI training env: torch_harmonics 0.8.0 (ocean_env's 0.9.2 changes the SI noise)
set -u

IFS=',' read -r -a GPU_ARR <<< "${GPU_IDS}"
NW=${#GPU_ARR[@]}
mkdir -p "${OUT_DIR}/_logs"
STAMP=$(date +%Y%m%d_%H%M%S)
ARGS=(--n-members "${N_MEMBERS}" --output-dir "${OUT_DIR}")
[[ -n "${ROLLOUT_LEN}" ]] && ARGS+=(--rollout-len "${ROLLOUT_LEN}")
ARGS+=(--ckpt-path "${CKPT}" --perturbation-level "${SIGMA}" --si-steps "${SI_STEPS}" --inc-clamp "${INC_CLAMP}")
echo "$(basename "4-SI_long.py"): ${N_MEMBERS} members on GPUs ${GPU_IDS} -> ${OUT_DIR}  (logs ${OUT_DIR}/_logs/${STAMP}_rank*.log)"

pids=()
for ((r = 0; r < NW; r++)); do
    CUDA_VISIBLE_DEVICES="${GPU_ARR[$r]}" PYTHONUNBUFFERED=1 \
        python "${SCRIPT_DIR}/4-SI_long.py" "${ARGS[@]}" --rank "$r" --world-size "${NW}" ${EXTRA_ARGS} \
        > "${OUT_DIR}/_logs/${STAMP}_rank${r}.log" 2>&1 &
    pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "finished (exit ${fail}): $(date '+%F %T')"
exit ${fail}
