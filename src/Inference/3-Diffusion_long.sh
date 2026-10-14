#!/bin/bash
# 1-degree SamudraV2 + per-step diffusion SDEdit long ensemble (member 0 = control)  (driver: 3-Diffusion_long.py)
#
#   bash 3-Diffusion_long.sh                                   # foreground
#   nohup setsid bash 3-Diffusion_long.sh > /dev/null 2>&1 &   # detached; logs in OUT_DIR/_logs
#   GPU_IDS=0 N_MEMBERS=2 ROLLOUT_LEN=4 OUT_DIR=/tmp/smoke bash 3-Diffusion_long.sh   # smoke test
#
# One worker per GPU in GPU_IDS; members are split stride-wise (worker r runs members[r::n]).
# Env: GPU_IDS N_MEMBERS ROLLOUT_LEN (empty = whole window 2014-10-10..2022-12-24) OUT_DIR
#      CKPT (Samudra) DIFF_CKPT SIGMA SDE_STEPS SDE_T0_0 SDE_T0_MID SDE_T0_1 SDE_PHASE1_STEPS
#      EXTRA_ARGS (any other flag of the .py, e.g. "--overwrite"; see python 3-Diffusion_long.py --help)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_IDS="${GPU_IDS:-0}"
N_MEMBERS="${N_MEMBERS:-8}"
ROLLOUT_LEN="${ROLLOUT_LEN:-}"
OUT_DIR="${OUT_DIR:-/data/weidong/InMOS_1deg/predictions/Diffusion_long}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
CKPT="${CKPT:-/data/weidong/InMOS_1deg/ckpt/ema_ckpt.pt}"
DIFF_CKPT="${DIFF_CKPT:-/data/weidong/InMOS_1deg/ckpt/Diffusion/condiff_1deg_date_best}"
SIGMA="${SIGMA:-0.025}"
SDE_STEPS="${SDE_STEPS:-100}"
SDE_T0_0="${SDE_T0_0:-95}"
SDE_T0_MID="${SDE_T0_MID:-50}"
SDE_T0_1="${SDE_T0_1:-30}"
SDE_PHASE1_STEPS="${SDE_PHASE1_STEPS:-40}"
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
ARGS+=(--ckpt-path "${CKPT}" --diff-ckpt "${DIFF_CKPT}" --perturbation-level "${SIGMA}" --sde-steps "${SDE_STEPS}"
       --sde-t0-0 "${SDE_T0_0}" --sde-t0-mid "${SDE_T0_MID}" --sde-t0-1 "${SDE_T0_1}" --sde-phase1-steps "${SDE_PHASE1_STEPS}")
echo "$(basename "3-Diffusion_long.py"): ${N_MEMBERS} members on GPUs ${GPU_IDS} -> ${OUT_DIR}  (logs ${OUT_DIR}/_logs/${STAMP}_rank*.log)"

pids=()
for ((r = 0; r < NW; r++)); do
    CUDA_VISIBLE_DEVICES="${GPU_ARR[$r]}" PYTHONUNBUFFERED=1 \
        python "${SCRIPT_DIR}/3-Diffusion_long.py" "${ARGS[@]}" --rank "$r" --world-size "${NW}" ${EXTRA_ARGS} \
        > "${OUT_DIR}/_logs/${STAMP}_rank${r}.log" 2>&1 &
    pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "finished (exit ${fail}): $(date '+%F %T')"
exit ${fail}
