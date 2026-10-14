#!/bin/bash
# === InMOS_1degree copy (2026-10-12) -- STAGE 3b (FiLM strength sweep): local 8xH100 box, no scheduler. Changed lines are
# marked "1deg:". Run detached:  nohup setsid bash -l <this>.sh > <log> 2>&1 & disown
# Default: GPU 0, M = 4 members in one process. Override: GPU_IDS=0,1,2,3 MEMBERS_PER_RANK=1 (M = NGPU x MEMBERS_PER_RANK).
#
# Sweep the FiLM strength of ONE checkpoint and report which scale is best for
# CRPS, for RMSE, and for calibration — they are three different answers.
#
# Scaling is exact and free at inference: `mod` is linear in the FiLM heads'
# zero-init output layers, so multiplying those four tensors per head multiplies
# gamma/beta exactly (see scale_film() in eval_film_candidates.py). s=0 is the
# checkpoint's own deterministic map, s=1 is as trained.
#
# 1deg: defaults to ckpt_best_ema.pt = newest gate-PASS; choose CKPT from the 3-eval_candidates ranking.
# 0.25deg note: Defaults swept ckpt_latest_ema.pt (iter 2199), NOT ckpt_best_ema.pt: stage 2
# showed the in-training gate ranked those two backwards, and picking a strength
# for the wrong checkpoint stacks two errors. See /home/nvidia/weidong/CRPS/README_FILM.md "Stage 2".
#
# Usage:
#   ./3-scale_sweep.sh
#   CKPT=<path> ./3-scale_sweep.sh
#   SCALES="0.75 1 1.25 1.5" ./3-scale_sweep.sh
#   VAL_ICS=20 ./3-scale_sweep.sh          # ~half the wall clock
#   PERT=0.05 ./3-scale_sweep.sh           # re-sweep at another IC sigma
#   SMOKE=1 ./3-scale_sweep.sh             # wiring check, ~3 min
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # 1deg
REPO="/data/weidong/InMOS_1deg"   # 1deg: checkpoint root (CKPT/OUT dirs below)
EVAL_PY="${SCRIPT_DIR}/3-eval_candidates.py"
ANALYZE_PY="${SCRIPT_DIR}/3-analyze_scale_sweep.py"

DET="/data/weidong/InMOS_1deg/ckpt/CRPS/1-IC_noise_best_ema.pt"   # 1deg: stage-1 newest PASS (symlink)   # pre-FiLM baseline
FILM_DIR="/data/weidong/InMOS_1deg/ckpt/CRPS"   # 1deg: the one CRPS folder
CKPT="${CKPT:-${FILM_DIR}/2-FiLM_noise_best_ema.pt}"   # 1deg: newest gate-PASS (symlink); pick another 2-FiLM_noise_it<N>_ema.pt after 3-eval_candidates
OUT_DIR="${FILM_DIR}/_logs"   # 1deg: stage-3 outputs are prefixed 3-
mkdir -p "${OUT_DIR}"

# 0 is included on purpose: it measures how much of this checkpoint's skill is
# the drifted backbone rather than the learned noise, and stage 2 found the
# backbone had turned IC spread from growing to CONTRACTING (SSR 0.095 at 80 d).
SCALES="${SCALES:-0 0.5 0.75 1.0 1.25 1.5 1.75 2.0}"
VAL_ICS="${VAL_ICS:-40}"
VAL_STEPS="${VAL_STEPS:-16}"   # 1deg: 80 d
VAL_START="${VAL_START:-2013-10-05}"
VAL_END="${VAL_END:-2022-12-29}"
PERT="${PERT:-0.025}"          # IC sigma; matches the long_perturb ensembles
WITH_BASELINE="${WITH_BASELINE:-1}"
LABEL=sweep

if [[ "${SMOKE:-0}" == "1" ]]; then
    SCALES="0 1.0"; VAL_ICS=2; VAL_STEPS=2; VAL_END="2014-10-05"; WITH_BASELINE=0
    echo "SMOKE: scales '${SCALES}', ${VAL_ICS} ICs, ${VAL_STEPS} steps"
fi

# The reference row must come first — the eval script's delta columns are
# relative to the first candidate.
CAND=()
[[ "${WITH_BASELINE}" == "1" ]] && CAND+=(--candidate baseline BASE)
for s in ${SCALES}; do
    CAND+=(--candidate "${LABEL}" "${CKPT}" "${s}")
done

N=$(( $(wc -w <<< "${SCALES}") + (WITH_BASELINE == 1 ? 1 : 0) ))
echo "ckpt:    ${CKPT}"
echo "scales:  ${SCALES}"
echo "val:     ${VAL_START}..${VAL_END}, ${VAL_ICS} ICs, ${VAL_STEPS} steps, IC sigma=${PERT}"
echo "budget:  ${N} candidates x ~$(( VAL_ICS * VAL_STEPS * 60 / 320 )) min (measured: 40 ICs x 8 steps = 8 min)"

source /home/nvidia/miniforge3/etc/profile.d/conda.sh   # 1deg
conda activate /data/weidong/ocean_env                      # 1deg
MEMBERS_PER_RANK="${MEMBERS_PER_RANK:-4}"   # 1deg: M = NGPU x this; must equal the training M
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0}"   # 1deg: default GPU 0 (several: GPU_IDS=0,1,2,3 MEMBERS_PER_RANK=1)
NGPU=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")  # 1deg
# 1deg: torchrun --standalone hangs in rendezvous on this box (the hostname resolves to an
# external address); 1 GPU needs no torchrun, >1 GPU pins the rendezvous to localhost.
if [[ "${NGPU}" == "1" ]]; then LAUNCH="python"
else LAUNCH="torchrun --nnodes=1 --nproc_per_node=${NGPU} --rdzv_backend=c10d --rdzv_endpoint=localhost:0"; fi

TS=$(date +%Y%m%d_%H%M%S)
LOG="${OUT_DIR}/3-sweep_${TS}.log"
JSON="${OUT_DIR}/3-sweep_${TS}.json"
echo "log:     ${LOG}"

PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
${LAUNCH} \
    "${EVAL_PY}" \
    --members-per-rank "${MEMBERS_PER_RANK}" \
    --ckpt-path          "${DET}" \
    "${CAND[@]}" \
    --val-start          "${VAL_START}" \
    --val-end            "${VAL_END}" \
    --val-ics            "${VAL_ICS}" \
    --val-steps          "${VAL_STEPS}" \
    --perturbation-level "${PERT}" \
    --out-json           "${JSON}" \
    2>&1 | tee "${LOG}"

# The analyzer reads the JSON for aggregates and this log for the per-lead rows.
echo
python "${ANALYZE_PY}" --json "${JSON}" --log "${LOG}" 2>&1 | tee -a "${LOG}"
