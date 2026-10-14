#!/bin/bash
# === InMOS_1degree copy (2026-10-12) -- STAGE 3a (candidate re-scoring): local 8xH100 box, no scheduler. Changed lines are
# marked "1deg:". Run detached:  nohup setsid bash -l <this>.sh > <log> 2>&1 & disown
# Default: GPU 0, M = 4 members in one process. Override: GPU_IDS=0,1,2,3 MEMBERS_PER_RANK=1 (M = NGPU x MEMBERS_PER_RANK).
#
# Stage 2 of the FiLM CRPS work: re-score the checkpoints the gate produced,
# on the FULL held-out period with near-independent ICs, before shipping one.
#
# The gate inside training judged on 6 overlapping ICs from a single year
# (effective n ~ 4, sigma ~ 3.7e-4) while asking for a 4.4e-4 improvement, so
# `ckpt_best` (iter 1599, CRPS 0.08582) sits ~2 sigma below a plateau whose mean
# is 0.08664 — plausibly just the luckiest draw. This run decides that.
#
# Usage:
#   ./3-eval_candidates.sh            # the 3 real candidates
#   SMOKE=1 ./3-eval_candidates.sh    # 2 ICs / 2 steps, wiring check
#   SWEEP=1 ./3-eval_candidates.sh    # + FiLM-strength sweep on ckpt_best
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # 1deg
REPO="/data/weidong/InMOS_1deg"   # 1deg: checkpoint root (CKPT/OUT dirs below)
PY_SCRIPT="${SCRIPT_DIR}/3-eval_candidates.py"

DET="/data/weidong/InMOS_1deg/ckpt/CRPS/1-IC_noise_best_ema.pt"   # 1deg: stage-1 newest PASS (symlink)   # pre-FiLM starting point
FILM_DIR="/data/weidong/InMOS_1deg/ckpt/CRPS"   # 1deg: the one CRPS folder
OUT_DIR="${FILM_DIR}/_logs"   # 1deg: stage-3 outputs are prefixed 3-
mkdir -p "${OUT_DIR}"

VAL_ICS="${VAL_ICS:-40}"       # over 9 years these are ~80 days apart, i.e. the
                               # rollouts no longer overlap
VAL_STEPS="${VAL_STEPS:-16}"   # 1deg: 80 d    # 80-day lead, as in the training gate
VAL_START="${VAL_START:-2013-10-05}"
VAL_END="${VAL_END:-2022-12-29}"

# Candidate order matters only for the delta column: the first is the reference.
#   baseline  = the deterministic CRPS model (FiLM zero-init) — the thing FiLM
#               has to beat to justify existing at all
#   iter1599  = ckpt_best, the gate's pick
#   iter2199  = ckpt_latest, a representative draw from the plateau
# 1deg: the training run keeps EVERY gate-PASS checkpoint as 2-FiLM_noise_it<NNNN>_ema.pt, so
# re-score all of them (the in-training gate cannot rank them reliably).
CAND=(--candidate baseline BASE)
for f in "${FILM_DIR}"/2-FiLM_noise_it*_ema.pt; do
    [[ -e "$f" ]] || continue
    n=$(basename "$f" _ema.pt); CAND+=(--candidate "film_${n#2-FiLM_noise_}" "$f")
done

if [[ "${SWEEP:-0}" == "1" ]]; then
    # Is the LEARNED NOISE buying anything, or is the drifted backbone doing the
    # work? scale=0 is ckpt_best's own deterministic map under the same IC
    # ensemble; if CRPS has no minimum away from 0, FiLM is not earning its keep.
    CAND+=(
      --candidate film_best "${FILM_DIR}/2-FiLM_noise_best_ema.pt" 0.0
      --candidate film_best "${FILM_DIR}/2-FiLM_noise_best_ema.pt" 0.5
      --candidate film_best "${FILM_DIR}/2-FiLM_noise_best_ema.pt" 1.5
      --candidate film_best "${FILM_DIR}/2-FiLM_noise_best_ema.pt" 2.0
    )
fi

if [[ "${SMOKE:-0}" == "1" ]]; then
    VAL_ICS=2; VAL_STEPS=2; VAL_END="2014-10-05"
    echo "SMOKE: ${VAL_ICS} ICs, ${VAL_STEPS} steps, ${VAL_START}..${VAL_END}"
fi

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
LOG="${OUT_DIR}/3-eval_${TS}.log"
JSON="${OUT_DIR}/3-eval_${TS}.json"
echo "log:  ${LOG}"
echo "json: ${JSON}"

PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
${LAUNCH} \
    "${PY_SCRIPT}" \
    --members-per-rank "${MEMBERS_PER_RANK}" \
    --ckpt-path  "${DET}" \
    "${CAND[@]}" \
    --val-start  "${VAL_START}" \
    --val-end    "${VAL_END}" \
    --val-ics    "${VAL_ICS}" \
    --val-steps  "${VAL_STEPS}" \
    --out-json   "${JSON}" \
    2>&1 | tee "${LOG}"
