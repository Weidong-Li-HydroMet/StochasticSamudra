#!/bin/bash
# InMOS_1degree (2026-10-12): stage 1 (IC noise) then stage 2 (FiLM noise), one GPU, one folder.
# Stage 2 starts from stage 1's newest gate-PASS (1-IC_noise_best_ema.pt) and only if it exists.
#   bash run_1_then_2.sh            -> detaches itself (survives logout); log in <OUT_DIR>/_logs/chain_<ts>.out
#   Defaults: GPU 0, M = 4 members, 4 ICs/iter, stage 1 3000 iters, stage 2 6000 iters (all env-overridable).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -z "${_CHAIN_DETACHED:-}" ]]; then
    mkdir -p "${OUT_DIR:-/data/weidong/InMOS_1deg/ckpt/CRPS}/_logs"
    LOGF="${OUT_DIR:-/data/weidong/InMOS_1deg/ckpt/CRPS}/_logs/chain_$(date +%Y%m%d_%H%M%S).out"
    _CHAIN_DETACHED=1 nohup setsid bash -l "$(pwd)/run_1_then_2.sh" > "${LOGF}" 2>&1 < /dev/null &
    echo "started (pid $!), log: ${LOGF}"; exit 0
fi
export GPU_IDS="${GPU_IDS:-0}" MEMBERS_PER_RANK="${MEMBERS_PER_RANK:-4}" BN_MODE="${BN_MODE:-sample}"
export OUT_DIR="${OUT_DIR:-/data/weidong/InMOS_1deg/ckpt/CRPS}"
echo "[$(date '+%F %T')] stage 1 start"; bash 1-IC_noise.sh; rc=$?
echo "[$(date '+%F %T')] stage 1 exit $rc"
if [[ ! -e "${OUT_DIR}/1-IC_noise_best_ema.pt" ]]; then
    echo "no stage-1 gate-PASS checkpoint -> stage 2 NOT started"; exit 1
fi
export CKPT="$(readlink -f "${OUT_DIR}/1-IC_noise_best_ema.pt")"
echo "[$(date '+%F %T')] stage 2 start from ${CKPT}"; bash 2-FiLM_noise.sh; rc=$?
echo "[$(date '+%F %T')] stage 2 exit $rc"
