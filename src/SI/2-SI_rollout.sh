#!/bin/bash
# 1deg SI forecaster, STAGE 2: rollout-loss fine-tune, K=8 calls (40 days), warm start from stage 1  (driver: 2-SI_rollout.py).
#
#   bash 2-SI_rollout.sh                                   # foreground
#   nohup setsid bash 2-SI_rollout.sh > /dev/null 2>&1 &   # detached; log in logs/
#   GPUS=3 LIMIT_WINDOWS=16 EPOCHS=2 USE_WANDB=0 CKPT_TAG=smoke bash 2-SI_rollout.sh   # smoke test
#
# Env overrides (defaults in the .py):
#   K=8 EPOCHS=30 BATCH_SIZE=4 GRAD_ACCUM=1 LR=1e-5 PCT_START=0.3 DIV_FACTOR=25 MIN_LR=1e-7  (OneCycle, per optimizer step; LR = peak) EMA_DECAY=0.999
#   INIT_CKPT=/data/weidong/InMOS_1deg/ckpt/SI/SI1deg_step1_best_ema.pt   (raw: .../SI1deg_step1_best)
#   FULL_GRAD=1 STEP_WEIGHTS=linear|uniform|last|w1,..,wK ROLLOUT_DDIM_STEPS=2 INC_CLAMP=8
#   ROLLOUT_EVERY=1 ROLLOUT_CALLS=146 GRAD_CKPT=1 INPUT_NOISE_REL=0.1
#   CKPT_TAG=SI1deg_rollout_K8 CKPT_DIR=/data/weidong/InMOS_1deg/ckpt/SI LIMIT_WINDOWS= USE_WANDB=1
# Re-running with the same CKPT_TAG resumes from CKPT_DIR/CKPT_TAG_best (RESUME_FROM=best, default;
# later epochs are rolled back and re-trained) or from the latest epoch (RESUME_FROM=latest).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPUS="${GPUS:-1,2,3,4,5,6,7}"          # NOT GPU 0 while CRPS runs there: peak 66.5 GiB at batch 4 (59 at batch 2) + CRPS 24.6
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
MASTER_PORT="${MASTER_PORT:-$(( 46500 + RANDOM % 2000 ))}"

source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate /data/weidong/wdenv
export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

mkdir -p "$SCRIPT_DIR/logs"

# wandb: the shared ~/.netrc is logged in as another user (binggong), who cannot
# write to the beauwdli entity ("permission denied"). Load beauwdli's key per
# process from the 0600 key file (same as Samudra-SI), no global relogin.
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$SCRIPT_DIR/logs}"      # run files in logs/wandb/
export WANDB__SERVICE_WAIT=300
WANDB_KEY_FILE="${WANDB_KEY_FILE:-/data/weidong/.wandb_key}"
if [ -f "$WANDB_KEY_FILE" ]; then
    export WANDB_API_KEY="$(cat "$WANDB_KEY_FILE")"
fi
LOG="$SCRIPT_DIR/logs/rollout_${CKPT_TAG:-default}_$(date +%Y%m%d_%H%M%S).log"
echo "GPUS=$GPUS NPROC=$NPROC PORT=$MASTER_PORT log=$LOG"
echo "Job started: $(date '+%F %T')" >> "$LOG"
torchrun --nnodes=1 --nproc_per_node="$NPROC" --master_addr=127.0.0.1 \
    --master_port="$MASTER_PORT" --max_restarts=0 \
    "$SCRIPT_DIR/2-SI_rollout.py" >> "$LOG" 2>&1
echo "Job finished: $(date '+%F %T')" >> "$LOG"
