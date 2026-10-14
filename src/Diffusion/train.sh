#!/bin/bash
# 1-degree conditional diffusion p(state | forcing, date)  (driver: train.py)
#
#   bash train.sh                                   # foreground
#   nohup setsid bash train.sh > /dev/null 2>&1 &   # detached; log in logs/
#   GPUS=0 EPOCHS=2 DEBUG_SAMPLES=16 bash train.sh  # quick debug run
#
# Env: GPUS (comma list), BATCH (per GPU, default 16), EPOCHS (default 2000), DEBUG_SAMPLES, CKPT_DIR,
#      RESUME_FROM (best = default: continue from the best epoch, later epochs re-trained | latest).
# Checkpoints: /data/weidong/InMOS_1deg/ckpt/Diffusion/condiff_1deg_date{,_best} (+ _meta.pt each); lr stays constant.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPUS="${GPUS:-0}"
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
MASTER_PORT="${MASTER_PORT:-$(( 48500 + RANDOM % 1000 ))}"

source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate /data/weidong/wdenv
export CUDA_VISIBLE_DEVICES="$GPUS"
export OMP_NUM_THREADS=8

mkdir -p "$SCRIPT_DIR/logs"
LOG="$SCRIPT_DIR/logs/train_$(date +%Y%m%d_%H%M%S).log"
echo "GPUS=$GPUS NPROC=$NPROC PORT=$MASTER_PORT log=$LOG"
echo "Job started: $(date '+%F %T')" >> "$LOG"
cd "$SCRIPT_DIR"
accelerate launch --num_processes="$NPROC" --num_machines=1 --mixed_precision=bf16 \
    --main_process_port="$MASTER_PORT" \
    train.py >> "$LOG" 2>&1
echo "Job finished: $(date '+%F %T')" >> "$LOG"
