#!/usr/bin/env bash
# Run unified UA+UB training (Option A: shared backbone, 2-channel head).
#
# Usage:
#   bash train_UAB_unified.sh <tag> <data_root_1> [<data_root_2> ...]
#
# Example:
#   UAB_DEVICE=4,5 bash /home/sylee/codes/DiVE/2_belief_unified/train_UAB_unified.sh ycb_v3 \
#       /data/APOBU/beliefmap_high_occlusion_ycb_v3 \
#       /result/DiVE_data/beliefmap_low_occlusion_ycb_v3
#
# Env var overrides (optional):
#   UAB_DEVICE    GPU ids,    default "0,1"
#   UAB_DDP_PORT  DDP port,   default 29299
#   UAB_EPOCHS    epochs,     default 50
#   UAB_PATIENCE  patience,   default 5
#   UAB_LAMBDA_B  UB loss weight, default 1.0

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <tag> <data_root_1> [<data_root_2> ...]" >&2
    exit 1
fi

TAG="$1"; shift
DATA_ROOTS=("$@")

PYTHON="/home/sylee/miniconda3/envs/APOBU/bin/python"
SCRIPT_DIR="/home/sylee/codes/DiVE/2_belief_unified"

SAVE_DIR="/result/APOBU/DiVE/UAB_unified_${TAG}_TFanneal"
WANDB_RUN="UAB_unified_${TAG}_TFanneal_1to0"

DEVICE="${UAB_DEVICE:-0,1}"
DDP_PORT="${UAB_DDP_PORT:-29099}"
EPOCHS="${UAB_EPOCHS:-50}"
PATIENCE="${UAB_PATIENCE:-5}"
LAMBDA_B="${UAB_LAMBDA_B:-1.0}"

echo "=========================================================="
echo "Train UAB_unified  tag=${TAG}"
echo "  data_roots: ${DATA_ROOTS[*]}"
echo "  save_dir:   ${SAVE_DIR}"
echo "  wandb_run:  ${WANDB_RUN}"
echo "  device:     ${DEVICE}    ddp_port: ${DDP_PORT}"
echo "  epochs:     ${EPOCHS}    patience: ${PATIENCE}"
echo "  lambda_b:   ${LAMBDA_B}"
echo "=========================================================="

cd "${SCRIPT_DIR}"
sudo -E "${PYTHON}" train_UAB_unified.py \
    --data_roots "${DATA_ROOTS[@]}" \
    --save_dir "${SAVE_DIR}" \
    --wandb_run_name "${WANDB_RUN}" \
    --device "${DEVICE}" \
    --teacher_forcing \
    --tf_anneal_to 0.0 \
    --lambda_b "${LAMBDA_B}" \
    --epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --ddp_port "${DDP_PORT}"
