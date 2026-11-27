#!/usr/bin/env bash
# Run unified UA+UB training (shared backbone, 2-channel head) with the
# teacher-forcing annealing schedule used for the released checkpoint.
#
# Usage (from the repo root):
#   bash dive/train_UAB_unified.sh <tag> <data_root_1> [<data_root_2> ...]
#
# Example:
#   UAB_DEVICE=4,5 bash dive/train_UAB_unified.sh ycb_v3 \
#       /path/to/data/beliefmap_high_occlusion_ycb_v3 \
#       /path/to/data/beliefmap_low_occlusion_ycb_v3
#
# Env var overrides (optional):
#   UAB_DEVICE    GPU ids,        default "0,1"
#   UAB_DDP_PORT  DDP port,       default 29099
#   UAB_EPOCHS    epochs,         default 50
#   UAB_PATIENCE  patience,       default 5
#   UAB_LAMBDA_B  UB loss weight, default 1.0
#   SAVE_DIR      output dir,     default runs/UAB_unified_<tag>_TFanneal
#   PYTHON        interpreter,    default python
#
# Note: the script cds into dive/ before launching the trainer, so relative
# paths such as the default SAVE_DIR resolve inside dive/.

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <tag> <data_root_1> [<data_root_2> ...]" >&2
    exit 1
fi

TAG="$1"; shift
DATA_ROOTS=("$@")

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SAVE_DIR="${SAVE_DIR:-runs/UAB_unified_${TAG}_TFanneal}"
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
"${PYTHON}" train_UAB_unified.py \
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
