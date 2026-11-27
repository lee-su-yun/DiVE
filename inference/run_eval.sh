#!/bin/bash
# DiVE + belief map estimator evaluation on a held-out test set.
# Reproduces the reference run in inference/reference_results/REPORT.md:
# 100 episodes x 25-step budget, 4 shards of 25 episodes in parallel, then aggregate.
#
#   TEST_SET=/path/to/low_occlusion_test_v1 bash inference/run_eval.sh [out_dir]
#
# Environment variables:
#   PY          python of the Isaac Lab environment            (default: python)
#   CKPT_DIR    checkpoints from scripts/download_checkpoints.sh (default: ./checkpoints)
#   TEST_SET    test-set directory (required)
#   SCENE_MODE  low | high | super_high, must match TEST_SET  (default: low)
set -e
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-python}
CKPT_DIR=${CKPT_DIR:-$ROOT/checkpoints}
SCENE_MODE=${SCENE_MODE:-low}
OUT=${1:-$ROOT/experiments/${SCENE_MODE}_v1_recursive_step25}
TS_DIR=${TEST_SET:-}

if [ -z "$TS_DIR" ]; then
  echo "TEST_SET is not set. Point it at the test-set directory, e.g."
  echo "  TEST_SET=/path/to/low_occlusion_test_v1 bash inference/run_eval.sh"
  exit 1
fi

cd "$ROOT"
export PYTHONPATH=$ROOT:$PYTHONPATH
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$OUT"
echo "[eval] === start $(date) ===" >> "$OUT/_sweep.log"

SHARDS=(0:25 25:50 50:75 75:100)
PIDS=(); SHARD=0
for RANGE in "${SHARDS[@]}"; do
  START="${RANGE%:*}"; END="${RANGE#*:}"; NUM=$((END - START))
  (
    nice -n 10 "$PY" -u -m inference.run_eval \
      --uab_belief   "$CKPT_DIR/DiVE_decomposed_visibility_estimator/best.pth" \
      --ua_reward    "$CKPT_DIR/view_selection_policy/best.pth" \
      --ub_reward    "$CKPT_DIR/push_selection_policy/best.pth" \
      --central_ckpt "$CKPT_DIR/belief_map_estimator/estimator.pt" \
      --n_sem 12 --n_sem_model 15 \
      --scene_mode "$SCENE_MODE" --action_budget 25 \
      --test_set_dir "$TS_DIR" \
      --episode_start "$START" --episode_end "$END" --num_episodes "$NUM" \
      --summary_suffix "_shard${SHARD}" \
      --out_dir "$OUT" --headless --seed 0 \
      > "$OUT/run_shard${SHARD}.log" 2>&1
  ) &
  PIDS+=($!)
  echo "[eval] shard$SHARD PID=$! eps=[$START,$END)" >> "$OUT/_sweep.log"
  SHARD=$((SHARD + 1))
  sleep 10
done
for p in "${PIDS[@]}"; do wait "$p" || echo "[eval] pid=$p failed" >> "$OUT/_sweep.log"; done
echo "[eval] === done $(date) ===" >> "$OUT/_sweep.log"
"$PY" -m inference.aggregate --run_dir "$OUT" --title "$SCENE_MODE recursive step25" >> "$OUT/_sweep.log" 2>&1 || true
echo "[eval] finished -> $OUT/REPORT.md"
