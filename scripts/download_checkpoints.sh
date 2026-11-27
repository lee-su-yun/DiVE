#!/usr/bin/env bash
# Download the released DiVE checkpoints from the Hugging Face Hub into checkpoints/.
#
#   bash scripts/download_checkpoints.sh [target_dir]
#
# Needs either the `hf` / `huggingface-cli` tool (pip install huggingface_hub)
# or curl as a fallback.
set -euo pipefail

REPO="leesuyun/DiVE"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/checkpoints}"

FILES=(
  "DiVE_decomposed_visibility_estimator/best.pth"
  "view_selection_policy/best.pth"
  "push_selection_policy/best.pth"
  "belief_map_estimator/estimator.pt"
)

mkdir -p "$OUT"

if command -v hf >/dev/null 2>&1; then
  hf download "$REPO" --local-dir "$OUT"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$REPO" --local-dir "$OUT"
else
  echo "huggingface_hub not found, falling back to curl"
  for f in "${FILES[@]}"; do
    mkdir -p "$OUT/$(dirname "$f")"
    echo "  $f"
    curl -fL "https://huggingface.co/$REPO/resolve/main/$f" -o "$OUT/$f"
  done
fi

echo
echo "Checkpoints in $OUT:"
for f in "${FILES[@]}"; do
  [ -f "$OUT/$f" ] && echo "  $f" || echo "  MISSING: $f"
done
echo
echo "To run the evaluation, point inference/run_eval.sh at them, e.g.:"
echo "  --uab_belief   $OUT/DiVE_decomposed_visibility_estimator/best.pth"
echo "  --ua_reward    $OUT/view_selection_policy/best.pth"
echo "  --ub_reward    $OUT/push_selection_policy/best.pth"
echo "  --central_ckpt $OUT/belief_map_estimator/estimator.pt"
