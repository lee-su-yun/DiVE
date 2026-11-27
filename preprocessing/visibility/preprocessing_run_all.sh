#!/usr/bin/env bash
# End-to-end preprocessing for DiVE datasets.
#
# Usage:
#   bash preprocessing/visibility/preprocessing_run_all.sh <data_root_1> [<data_root_2> ...]
#
# Example:
#   bash preprocessing/visibility/preprocessing_run_all.sh \
#       /path/to/data/beliefmap_low_occlusion_ycb_v3 \
#       /path/to/data/beliefmap_high_occlusion_ycb_v3
#
# Steps:
#   0) fallen_log.txt → fallen_log_revised.txt  +  tipped_log.txt → tipped_log_revised.txt
#   1) extract_occ_views (per-view OCC split)  +  pose.npy generation
#   2) per-view binary ray casting (UA training)
#   3) push_visibility_all_marginal (the phi_push training target)

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <data_root_1> [<data_root_2> ...]" >&2
    exit 1
fi
DATA_ROOTS=("$@")

PYTHON="${PYTHON:-python}"
PREP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKERS=32
POSTS_PER_PUSH=20

# ── helpers ──────────────────────────────────────────────────────────────────
stage()    { echo; echo "============================================================"; echo "$*"; echo "============================================================"; }

# Plain python (no sudo). If the data root is only writable as root, run the whole
# script with elevated privileges rather than any individual step:
#   bash preprocessing/visibility/preprocessing_run_all.sh <data_root>
PYRUN=( "$PYTHON" )

stage "DATA_ROOTS"
for r in "${DATA_ROOTS[@]}"; do
    [ -d "$r" ] || { echo "ERROR: not a directory: $r" >&2; exit 1; }
    echo "  $r"
done

# ── Step 0: log conversion ───────────────────────────────────────────────────
stage "Step 0 — fallen/tipped log conversion"
"${PYRUN[@]}" - "${DATA_ROOTS[@]}" <<EOF
import os, re, shutil, sys
roots = sys.argv[1:]
posts_per_push = ${POSTS_PER_PUSH}
pat = re.compile(r"^(episode=\d+)\s+post_(\d+)\s+(fallen=.*)$")
for ROOT in roots:
    fallen_in  = os.path.join(ROOT, "fallen_log.txt")
    fallen_out = os.path.join(ROOT, "fallen_log_revised.txt")
    tipped_in  = os.path.join(ROOT, "tipped_log.txt")
    tipped_out = os.path.join(ROOT, "tipped_log_revised.txt")

    if os.path.exists(fallen_out):
        print(f"[fallen] {ROOT}: already exists, skip")
    elif not os.path.exists(fallen_in):
        print(f"[fallen] {ROOT}: missing fallen_log.txt, skip")
    else:
        out = []
        with open(fallen_in) as f:
            for ln, line in enumerate(f, 1):
                raw = line.rstrip("\n")
                if not raw.strip(): continue
                m = pat.match(raw)
                if not m: raise SystemExit(f"{ROOT} bad line {ln}: {raw!r}")
                ep, p, fallen = m.group(1), int(m.group(2)), m.group(3)
                push_idx, new_post = p // posts_per_push + 1, p % posts_per_push
                out.append(f"{ep}  push_{push_idx}  post_{new_post:02d}  {fallen}")
        with open(fallen_out, "w") as f:
            f.write("\n".join(out) + "\n")
        print(f"[fallen] {ROOT}: wrote {len(out)} lines")

    if os.path.exists(tipped_out):
        print(f"[tipped] {ROOT}: already exists, skip")
    elif not os.path.exists(tipped_in):
        print(f"[tipped] {ROOT}: missing tipped_log.txt, skip")
    else:
        shutil.copy(tipped_in, tipped_out)
        print(f"[tipped] {ROOT}: copied")
EOF

# ── Step 1: extract_occ_views + pose.npy ─────────────────────────────────────
stage "Step 1 — extract_occ_views + pose.npy"
"${PYRUN[@]}" - "${DATA_ROOTS[@]}" <<EOF
import os, sys
from glob import glob
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "${PREP_DIR}")
from data_preprocessing import extract_occ_views, load_pose
roots = sys.argv[1:]

print(">>> extract_occ_views")
for r in roots:
    extract_occ_views(r, workers=${WORKERS})

def _save_pose(push_dir):
    cam = os.path.join(push_dir, "camera_poses.npz")
    out = os.path.join(push_dir, "pose.npy")
    if not os.path.exists(cam) or os.path.exists(out):
        return push_dir
    np.save(out, load_pose(cam))
    return push_dir

print(">>> pose.npy")
for r in roots:
    push_dirs = sorted(glob(os.path.join(r, "*/push_*")))
    print(f"[pose] {r}: {len(push_dirs)} push")
    with Pool(${WORKERS}) as pool:
        for _ in pool.imap_unordered(_save_pose, push_dirs):
            pass
EOF

# ── Step 2: per-view ray casting ─────────────────────────────────────────────
stage "Step 2 — making_ray_casting_per_view.py"
"${PYRUN[@]}" "${PREP_DIR}/making_ray_casting_per_view.py" \
    --data-roots "${DATA_ROOTS[@]}" \
    --workers ${WORKERS}

# ── Step 3: push_visibility_all_marginal ─────────────────────────────────────
# Produces push_X/push_visibility_all_marginal.npy, the push-resolvable target
# (phi_push) that dive/train_UAB_unified.py reads through common/dataset_UAB_unified.py.
stage "Step 3 — making_push_visibility_per_view_all_marginal.py"
"${PYRUN[@]}" "${PREP_DIR}/making_push_visibility_per_view_all_marginal.py" \
    --data-roots "${DATA_ROOTS[@]}" \
    --workers ${WORKERS}

# ── Summary ──────────────────────────────────────────────────────────────────
stage "Summary"
for r in "${DATA_ROOTS[@]}"; do
    echo "--- $r ---"
    echo "  fallen_log_revised:    $(test -f "$r/fallen_log_revised.txt" && echo OK || echo MISSING)"
    echo "  tipped_log_revised:    $(test -f "$r/tipped_log_revised.txt" && echo OK || echo MISSING)"
    echo "  pre_occ dirs:          $(find "$r" -maxdepth 3 -type d -name pre_occ 2>/dev/null | wc -l)"
    echo "  pose.npy:              $(find "$r" -maxdepth 3 -name pose.npy 2>/dev/null | wc -l)"
    echo "  pre_ray_casting.npy:   $(find "$r" -maxdepth 3 -name pre_ray_casting.npy 2>/dev/null | wc -l)"
    echo "  push_vis_marginal:     $(find "$r" -maxdepth 3 -name push_visibility_all_marginal.npy 2>/dev/null | wc -l)"
done
echo
echo "Done."
