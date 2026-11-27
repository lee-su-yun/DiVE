"""
making_push_visibility_per_view_all_marginal.py

Build the scalar push visibility for every push directory from the change in the
view-marginal visibility, averaged over the post axis: the absolute difference of the
view-marginals before and after each candidate push.

Definition (per push_X)
-----------------------
  pre_marginal[voxel]    = (1 / N_views) sum_k pre_ray_casting[k, voxel]
  post_marginal_j[voxel] = (1 / N_views) sum_k post_ray_casting_j[k, voxel]
  v_B[voxel] = (1 / N_posts) sum_j | post_marginal_j[voxel] - pre_marginal[voxel] |

  - no moved_mask
  - no positive clipping (the absolute difference counts changes in both directions)
  - marginalize over the 98 views first, take the absolute post-minus-pre difference,
    then average over the N_posts that survive the fallen/tipped filter
  - range [0, 1] = "the average magnitude of the shift the push causes in this
    voxel's marginal visibility"

  Any (scene, push_id, post_id) listed in fallen_log_revised.txt or
  tipped_log_revised.txt is dropped from the average: with 20 posts per push and 2
  of them flagged, only 18 are summed and the sum is divided by 18.

Output
------
  push_X/push_visibility_all_marginal.npy  (60, 120, 80) float32

Usage
-----
  cd <repo root>
  python preprocessing/visibility/making_push_visibility_per_view_all_marginal.py \
      --data-roots /path/to/data/beliefmap_low_occlusion_ycb_v3 \
                   /path/to/data/beliefmap_high_occlusion_ycb_v3 \
      --workers 32
"""

import argparse
import os
import sys
from glob import glob
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from data_utils import parse_fallen_log  # noqa: E402  (regex shared with tipped log)


NUM_CAMERAS = 98
VZ, VX, VY = 60, 120, 80


def _process_push(task):
    push_dir, excluded, overwrite = task
    scene_dir = os.path.dirname(push_dir)
    scene_name = os.path.basename(scene_dir)
    push_name = os.path.basename(push_dir)
    try:
        push_id = int(push_name.split('_')[1])
    except (ValueError, IndexError):
        return push_dir

    out_path = os.path.join(push_dir, 'push_visibility_all_marginal.npy')
    if (not overwrite) and os.path.exists(out_path):
        return push_dir

    pre_rc_path = os.path.join(push_dir, 'pre_ray_casting.npy')
    if not os.path.exists(pre_rc_path):
        return push_dir

    pre_packed = np.load(pre_rc_path)                       # (98, 60, 120, 10) uint8
    pre_unpacked = np.unpackbits(pre_packed, axis=-1)       # (98, 60, 120, 80) uint8 binary
    # pre is the same for every post, so compute it once outside the loop
    pre_marginal = pre_unpacked.mean(axis=0, dtype=np.float64)  # (60, 120, 80) float64

    acc = np.zeros((VZ, VX, VY), dtype=np.float64)
    n_posts = 0

    post_dirs = sorted(d for d in glob(os.path.join(push_dir, 'post_*'))
                       if os.path.isdir(d))
    for post_dir in post_dirs:
        post_name = os.path.basename(post_dir)
        try:
            post_id = int(post_name.split('_')[1])
        except (ValueError, IndexError):
            continue
        if (scene_name, push_id, post_id) in excluded:
            continue

        post_rc_path = os.path.join(post_dir, 'post_ray_casting.npy')
        if not os.path.exists(post_rc_path):
            continue

        post_packed = np.load(post_rc_path)
        post_unpacked = np.unpackbits(post_packed, axis=-1)

        # Absolute marginal-visibility difference: the view axis is averaged first,
        # and changes in either direction count.
        post_marginal = post_unpacked.mean(axis=0, dtype=np.float64)  # (60, 120, 80)
        acc += np.abs(post_marginal - pre_marginal)
        n_posts += 1

    if n_posts == 0:
        print(f"[skip] {push_dir}: every post is fallen/tipped or missing", flush=True)
        return push_dir

    acc /= n_posts
    result = acc.astype(np.float32)

    tmp_path = out_path + '.tmp'
    with open(tmp_path, 'wb') as f:
        np.save(f, result)
    os.replace(tmp_path, out_path)
    return push_dir


def _process_push_safe(task):
    try:
        return _process_push(task)
    except Exception as e:
        print(f"[error] {task[0]}: {e}", flush=True)
        return task[0]


def generate_push_visibility_all(data_roots, workers=16, overwrite=False):
    tasks = []
    for root in data_roots:
        fallen = parse_fallen_log(os.path.join(root, 'fallen_log_revised.txt'))
        tipped = parse_fallen_log(os.path.join(root, 'tipped_log_revised.txt'))
        excluded = fallen | tipped
        push_dirs = sorted(glob(os.path.join(root, '*/push_*')))
        print(f'[push_v_B_all_marginal] {root}: {len(push_dirs)} push, '
              f'fallen={len(fallen)} tipped={len(tipped)} excluded={len(excluded)}')
        for pd in push_dirs:
            tasks.append((pd, excluded, overwrite))

    print(f'[push_v_B_all_marginal] {len(tasks)} pushes in total '
          f'(workers={workers}, overwrite={overwrite})')
    if not tasks:
        return

    with Pool(workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(_process_push_safe, tasks),
            total=len(tasks),
            desc='push_v_B_all_marginal',
        ):
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-roots', type=str, nargs='+', required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    generate_push_visibility_all(args.data_roots, workers=args.workers,
                                 overwrite=args.overwrite)
