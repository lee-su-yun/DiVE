"""
making_push_visibility_per_view_all.py

각 push 디렉토리에 대해 view + post 모두 평균낸 scalar push visibility를 생성.
making_push_visibility_per_view.py의 scalar 버전 — view 차원 없이 (60, 120, 80) 결과.

정의 (per push_X)
----------------
  v_B[voxel] = (1 / (N_views * N_posts)) Σ_{k, j} (post_ray_casting_j[k, voxel] XOR pre_ray_casting[k, voxel])

  - moved_mask 적용 안 함
  - positive clipping 없음 (XOR이라 양방향 flip 모두 카운트)
  - 98개 view + N_posts(fallen/tipped 제외) 평균 → scalar per voxel
  - 값 범위 ∈ [0, 1] = "push로 인해 임의 view에서 voxel visibility가 flip될 평균 확률"

  fallen_log_revised.txt + tipped_log_revised.txt 등록된 (scene, push_id, post_id)는 평균에서 제외.
  29개 post 중 fallen=2면 27개만 더하고 27로 나눔.

저장 형식
---------
  push_X/push_visibility_all.npy  (60, 120, 80) float32

실행
----
  cd /home/sylee/codes/DiVE/1_preprocessing
  sudo -E /home/sylee/miniconda3/envs/APOBU/bin/python making_push_visibility_per_view_all.py \
      --data-roots /data/APOBU/beliefmap_low_occlusion_0423 \
                   /data/APOBU/beliefmap_high_occlusion_0423 \
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "support_code"))
from dataset_UAB import parse_fallen_log  # noqa: E402  (regex shared with tipped log)


DEFAULT_DATA_ROOTS = [
    "/data/APOBU/beliefmap_low_occlusion_0423",
    "/data/APOBU/beliefmap_high_occlusion_0423",
]
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

    out_path = os.path.join(push_dir, 'push_visibility_all.npy')
    if (not overwrite) and os.path.exists(out_path):
        return push_dir

    pre_rc_path = os.path.join(push_dir, 'pre_ray_casting.npy')
    if not os.path.exists(pre_rc_path):
        return push_dir

    pre_packed = np.load(pre_rc_path)                       # (98, 60, 120, 10) uint8
    pre_unpacked = np.unpackbits(pre_packed, axis=-1)       # (98, 60, 120, 80) uint8 binary

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

        # XOR: 양방향 flip 모두 1. moved_mask, clipping 없음.
        diff = post_unpacked ^ pre_unpacked                  # (98, 60, 120, 80) uint8 0/1
        # view 축 평균 → (60, 120, 80) float64 (uint8 sum이 98*1=98로 안전)
        acc += diff.sum(axis=0, dtype=np.float64) / NUM_CAMERAS
        n_posts += 1

    if n_posts == 0:
        print(f"[skip] {push_dir}: 모든 post가 fallen/tipped 또는 누락", flush=True)
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
        print(f'[push_v_B_all] {root}: {len(push_dirs)} push, '
              f'fallen={len(fallen)} tipped={len(tipped)} excluded={len(excluded)}')
        for pd in push_dirs:
            tasks.append((pd, excluded, overwrite))

    print(f'[push_v_B_all] 합계 {len(tasks)}개 push 처리 '
          f'(workers={workers}, overwrite={overwrite})')
    if not tasks:
        return

    with Pool(workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(_process_push_safe, tasks),
            total=len(tasks),
            desc='push_v_B_all',
        ):
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-roots', type=str, nargs='+', default=DEFAULT_DATA_ROOTS)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    generate_push_visibility_all(args.data_roots, workers=args.workers,
                                 overwrite=args.overwrite)
