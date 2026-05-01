"""
making_ray_casting_per_view.py

각 push 디렉토리에 대해 98개 카메라 view 각각의 binary visibility (ray casting)을
별도로 저장한다. 학습 시 K_t에 해당하는 view만 mmap으로 슬라이싱해서 GT를 즉석 구성하기 위함.

출력 형식
---------
  push_X/pre_ray_casting.npy           shape (98, 60, 120, 10) uint8
  push_X/post_NN/post_ray_casting.npy  shape (98, 60, 120, 10) uint8

  마지막 축 80 → np.packbits로 10바이트로 압축. 파일당 정확히 6.73 MB.
  복원: np.unpackbits(packed[k], axis=-1) → (60, 120, 80) uint8 binary

학습 시 사용 예시
----------------
    packed = np.load(path, mmap_mode='r')             # (98, 60, 120, 10)
    view_k = np.unpackbits(packed[k], axis=-1)        # (60, 120, 80) uint8

기존 compute_visibility와 동일한 ray casting 파이프라인 (depth buffer + 표면 tolerance 0.5)을
사용하므로, 결과를 view 축으로 평균내면 기존 pre_visibility.npy와 일치한다.

실행
----
    cd /home/sylee/codes/DiVE/1_preprocessing
    sudo -E /home/sylee/miniconda3/envs/APOBU/bin/python making_ray_casting_per_view.py \
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
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# support_code/gen_visibility.py의 카메라 intrinsics, voxel grid 상수 재사용
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "support_code"))
from gen_visibility import (  # noqa: E402
    ALL_CENTERS, fx, fy, cx, cy, IMG_W, IMG_H, VZ, VX, VY, N_VOXELS,
)


DEFAULT_DATA_ROOTS = [
    "/data/APOBU/beliefmap_low_occlusion_0423",
    "/data/APOBU/beliefmap_high_occlusion_0423",
]
VOXEL_SIZE = 0.005
NUM_CAMERAS = 98


def load_pose(cam_npz_path):
    """data_preprocessing.py와 동일한 규칙으로 (98, 6) pose 계산."""
    cp = np.load(cam_npz_path)
    positions = cp["positions"]        # (98, 3)
    orientations = cp["orientations"]  # (98, 4) wxyz

    cam0 = positions[0]
    origin = np.array([cam0[0], cam0[1] + 0.50, cam0[2] - 0.03])
    voxel_pos = (positions - origin) / VOXEL_SIZE

    quat_xyzw = np.stack(
        [orientations[:, 1], orientations[:, 2], orientations[:, 3], orientations[:, 0]],
        axis=1,
    )
    aa = Rotation.from_quat(quat_xyzw).as_rotvec()
    return np.hstack([voxel_pos, aa]).astype(np.float64)


def compute_per_view_visibility(poses, gt_3d):
    """gen_visibility.compute_visibility와 동일 로직, view별 binary 결과를 따로 보관.

    Returns
    -------
    per_view : (n_cams, 60, 120, 80) uint8, 1 = 해당 카메라에서 그 voxel이 보임
    """
    n_cams = poses.shape[0]
    per_view = np.zeros((n_cams, N_VOXELS), dtype=np.uint8)

    occ_flat = gt_3d.ravel() > 0.5
    occ_centers = ALL_CENTERS[occ_flat]
    has_occ = len(occ_centers) > 0

    for i in range(n_cams):
        cam_pos = poses[i, :3].astype(np.float64)
        cam_aa = poses[i, 3:].astype(np.float64)
        R_cw = Rotation.from_rotvec(cam_aa).as_matrix().T  # (3, 3)

        # 1. 점유 voxel로 depth buffer 구성
        depth_buf = np.full((IMG_H, IMG_W), np.inf, dtype=np.float64)
        if has_occ:
            pts_occ = (R_cw @ (occ_centers - cam_pos).T).T
            front = pts_occ[:, 2] > 0
            if front.any():
                pf = pts_occ[front]
                inv_z = 1.0 / pf[:, 2]
                u = np.round(fx * pf[:, 0] * inv_z + cx).astype(np.int32)
                v = np.round(fy * pf[:, 1] * inv_z + cy).astype(np.int32)
                d = pf[:, 2]
                ok = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
                np.minimum.at(depth_buf, (v[ok], u[ok]), d[ok])

        # 2. 모든 voxel을 투영하고 가시성 체크
        pts_all = (R_cw @ (ALL_CENTERS - cam_pos).T).T
        front_all = pts_all[:, 2] > 0
        idx = np.where(front_all)[0]
        pf = pts_all[idx]
        inv_z = 1.0 / pf[:, 2]
        u = np.round(fx * pf[:, 0] * inv_z + cx).astype(np.int32)
        v = np.round(fy * pf[:, 1] * inv_z + cy).astype(np.int32)
        d = pf[:, 2]
        ok = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
        idx_ok = idx[ok]
        visible = d[ok] <= depth_buf[v[ok], u[ok]] + 0.5
        per_view[i, idx_ok[visible]] = 1

    return per_view.reshape((n_cams, VZ, VX, VY))


def _save_packed(out_path, per_view):
    """마지막 축(80) packbits → (n_cams, VZ, VX, 10) uint8로 .npy 저장. mmap 가능."""
    packed = np.packbits(per_view, axis=-1)
    np.save(out_path, packed)


def _process_push(push_dir):
    cam_npz = os.path.join(push_dir, "camera_poses.npz")
    pre_gt_path = os.path.join(push_dir, "pre_gt.npz")
    if not (os.path.exists(cam_npz) and os.path.exists(pre_gt_path)):
        return push_dir

    pre_out = os.path.join(push_dir, "pre_ray_casting.npy")
    post_dirs = sorted(d for d in glob(os.path.join(push_dir, "post_*")) if os.path.isdir(d))
    post_targets = []
    for pd in post_dirs:
        gt_path = os.path.join(pd, "gt.npz")
        if not os.path.exists(gt_path):
            continue
        post_out = os.path.join(pd, "post_ray_casting.npy")
        if not os.path.exists(post_out):
            post_targets.append((gt_path, post_out))

    if os.path.exists(pre_out) and not post_targets:
        return push_dir

    pose_npy = os.path.join(push_dir, "pose.npy")
    pose = np.load(pose_npy) if os.path.exists(pose_npy) else load_pose(cam_npz)
    if pose.shape[0] != NUM_CAMERAS:
        print(f"[skip] {push_dir}: pose shape {pose.shape}", flush=True)
        return push_dir

    if not os.path.exists(pre_out):
        pre_gt = np.load(pre_gt_path)["gt"]
        per_view = compute_per_view_visibility(pose, pre_gt)
        _save_packed(pre_out, per_view)

    for gt_path, post_out in post_targets:
        post_gt = np.load(gt_path)["gt"]
        per_view = compute_per_view_visibility(pose, post_gt)
        _save_packed(post_out, per_view)

    return push_dir


def _process_push_safe(push_dir):
    try:
        return _process_push(push_dir)
    except Exception as e:
        print(f"[error] {push_dir}: {e}", flush=True)
        return push_dir


def generate_per_view_ray_casting(data_roots, workers=16):
    push_dirs = []
    for root in data_roots:
        found = sorted(glob(os.path.join(root, "*/push_*")))
        print(f"[ray_casting] {root}: {len(found)}개 push")
        push_dirs.extend(found)

    print(f"[ray_casting] 합계 {len(push_dirs)}개 push 처리 (workers={workers})")
    if not push_dirs:
        return

    with Pool(workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(_process_push_safe, push_dirs),
            total=len(push_dirs),
            desc="ray_casting",
        ):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-roots", type=str, nargs="+", default=DEFAULT_DATA_ROOTS,
                        help="하나 이상의 데이터셋 루트 디렉토리")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    generate_per_view_ray_casting(args.data_roots, workers=args.workers)
