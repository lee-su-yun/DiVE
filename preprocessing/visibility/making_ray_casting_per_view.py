"""
making_ray_casting_per_view.py

For every push directory, store the binary visibility (ray casting) of each of the
98 camera views separately, so that training can mmap-slice just the views in K_t
and assemble the ground truth on the fly.

Output
------
  push_X/pre_ray_casting.npy           shape (98, 60, 120, 10) uint8
  push_X/post_NN/post_ray_casting.npy  shape (98, 60, 120, 10) uint8

  The last axis (80 voxels) is bit-packed with np.packbits into 10 bytes, giving
  exactly 6.73 MiB per file.
  To restore: np.unpackbits(packed[k], axis=-1) -> (60, 120, 80) uint8 binary

Usage in training
-----------------
    packed = np.load(path, mmap_mode='r')             # (98, 60, 120, 10)
    view_k = np.unpackbits(packed[k], axis=-1)        # (60, 120, 80) uint8

This uses the same ray-casting pipeline as compute_visibility (depth buffer plus a
0.5-voxel surface tolerance), so averaging the result over the view axis reproduces
pre_visibility.npy exactly.

Usage
-----
    cd <repo root>
    python preprocessing/visibility/making_ray_casting_per_view.py \
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
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# reuses the camera intrinsics and voxel-grid constants from common/gen_visibility.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from gen_visibility import (  # noqa: E402
    ALL_CENTERS, fx, fy, cx, cy, IMG_W, IMG_H, VZ, VX, VY, N_VOXELS,
)


VOXEL_SIZE = 0.005
NUM_CAMERAS = 98


def load_pose(cam_npz_path):
    """Compute the (98, 6) camera poses with the same convention as data_preprocessing.py."""
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
    """Same logic as gen_visibility.compute_visibility, but keeping each view's
    binary result instead of averaging them.

    Returns
    -------
    per_view : (n_cams, 60, 120, 80) uint8, 1 = that camera sees that voxel
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

        # 1. Build the depth buffer from the occupied voxels
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

        # 2. Project every voxel and test its visibility
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
    """Bit-pack the last axis (80) and save as an mmap-friendly (n_cams, VZ, VX, 10) uint8 .npy."""
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
        print(f"[ray_casting] {root}: {len(found)} pushes")
        push_dirs.extend(found)

    print(f"[ray_casting] {len(push_dirs)} pushes in total (workers={workers})")
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
    parser.add_argument("--data-roots", type=str, nargs="+", required=True,
                        help="One or more dataset root directories")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    generate_per_view_ray_casting(args.data_roots, workers=args.workers)
