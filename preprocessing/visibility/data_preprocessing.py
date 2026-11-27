import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from glob import glob
from scipy.spatial.transform import Rotation
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from gen_visibility import compute_visibility  # noqa: E402

"""
1. Unpack the stacked occupancy volumes (extract_occ_views)
    push_X/pre_occ/000.npz ~ 097.npz — pre-push occupancy for each of the 98 camera views
    push_X/post_NN/post_occ/000.npz ~ 097.npz — the same 98 views for every post state
    Every npz holds a single `occ` key of shape (60, 120, 80).

2. Pose & visibility (generate_visibility)
    push_X/pose.npy — (98, 6) camera poses (voxel-space position + axis-angle)
    push_X/pre_visibility.npy — pre-push visibility, (60, 120, 80) float32 in [0, 1]
        (compute_visibility output: fraction of the 98 cameras that see each voxel)
    push_X/post_NN/post_visibility.npy — the same for every post state
    push_X/push_visibility.npy — mean(max(post_vis - pre_vis, 0)), with voxels the
        push emptied out (occupied before, free after) masked to 0

Before running, the raw collection has to be normalized:
    1) name every pre-push occupancy file pre_occ
    2) group the per-episode post directories under push_1, push_2, ... so that each
       push_N holds its own post_NN sequence
    3) fallen_log.txt -> fallen_log_revised.txt, converted inline by
       preprocessing_run_all.sh (step 0)

Usage:
    cd <repo root>
    python preprocessing/visibility/data_preprocessing.py \
        --data-root /path/to/data/beliefmap_high_occlusion_ycb_v3 --workers 32
"""




DEFAULT_DATA_ROOT = None  # pass --data-root
VOXEL_SIZE = 0.005

NUM_CAMERAS = 98


# ── Pose ─────────────────────────────────────────────────────────────────────

def load_pose(cam_npz_path):
    """Load all 98 camera poses from camera_poses.npz as a (98, 6) array."""
    cp = np.load(cam_npz_path)
    positions = cp["positions"]       # (98, 3)
    orientations = cp["orientations"]  # (98, 4) wxyz

    cam0 = positions[0]
    origin = np.array([cam0[0], cam0[1] + 0.50, cam0[2] - 0.03])

    voxel_pos = (positions - origin) / VOXEL_SIZE
    quat_xyzw = np.stack(
        [orientations[:, 1], orientations[:, 2], orientations[:, 3], orientations[:, 0]],
        axis=1,
    )
    aa = Rotation.from_quat(quat_xyzw).as_rotvec()
    return np.hstack([voxel_pos, aa]).astype(np.float64)  # (98, 6)


# ── 1. Unpack occupancy views ───────────────────────────────────────────────

def _extract_single_occ(fpath):
    """Split one stacked npz into its 98 per-camera views (pool worker)."""
    parent = os.path.dirname(fpath)
    # pre_occ.npz -> pre_occ/,  occ.npz -> post_occ/
    basename = os.path.splitext(os.path.basename(fpath))[0]
    name = "post_occ" if basename == "occ" else basename
    out_dir = os.path.join(parent, name)

    if os.path.isdir(out_dir) and all(
        os.path.exists(os.path.join(out_dir, f"{idx:03d}.npz"))
        for idx in range(NUM_CAMERAS)
    ):
        return fpath

    os.makedirs(out_dir, exist_ok=True)
    data = np.load(fpath)["occupied"]  # (98, 60, 120, 80)
    for idx in range(NUM_CAMERAS):
        out_path = os.path.join(out_dir, f"{idx:03d}.npz")
        if os.path.exists(out_path):
            continue
        np.savez_compressed(out_path, occ=data[idx])
    return fpath


def extract_occ_views(data_root, workers=16):
    """Unpack every stacked occupancy volume under data_root (multiprocessing)."""
    all_files = sorted(
        glob(os.path.join(data_root, "*/push_*/pre_occ.npz"))
        + glob(os.path.join(data_root, "*/push_*/post_*/occ.npz"))
    )
    print(f"[occ] processing {len(all_files)} files (workers={workers})")

    with Pool(workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(_extract_single_occ, all_files),
            total=len(all_files),
            desc="occ",
        ):
            pass


# ── 2. Visibility generation ────────────────────────────────────────────────

def _process_push_visibility(push_dir):
    """Generate every visibility map for a single push directory (pool worker)."""
    cam_npz = os.path.join(push_dir, "camera_poses.npz")
    pre_gt_path = os.path.join(push_dir, "pre_gt.npz")
    if not os.path.exists(cam_npz) or not os.path.exists(pre_gt_path):
        return push_dir

    pose_path = os.path.join(push_dir, "pose.npy")
    pre_vis_path = os.path.join(push_dir, "pre_visibility.npy")
    pv_path = os.path.join(push_dir, "push_visibility.npy")
    post_dirs_early = sorted(
        d for d in glob(os.path.join(push_dir, "post_*")) if os.path.isdir(d)
    )
    if (
        os.path.exists(pose_path)
        and os.path.exists(pre_vis_path)
        and os.path.exists(pv_path)
        and all(
            os.path.exists(os.path.join(pd, "post_visibility.npy"))
            for pd in post_dirs_early
            if os.path.exists(os.path.join(pd, "gt.npz"))
        )
    ):
        return push_dir

    pose = load_pose(cam_npz)  # (98, 6)

    # save the poses
    pose_path = os.path.join(push_dir, "pose.npy")
    if not os.path.exists(pose_path):
        np.save(pose_path, pose)

    pre_gt = np.load(pre_gt_path)["gt"]  # (60, 120, 80)

    # pre_visibility
    pre_vis_path = os.path.join(push_dir, "pre_visibility.npy")
    if os.path.exists(pre_vis_path):
        pre_vis = np.load(pre_vis_path)
    else:
        pre_vis = compute_visibility(pose, pre_gt).astype(np.float32)
        np.save(pre_vis_path, pre_vis)

    # per-post visibility
    post_dirs = sorted(
        d for d in glob(os.path.join(push_dir, "post_*"))
        if os.path.isdir(d)
    )

    post_vis_list = []
    post_gt_list = []
    for pd in post_dirs:
        gt_path = os.path.join(pd, "gt.npz")
        if not os.path.exists(gt_path):
            continue

        post_gt = np.load(gt_path)["gt"]  # (60, 120, 80)
        post_gt_list.append(post_gt)

        # post_visibility
        vis_path = os.path.join(pd, "post_visibility.npy")
        if os.path.exists(vis_path):
            vis = np.load(vis_path)
        else:
            vis = compute_visibility(pose, post_gt).astype(np.float32)
            np.save(vis_path, vis)
        post_vis_list.append(vis)

    # push_visibility = mean_{j} max(vis_j - pre_vis, 0)
    # Voxels the push vacated (occupied in pre_gt, free in post_gt) are zeroed out:
    # the object moving away trivially "reveals" them, which is not what the push
    # target should reward.
    pv_path = os.path.join(push_dir, "push_visibility.npy")
    if not os.path.exists(pv_path) and post_vis_list:
        acc = np.zeros_like(pre_vis, dtype=np.float32)
        for vis_j, gt_j in zip(post_vis_list, post_gt_list):
            diff = np.maximum(vis_j - pre_vis, 0.0)
            moved_mask = (pre_gt > 0.5) & (gt_j < 0.5)
            diff[moved_mask] = 0.0
            acc += diff
        acc /= len(post_vis_list)
        np.save(pv_path, acc)

    return push_dir


def generate_visibility(data_root, workers=16):
    """Generate visibility maps for every push under data_root (multiprocessing)."""
    push_dirs = sorted(glob(os.path.join(data_root, "*/push_*")))
    print(f"[visibility] processing {len(push_dirs)} pushes (workers={workers})")

    with Pool(workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(_process_push_visibility, push_dirs),
            total=len(push_dirs),
            desc="visibility",
        ):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    extract_occ_views(args.data_root, workers=args.workers)
    generate_visibility(args.data_root, workers=args.workers)
