import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from glob import glob
from scipy.spatial.transform import Rotation
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "support_code"))
from gen_visibility import compute_visibility  # noqa: E402

"""
1. OCC 풀어 저장 (extract_occ_views)
    push_X/pre_occ/000.npz ~ 097.npz — 98개 카메라 시점의 pre occupancy
    push_X/post_NN/post_occ/000.npz ~ 097.npz — 각 post 시점의 98개 occupancy
    각 npz 안에 occ 키, shape (60, 120, 80)

2. Pose & Visibility (generate_visibility)
    push_X/pose.npy — (98, 6) 카메라 pose (voxel pos + axis-angle)
    push_X/pre_visibility.npy — pre 시점 visibility, shape (98, 60, 120, 80) 추정 (compute_visibility 출력)
    push_X/post_NN/post_visibility.npy — 각 post 시점 visibility
    push_X/push_visibility.npy — mean(max(post_vis - pre_vis, 0)), 이동한 물체 마스킹 적용

실행하기 전,
    1) pre_occ로 통일
    2) push_1, 2, 3 -> post_00~28로 변경
    3) fallen_log.txt -> fallen_log_revised.txt로 변경 (convert_fallen_log.py)

명령어 :
    cd /home/sylee/codes/DiVE/1_preprocessing
    다른 경로 지정 :
    sudo -E /home/sylee/miniconda3/envs/APOBU/bin/python data_preprocessing.py \
    --data-root /data/APOBU/beliefmap_high_occlusion_0423 --workers 32
    기본값 :
    sudo -E /home/sylee/miniconda3/envs/APOBU/bin/python data_preprocessing.py --workers 32
"""







DEFAULT_DATA_ROOT = "/data/APOBU/beliefmap_low_occlusion_0423"
VOXEL_SIZE = 0.005

NUM_CAMERAS = 98


# ── Pose ─────────────────────────────────────────────────────────────────────

def load_pose(cam_npz_path):
    """camera_poses.npz에서 98개 카메라 전체를 (98, 6) pose로 반환."""
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


# ── 1. OCC 뷰 추출 ──────────────────────────────────────────────────────────

def _extract_single_occ(fpath):
    """한 npz 파일에서 98개 카메라 뷰 전체 추출 (워커 함수)."""
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
    """OCC 뷰 추출 (multiprocessing)."""
    all_files = sorted(
        glob(os.path.join(data_root, "*/push_*/pre_occ.npz"))
        + glob(os.path.join(data_root, "*/push_*/post_*/occ.npz"))
    )
    print(f"[occ] {len(all_files)}개 파일 처리 (workers={workers})")

    with Pool(workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(_extract_single_occ, all_files),
            total=len(all_files),
            desc="occ",
        ):
            pass


# ── 2~5. Visibility 생성 ────────────────────────────────────────────────────

def _process_push_visibility(push_dir):
    """한 push 디렉토리의 visibility 전체 생성 (워커 함수)."""
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

    # pose 저장
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
    # 결과에서 움직인 물체의 pre 위치 (pre_gt > 0 & post_gt == 0) 만 0으로 마스킹
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
    """Visibility 생성 (multiprocessing)."""
    push_dirs = sorted(glob(os.path.join(data_root, "*/push_*")))
    print(f"[visibility] {len(push_dirs)}개 push 처리 (workers={workers})")

    with Pool(workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(_process_push_visibility, push_dirs),
            total=len(push_dirs),
            desc="visibility",
        ):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    extract_occ_views(args.data_root, workers=args.workers)
    generate_visibility(args.data_root, workers=args.workers)
