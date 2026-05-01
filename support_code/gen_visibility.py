"""
Generate per-voxel visibility count from 96 camera viewpoints.

For each voxel in the (60, 120, 80) grid, counts how many of the 96 cameras
can "see" it — i.e., a ray from the camera reaches the voxel without being
blocked by an occupied voxel in front.

Input
-----
  /data/APOBU/processed_data/{ep:09d}/pose.npy       (96, 6) voxel-coord poses
  /data/APOBU/beliefmap_0325_all/{ep:09d}/data.npz    gt_3d (60, 120, 80)

Output
------
  /data/APOBU/U_AB/{ep:09d}/visibility.npy            (60, 120, 80) float32, 0~96
"""

import argparse
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

# ── Camera intrinsics ────────────────────────────────────────────────────────
focal_length = 24.0
horizontal_aperture = 20.955
IMG_W = IMG_H = 256
fx = fy = focal_length / horizontal_aperture * IMG_W   # ~293.22
cx = cy = IMG_W / 2.0                                  # 128.0

# ── Voxel grid ───────────────────────────────────────────────────────────────
VZ, VX, VY = 60, 120, 80
N_VOXELS = VZ * VX * VY  # 576,000

# Precompute all voxel centers in world-axis voxel coords (x, y, z)
# gt_3d is stored as (Vz, Vx, Vy) → index [iz, ix, iy]
#   world x = ix + 0.5,  world y = iy + 0.5,  world z = iz + 0.5
_iz, _ix, _iy = np.mgrid[0:VZ, 0:VX, 0:VY]
ALL_CENTERS = np.stack([
    _ix.ravel().astype(np.float64) + 0.5,   # world x
    _iy.ravel().astype(np.float64) + 0.5,   # world y
    _iz.ravel().astype(np.float64) + 0.5,   # world z
], axis=1)  # (576000, 3)


def compute_visibility(poses: np.ndarray, gt_3d: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    poses : (N_cam, 6)  — [x, y, z, ax, ay, az] in voxel coords (world axis order)
    gt_3d : (60, 120, 80) — binary occupancy

    Returns
    -------
    visibility : (60, 120, 80) float32, count of cameras that see each voxel (0~N_cam)
    """
    n_cams = poses.shape[0]
    vis_count = np.zeros(N_VOXELS, dtype=np.float32)

    # Occupied voxel centers for depth buffer construction
    occ_flat = gt_3d.ravel() > 0.5
    occ_centers = ALL_CENTERS[occ_flat]  # (N_occ, 3)
    has_occ = len(occ_centers) > 0

    for i in range(n_cams):
        cam_pos = poses[i, :3].astype(np.float64)
        cam_aa = poses[i, 3:].astype(np.float64)

        # camera-to-world → world-to-camera rotation
        R_cw = Rotation.from_rotvec(cam_aa).as_matrix().T  # (3, 3)

        # ── 1. Build depth buffer from occupied voxels ───────────────────
        depth_buf = np.full((IMG_H, IMG_W), np.inf, dtype=np.float64)

        if has_occ:
            pts_occ = (R_cw @ (occ_centers - cam_pos).T).T  # (N_occ, 3)
            front = pts_occ[:, 2] > 0
            if front.any():
                pf = pts_occ[front]
                inv_z = 1.0 / pf[:, 2]
                u = np.round(fx * pf[:, 0] * inv_z + cx).astype(np.int32)
                v = np.round(fy * pf[:, 1] * inv_z + cy).astype(np.int32)
                d = pf[:, 2]
                ok = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
                np.minimum.at(depth_buf, (v[ok], u[ok]), d[ok])

        # ── 2. Project ALL voxels, check visibility ──────────────────────
        pts_all = (R_cw @ (ALL_CENTERS - cam_pos).T).T  # (576000, 3)
        front_all = pts_all[:, 2] > 0
        idx = np.where(front_all)[0]
        pf = pts_all[idx]

        inv_z = 1.0 / pf[:, 2]
        u = np.round(fx * pf[:, 0] * inv_z + cx).astype(np.int32)
        v = np.round(fy * pf[:, 1] * inv_z + cy).astype(np.int32)
        d = pf[:, 2]

        ok = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
        idx_ok = idx[ok]
        # visible = depth ≤ first-occupied depth + tolerance (include surface voxel)
        visible = d[ok] <= depth_buf[v[ok], u[ok]] + 0.5
        vis_count[idx_ok[visible]] += 1.0

    return vis_count.reshape((VZ, VX, VY)) / n_cams  # normalize to 0~1


# ── Per-episode worker ────────────────────────────────────────────────────────

PROC_ROOT = Path("/data/APOBU/processed_data")
OCC_ROOT = Path("/data/APOBU/beliefmap_0325_all")
OUT_ROOT = Path("/data/APOBU/U_AB")


def process_episode(ep_id: int) -> int:
    out_dir = OUT_ROOT / f"{ep_id:09d}"
    out_path = out_dir / "visibility.npy"
    if out_path.exists():
        return ep_id  # skip already done

    pose = np.load(str(PROC_ROOT / f"{ep_id:09d}" / "pose.npy"))   # (96, 6)
    data = np.load(str(OCC_ROOT / f"{ep_id:09d}" / "data.npz"))
    gt_3d = data["gt_3d"]  # (60, 120, 80)

    vis = compute_visibility(pose, gt_3d)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), vis)
    return ep_id


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--test", action="store_true", help="Run 1 episode only")
    args = parser.parse_args()

    ep_ids = sorted(
        int(p.name) for p in PROC_ROOT.iterdir() if p.is_dir()
    )
    print(f"Total episodes: {len(ep_ids)}, workers: {args.workers}")

    if args.test:
        ep_ids = ep_ids[:1]
        print("=== TEST MODE: 1 episode ===")
        import time
        t0 = time.time()
        process_episode(ep_ids[0])
        elapsed = time.time() - t0
        print(f"Episode {ep_ids[0]} done in {elapsed:.1f}s")

        # Quick sanity check
        vis = np.load(str(OUT_ROOT / f"{ep_ids[0]:09d}" / "visibility.npy"))
        gt = np.load(str(OCC_ROOT / f"{ep_ids[0]:09d}" / "data.npz"))["gt_3d"]
        print(f"visibility shape: {vis.shape}, min: {vis.min():.0f}, max: {vis.max():.0f}, "
              f"mean (all): {vis.mean():.2f}, mean (occupied): {vis[gt > 0.5].mean():.2f}")
    else:
        with Pool(args.workers) as pool:
            list(tqdm(
                pool.imap_unordered(process_episode, ep_ids),
                total=len(ep_ids),
                desc="Generating visibility",
            ))
        print("Done!")
