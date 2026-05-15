"""Step 2: turn the 98 sim viewpoints into world-frame EE commands.

For each viewpoint i:
    T_wc_i = T_ws @ T_sc_i             (target camera pose in world)
    T_we_i = T_wc_i @ inv(T_ec)        (target EE pose in world)
"""
import numpy as np

from transform_utils import inverse_transform, transform_to_pq


def solve_ee_poses(T_ws, T_sc_list, T_ec):
    """Compute EE poses for every viewpoint.

    Args:
        T_ws: (4,4) voxel-map origin -> world.
        T_sc_list: (N,4,4) camera_i -> voxel-map origin for each viewpoint.
        T_ec: (4,4) EE -> camera (constant hand-eye offset).

    Returns:
        T_we: (N,4,4) target EE pose in world frame for each viewpoint.
    """
    T_ws = np.asarray(T_ws, dtype=np.float64)
    T_sc_list = np.asarray(T_sc_list, dtype=np.float64)
    T_ec = np.asarray(T_ec, dtype=np.float64)
    T_ce = inverse_transform(T_ec)

    N = T_sc_list.shape[0]
    T_we = np.zeros((N, 4, 4), dtype=np.float64)
    for i in range(N):
        T_wc_i = T_ws @ T_sc_list[i]
        T_we[i] = T_wc_i @ T_ce
    return T_we


def ee_poses_to_pq(T_we_list, quat="wxyz"):
    """Convert (N,4,4) to (N,3) positions and (N,4) quaternions."""
    N = T_we_list.shape[0]
    positions = np.zeros((N, 3), dtype=np.float64)
    quats = np.zeros((N, 4), dtype=np.float64)
    for i in range(N):
        p, q = transform_to_pq(T_we_list[i], quat=quat)
        positions[i] = p
        quats[i] = q
    return positions, quats


def save_ee_poses(out_path, T_we_list, T_ws, T_ec, T_tag_voxel):
    """Save EE poses + metadata to a single .npz file."""
    positions, quats_wxyz = ee_poses_to_pq(T_we_list, quat="wxyz")
    np.savez(
        out_path,
        T_we=T_we_list,
        positions=positions,
        quaternions_wxyz=quats_wxyz,
        T_ws=T_ws,
        T_ec=T_ec,
        T_tag_voxel=T_tag_voxel,
    )
