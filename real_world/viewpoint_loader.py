"""Load 98 viewpoints from pose.npy and convert them to 4x4 transforms.

pose.npy format (per the APOBU dataset):
    shape (N, 6)  — typically N = 98
    columns [0:3] — voxel index of the camera in the voxel-map frame S
                    (1 voxel = 5 mm by default)
    columns [3:6] — axis-angle rotation vector (scipy.Rotation.from_rotvec)

Output: list of 4x4 transforms T_sc_i where i is the viewpoint index, each
mapping points in the i-th camera frame to the voxel-map (shelf) frame S.
"""
import numpy as np

from transform_utils import rotvec_t_to_transform


def load_viewpoints(pose_npy_path, voxel_size_m=0.005):
    """Load pose.npy and return (98, 4, 4) array of T_sc_i transforms.

    Args:
        pose_npy_path: path to pose.npy.
        voxel_size_m: meters per voxel index (default 0.005 m = 5 mm).

    Returns:
        T_sc: (N, 4, 4) float64 array.
    """
    raw = np.load(pose_npy_path)
    if raw.ndim != 2 or raw.shape[1] != 6:
        raise ValueError(f"pose.npy must be (N,6), got {raw.shape}")

    N = raw.shape[0]
    T_sc = np.zeros((N, 4, 4), dtype=np.float64)
    for i in range(N):
        voxel_xyz = raw[i, 0:3]
        rotvec = raw[i, 3:6]
        t = voxel_xyz * float(voxel_size_m)
        T_sc[i] = rotvec_t_to_transform(rotvec, t)
    return T_sc


def summarize_viewpoints(T_sc):
    """Print a quick layout sanity check for debugging."""
    pts = T_sc[:, :3, 3]
    print(f"  viewpoints: {len(T_sc)}")
    print(f"  x range:  [{pts[:, 0].min():+.3f}, {pts[:, 0].max():+.3f}] m"
          f"   ({len(np.unique(np.round(pts[:, 0], 4)))} unique)")
    print(f"  y range:  [{pts[:, 1].min():+.3f}, {pts[:, 1].max():+.3f}] m"
          f"   ({len(np.unique(np.round(pts[:, 1], 4)))} unique)")
    print(f"  z range:  [{pts[:, 2].min():+.3f}, {pts[:, 2].max():+.3f}] m"
          f"   ({len(np.unique(np.round(pts[:, 2], 4)))} unique)")
