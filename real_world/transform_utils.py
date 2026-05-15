"""SE(3) transform helpers.

All transforms are 4x4 homogeneous matrices.
Convention: T_AB transforms points from frame B to frame A:
    p_A = T_AB @ p_B
"""
import numpy as np
from scipy.spatial.transform import Rotation as R


def rt_to_transform(R_mat, t):
    """Build a 4x4 from a 3x3 rotation and a length-3 translation."""
    R_mat = np.asarray(R_mat, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    assert R_mat.shape == (3, 3), f"R must be (3,3), got {R_mat.shape}"

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    return T


def inverse_transform(T):
    """Inverse of a 4x4 rigid transform (no scaling)."""
    T = np.asarray(T, dtype=np.float64)
    R_mat = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R_mat.T
    T_inv[:3, 3] = -R_mat.T @ t
    return T_inv


def rotvec_t_to_transform(rotvec, t):
    """Build a 4x4 from an axis-angle rotation vector and a translation.

    Matches scipy convention: R = Rotation.from_rotvec(rotvec).as_matrix().
    """
    R_mat = R.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_matrix()
    return rt_to_transform(R_mat, t)


def transform_to_pq(T, quat="wxyz"):
    """Decompose a 4x4 transform into (position, quaternion)."""
    T = np.asarray(T, dtype=np.float64)
    p = T[:3, 3].copy()
    q_xyzw = R.from_matrix(T[:3, :3]).as_quat()  # scipy returns xyzw
    if quat == "xyzw":
        return p, q_xyzw
    elif quat == "wxyz":
        return p, np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    else:
        raise ValueError("quat must be 'wxyz' or 'xyzw'")


def pq_to_transform(p, q, quat="wxyz"):
    """Build a 4x4 from (position, quaternion)."""
    q = np.asarray(q, dtype=np.float64)
    if quat == "wxyz":
        q_xyzw = np.array([q[1], q[2], q[3], q[0]])
    elif quat == "xyzw":
        q_xyzw = q
    else:
        raise ValueError("quat must be 'wxyz' or 'xyzw'")
    R_mat = R.from_quat(q_xyzw).as_matrix()
    return rt_to_transform(R_mat, p)


def _rpy_to_rot_mat(rpy, degrees=False):
    """[rx, ry, rz] -> 3x3 R = Rz(rz) @ Ry(ry) @ Rx(rx)  (Piper convention)."""
    rpy = np.asarray(rpy, dtype=np.float64)
    if degrees:
        rpy = np.deg2rad(rpy)
    rx, ry, rz = rpy
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _rot_mat_to_rpy(R_mat, degrees=False, eps=1e-12):
    """Inverse of _rpy_to_rot_mat. Returns [rx, ry, rz] with ZYX gimbal lock handled."""
    R_mat = np.asarray(R_mat, dtype=np.float64)
    sy = -R_mat[2, 0]
    sy = np.clip(sy, -1.0 + eps, 1.0 - eps)
    pitch = np.arcsin(sy)
    cy = np.cos(pitch)
    if np.abs(cy) > 1e-8:
        roll = np.arctan2(R_mat[2, 1], R_mat[2, 2])
        yaw = np.arctan2(R_mat[1, 0], R_mat[0, 0])
    else:
        roll = 0.0
        yaw = np.arctan2(-R_mat[0, 1], R_mat[1, 1])
    rpy = np.array([roll, pitch, yaw], dtype=np.float64)
    if degrees:
        rpy = np.rad2deg(rpy)
    return rpy


def pose6d_to_transform(pose6d, degrees=False):
    """[x, y, z, rx, ry, rz] -> 4x4 with Piper's Rz@Ry@Rx convention."""
    pose6d = np.asarray(pose6d, dtype=np.float64).reshape(-1)
    t = pose6d[:3]
    R_mat = _rpy_to_rot_mat(pose6d[3:6], degrees=degrees)
    return rt_to_transform(R_mat, t)


def transform_to_pose6d(T, degrees=False):
    """4x4 -> [x, y, z, rx, ry, rz] with Piper's Rz@Ry@Rx convention."""
    T = np.asarray(T, dtype=np.float64)
    t = T[:3, 3]
    rpy = _rot_mat_to_rpy(T[:3, :3], degrees=degrees)
    return np.concatenate([t, rpy])
