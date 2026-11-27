"""
Belief map computation from depth observations.

Follows the height-map formulation of the original BEVMapping /
find_update_cells implementation:
  depth image → height map (W, D)
    per (vx, vz) column, the largest vy among the back-projected points
  height map → occupied / free
    vy < height  → occupied  (below the surface = solid)
    vy >= height → free      (at or above the surface = open space)
    height = -1  → unobserved (left untouched)

Beta distribution belief: (α, β) per voxel
  α: occupied evidence count
  β: free evidence count

Voxel space: W=120(x), H=60(z-height), D=80(y-depth), 0.5cm/voxel
Camera convention (Isaac Sim):
  local X = right, local Y = up, local Z = forward
  image u = fx * p_cam_x / p_cam_z + cx
  image v = cy - fy * p_cam_y / p_cam_z  (Y-up → v increases downward)
  depth   = radial (Euclidean) distance
"""

import numpy as np
from scipy.spatial.transform import Rotation

# ── Voxel space constants ──────────────────────────────────────────────────────
VOXEL_W    = 120
VOXEL_H    = 60
VOXEL_D    = 80
VOXEL_SIZE = 0.005  # 0.5 cm/voxel

SHELF_X_OFFSET = -0.30
SHELF_Z_OFFSET =  0.80
SHELF_Y_OFFSET = -0.20

# ── Camera intrinsics (must match inference/sim/scene_cfg.py) ────────────────
FX = FY = 24.0 / 20.955 * 256   # ≈ 293.22 px
CX = CY = 128.0
IMG_H = IMG_W = 256


# ── Quaternion helper ──────────────────────────────────────────────────────────

def quat_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    """wxyz quaternion → 3×3 rotation matrix (world ← camera)."""
    q_xyzw = quat_wxyz[[1, 2, 3, 0]]
    return Rotation.from_quat(q_xyzw).as_matrix()


# ── Height map ────────────────────────────────────────────────────────────────

def depth_to_height_map(
        depth_m: np.ndarray,        # (H_img, W_img) float32 [m], NaN=invalid
        cam_pos: np.ndarray,        # (3,) world xyz
        cam_quat_wxyz: np.ndarray,  # (4,) wxyz
        env_origin: np.ndarray,     # (3,)
) -> np.ndarray:
    """
    depth image → 2D height map in voxel space.

    Same as the original BEVMapping / HeightmapGeneration:
      depth → back-project → 3D world points
      per (vx, vz) column, take the largest vy → height_map

    Returns: (W, D) int16, -1 = unobserved
    """
    H_img, W_img = depth_m.shape

    us = np.arange(W_img, dtype=np.float32)
    vs = np.arange(H_img, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)

    valid = np.isfinite(depth_m) & (depth_m > 0.01)
    d     = np.where(valid, depth_m, 0.0)

    # radial depth → z component (Isaac Sim Y-up camera)
    x_norm = (uu - CX) / FX
    y_norm = (vv - CY) / FY
    z = d / np.sqrt(1.0 + x_norm ** 2 + y_norm ** 2)

    p_cam = np.stack([x_norm * z, y_norm * z, z], axis=-1).reshape(-1, 3)  # (N, 3)
    valid_flat = valid.reshape(-1)

    # camera → world
    R       = quat_wxyz_to_rotmat(cam_quat_wxyz)
    p_world = (R @ p_cam.T).T + cam_pos[None, :]  # (N, 3)

    # world → voxel index
    ox = env_origin[0] + SHELF_X_OFFSET
    oy = env_origin[1] + SHELF_Y_OFFSET
    oz = env_origin[2] + SHELF_Z_OFFSET

    vx = ((p_world[:, 0] - ox) / VOXEL_SIZE).astype(np.int32)
    vy = ((p_world[:, 2] - oz) / VOXEL_SIZE).astype(np.int32)  # world z → height
    vz = ((p_world[:, 1] - oy) / VOXEL_SIZE).astype(np.int32)  # world y → depth

    in_range = (valid_flat &
                (vx >= 0) & (vx < VOXEL_W) &
                (vy >= 0) & (vy < VOXEL_H) &
                (vz >= 0) & (vz < VOXEL_D))

    vx = vx[in_range]
    vy = vy[in_range]
    vz = vz[in_range]

    # Largest vy per (vx, vz) column (= the height find_update_cells uses)
    height_map = np.full((VOXEL_W, VOXEL_D), -1, dtype=np.int16)
    np.maximum.at(height_map, (vx, vz), vy.astype(np.int16))

    return height_map  # (W, D), -1 = unobserved


def height_map_to_obs(height_map: np.ndarray) -> tuple:
    """
    height map → occupied / free mask.

    Same as the original find_update_cells:
      vy < height  → occupied  (below the surface)
      vy >= height → free      (at or above the surface)
      height = -1  → False in both masks (unobserved)

    Returns:
      occupied (H, W, D) float32
      free     (H, W, D) float32
    """
    observed = height_map >= 0  # (W, D)

    h_idx = np.arange(VOXEL_H, dtype=np.int16)  # (H,)
    hm  = height_map[:, :, np.newaxis]           # (W, D, 1)
    obs = observed  [:, :, np.newaxis]           # (W, D, 1)

    occupied = obs & (h_idx[np.newaxis, np.newaxis, :] < hm)   # (W, D, H)
    free     = obs & (h_idx[np.newaxis, np.newaxis, :] >= hm)  # (W, D, H)

    # (W, D, H) → (H, W, D)
    occupied = occupied.transpose(2, 0, 1).astype(np.float32)
    free     = free    .transpose(2, 0, 1).astype(np.float32)

    return occupied, free


# ── Single-view belief update (via the height map) ────────────────────────────

def update_belief_height_map(
    alpha: np.ndarray,          # (W*H*D,) float32, in-place
    beta:  np.ndarray,          # (W*H*D,) float32, in-place
    depth_m: np.ndarray,        # (H_img, W_img) float32 [m]
    cam_pos: np.ndarray,        # (3,) world xyz
    cam_quat_wxyz: np.ndarray,  # (4,) wxyz
    env_origin: np.ndarray,     # (3,)
) -> None:
    """
    One depth image → height map → occupied/free → Beta belief update.

    Beta-distribution version of the original BEVMapping.update_occupancy_map:
      occupied voxel: α += 1
      free     voxel: β += 1
      unobserved:     unchanged
    """
    hm = depth_to_height_map(depth_m, cam_pos, cam_quat_wxyz, env_origin)
    occupied, free = height_map_to_obs(hm)   # (H, W, D) each

    # (H, W, D) → (W, H, D) → flat (W*H*D) — matches the belief array layout
    alpha += occupied.transpose(1, 0, 2).reshape(-1)
    beta  += free    .transpose(1, 0, 2).reshape(-1)


# ── Multi-view belief computation ──────────────────────────────────────────────

def compute_belief(
    depth_views: np.ndarray,    # (N_views, H_img, W_img) float32 [m]
    cam_positions: np.ndarray,  # (N_views, 3)
    cam_quats: np.ndarray,      # (N_views, 4) wxyz
    env_origin: np.ndarray,     # (3,)
    view_indices=None,
    prior_alpha: float = 0.5,
    prior_beta:  float = 0.5,
) -> np.ndarray:
    """
    Accumulate a belief map over several views.
    Returns: (W, H, D, 2) float32, [..., 0]=α, [..., 1]=β
    """
    N = VOXEL_W * VOXEL_H * VOXEL_D
    alpha = np.full(N, prior_alpha, dtype=np.float32)
    beta  = np.full(N, prior_beta,  dtype=np.float32)

    if view_indices is None:
        view_indices = range(len(depth_views))

    for i in view_indices:
        update_belief_height_map(
            alpha, beta,
            depth_views[i], cam_positions[i], cam_quats[i],
            env_origin,
        )

    belief = np.stack([alpha, beta], axis=-1)
    return belief.reshape(VOXEL_W, VOXEL_H, VOXEL_D, 2)


def compute_incremental_beliefs(
    depth_views: np.ndarray,    # (N_views, H_img, W_img) float32 [m]
    cam_positions: np.ndarray,  # (N_views, 3)
    cam_quats: np.ndarray,      # (N_views, 4) wxyz
    env_origin: np.ndarray,     # (3,)
    prior_alpha: float = 0.5,
    prior_beta:  float = 0.5,
) -> np.ndarray:
    """
    Sequential belief update: returns the belief after each additional view.
    Used to generate σ_o training data.

    Returns: (N_views+1, W, H, D, 2) float32
      [0] = prior
      [k] = after observing views 0..k-1
    """
    N_views = len(depth_views)
    N       = VOXEL_W * VOXEL_H * VOXEL_D
    shape   = (VOXEL_W, VOXEL_H, VOXEL_D, 2)

    alpha = np.full(N, prior_alpha, dtype=np.float32)
    beta  = np.full(N, prior_beta,  dtype=np.float32)

    results = np.empty((N_views + 1,) + shape, dtype=np.float32)
    results[0] = np.stack([alpha, beta], axis=-1).reshape(shape)

    for i in range(N_views):
        update_belief_height_map(
            alpha, beta,
            depth_views[i], cam_positions[i], cam_quats[i],
            env_origin,
        )
        results[i + 1] = np.stack([alpha.copy(), beta.copy()], axis=-1).reshape(shape)

    return results


# ── Utility ────────────────────────────────────────────────────────────────────

def depth_semantic_to_voxel(
        depth_m: np.ndarray,          # (H_img, W_img) float32 [m]
        semantic_2d: np.ndarray,      # (H_img, W_img) int32
        cam_pos: np.ndarray,          # (3,)
        cam_quat_wxyz: np.ndarray,    # (4,) wxyz
        env_origin: np.ndarray,       # (3,)
) -> np.ndarray:
    """
    per-view depth + semantic image → 3D voxel semantic map (W, H, D) int32.

    Same point-cloud style as occ_pre.npz: each depth pixel is back-projected
    and its semantic label written to the single voxel it lands in. Columns
    are not filled in.

    Returns: (W, H, D) int32, 0 = unobserved / background
    """
    H_img, W_img = depth_m.shape
    us = np.arange(W_img, dtype=np.float32)
    vs = np.arange(H_img, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)

    valid = np.isfinite(depth_m) & (depth_m > 0.01)
    d     = np.where(valid, depth_m, 0.0)

    x_norm = (uu - CX) / FX
    y_norm = (vv - CY) / FY
    z = d / np.sqrt(1.0 + x_norm ** 2 + y_norm ** 2)

    p_cam   = np.stack([x_norm * z, y_norm * z, z], axis=-1).reshape(-1, 3)
    valid_f = valid.reshape(-1)
    sem_f   = semantic_2d.reshape(-1)

    R       = quat_wxyz_to_rotmat(cam_quat_wxyz)
    p_world = (R @ p_cam.T).T + cam_pos[None, :]

    ox = env_origin[0] + SHELF_X_OFFSET
    oy = env_origin[1] + SHELF_Y_OFFSET
    oz = env_origin[2] + SHELF_Z_OFFSET

    vx = ((p_world[:, 0] - ox) / VOXEL_SIZE).astype(np.int32)
    vy = ((p_world[:, 2] - oz) / VOXEL_SIZE).astype(np.int32)
    vz = ((p_world[:, 1] - oy) / VOXEL_SIZE).astype(np.int32)

    in_range = (valid_f &
                (vx >= 0) & (vx < VOXEL_W) &
                (vy >= 0) & (vy < VOXEL_H) &
                (vz >= 0) & (vz < VOXEL_D))

    vx_v = vx[in_range]; vy_v = vy[in_range]
    vz_v = vz[in_range]; sem_v = sem_f[in_range]

    # point cloud: one voxel per pixel; on collision the last write wins
    sem_3d = np.zeros((VOXEL_W, VOXEL_H, VOXEL_D), dtype=np.int32)
    sem_3d[vx_v, vy_v, vz_v] = sem_v

    return sem_3d  # (W, H, D)


def belief_to_probability(belief: np.ndarray) -> np.ndarray:
    """Beta(α, β) → E[p] = α / (α + β)."""
    a = belief[..., 0]
    b = belief[..., 1]
    return a / (a + b + 1e-8)


def depth_to_obs(
        depth_m: np.ndarray,
        cam_pos: np.ndarray,
        cam_quat_wxyz: np.ndarray,
        env_origin: np.ndarray,
) -> tuple:
    """One depth image → occupied (H,W,D), free (H,W,D) float32."""
    hm = depth_to_height_map(depth_m, cam_pos, cam_quat_wxyz, env_origin)
    return height_map_to_obs(hm)


def load_depth_from_npz(rgbd_npz) -> np.ndarray:
    """rgbd.npz → (N_views, H, W) float32 [m]."""
    if 'depth_uint16' in rgbd_npz:
        d = rgbd_npz['depth_uint16'].astype(np.float32) / 1000.0
        d[rgbd_npz['depth_uint16'] == 0] = np.nan
        return d
    return rgbd_npz['depth']
