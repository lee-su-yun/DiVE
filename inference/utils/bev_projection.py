"""
2D BEV data utilities for the CNABU baseline.

Same height-as-channels representation as the CNABU paper:
  - Occupancy belief : (2×H_BINS, W, D) — β₀,α₀,β₁,α₁,...
  - Semantic belief  : (N_SEM, W, D)    — 2D BEV Dirichlet

Difference from the sibling module:
  - voxel_projection.py → full 3D voxel representation (W, H, D)
  - bev_projection.py   → 2D BEV + height-as-channels (the paper's form)
"""

import numpy as np
from inference.utils.voxel_projection import (
    quat_wxyz_to_rotmat,
    depth_to_height_map,
    height_map_to_obs,
    VOXEL_W, VOXEL_H, VOXEL_D, VOXEL_SIZE,
    SHELF_X_OFFSET, SHELF_Y_OFFSET, SHELF_Z_OFFSET,
    FX, FY, CX, CY,
)

H_BINS = VOXEL_H   # 60
MAP_W  = VOXEL_W   # 120
MAP_D  = VOXEL_D   # 80


# ── per-view observation → 2D BEV semantic ───────────────────────────────────

def depth_semantic_to_bev(
        depth_m: np.ndarray,        # (H_img, W_img) float32 [m]
        semantic_2d: np.ndarray,    # (H_img, W_img) int32
        cam_pos: np.ndarray,        # (3,) world xyz
        cam_quat_wxyz: np.ndarray,  # (4,) wxyz
        env_origin: np.ndarray,     # (3,)
) -> tuple:
    """
    depth + 2D semantic image → (height_map, sem_bev).

    height_map : (W, D) int16  — surface height as a voxel index, -1=unobserved
    sem_bev    : (W, D) int32  — semantic class at the surface, 0=unobserved

    Same height-map-based 2D BEV semantic as the paper: per (vx, vz) column,
    record the semantic label of the pixel with the largest vy.
    """
    H_img, W_img = depth_m.shape
    us = np.arange(W_img, dtype=np.float32)
    vs = np.arange(H_img, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)

    valid = np.isfinite(depth_m) & (depth_m > 0.01)
    d = np.where(valid, depth_m, 0.0)

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
                (vx >= 0) & (vx < MAP_W) &
                (vy >= 0) & (vy < H_BINS) &
                (vz >= 0) & (vz < MAP_D))

    vx = vx[in_range]; vy = vy[in_range]
    vz = vz[in_range]; sem = sem_f[in_range]

    # height map: max vy per (vx, vz)
    height_map = np.full((MAP_W, MAP_D), -1, dtype=np.int16)
    np.maximum.at(height_map, (vx, vz), vy.astype(np.int16))

    # BEV semantic: sort by ascending vy so the last write (= largest vy) wins
    sort_idx = np.argsort(vy)
    sem_bev  = np.zeros((MAP_W, MAP_D), dtype=np.int32)
    sem_bev[vx[sort_idx], vz[sort_idx]] = sem[sort_idx]
    sem_bev[height_map == -1] = 0   # unobserved columns → 0 (background)

    return height_map, sem_bev


# ── 3D GT → 2D BEV GT projection ─────────────────────────────────────────────

def project_semantic_3d_to_2d(
        semantic_3d: np.ndarray,    # (W, H, D) int
        occ_3d: np.ndarray,         # (W, H, D) float  0/1 occupancy GT
) -> np.ndarray:
    """
    3D semantic GT → 2D BEV semantic GT (W, D) int.
    Per (x, z) column, return the class of the topmost occupied voxel.
    Columns with no occupied voxel become 0 (background).
    """
    occupied = occ_3d > 0.5         # (W, H, D) bool
    W, H, D  = semantic_3d.shape

    h_idx    = np.arange(H)[np.newaxis, :, np.newaxis]   # (1, H, 1)
    # height index where occupied, -1 elsewhere
    h_masked = np.where(occupied, h_idx, -1)              # (W, H, D)
    top_h    = h_masked.max(axis=1)                       # (W, D)  -1=no occ

    # read the semantic label at the topmost occupied height
    any_occ  = occupied.any(axis=1)                       # (W, D) bool
    top_h_c  = np.clip(top_h, 0, H - 1)

    x_idx    = np.arange(W)[:, np.newaxis]
    z_idx    = np.arange(D)[np.newaxis, :]
    sem_2d   = semantic_3d[x_idx, top_h_c, z_idx]        # (W, D)
    sem_2d[~any_occ] = 0                                  # empty column → background

    return sem_2d.astype(np.int64)


def project_diff_3d_to_2d(
        pre_gt: np.ndarray,    # (W, H, D) float
        post_gt: np.ndarray,   # (W, H, D) float
) -> np.ndarray:
    """
    3D pre/post GT → 2D BEV difference mask (W, D) float.
    1 wherever the column changed at any height.
    """
    diff_3d = (pre_gt > 0.5) != (post_gt > 0.5)          # (W, H, D) bool
    return diff_3d.any(axis=1).astype(np.float32)         # (W, D)


# ── belief accumulation ───────────────────────────────────────────────────────

def accumulate_belief_cnabu(
        belief: np.ndarray,         # (2×H, W, D) modified in place
        sem_dir: np.ndarray,        # (N_SEM, W, D) modified in place
        height_map: np.ndarray,     # (W, D) int16  surface height (-1=unobserved)
        sem_bev: np.ndarray,        # (W, D) int32  surface semantic
        n_sem: int,
) -> None:
    """
    Accumulate the belief the way the paper's 3D denoising code does:

      belief[1::2] += occupied  (α channels)
      belief[0::2] += free      (β channels)
      sem_dir      += one_hot(sem_bev) * observed_mask
    """
    # height map → occupied/free (H, W, D) each
    occupied_hwd, free_hwd = height_map_to_obs(height_map)   # (H, W, D)

    # channel layout: [β₀, α₀, β₁, α₁, ...]
    belief[1::2] += occupied_hwd   # α += occupied, broadcast over (H, W, D)
    belief[0::2] += free_hwd       # β += free

    # accumulate the 2D BEV semantic counts
    observed_2d = (height_map >= 0).astype(np.float32)       # (W, D)
    # one-hot: (N_SEM, W, D)
    sem_oh = np.zeros((n_sem, MAP_W, MAP_D), dtype=np.float32)
    sem_bev_c = np.clip(sem_bev, 0, n_sem - 1)
    row_idx   = sem_bev_c.ravel()
    col_idx   = np.arange(MAP_W * MAP_D)
    sem_oh_flat = sem_oh.reshape(n_sem, -1)
    np.add.at(sem_oh_flat, (row_idx, col_idx), observed_2d.ravel())
    # sem_oh already updated in-place via sem_oh_flat (shared memory)
    sem_dir += sem_oh


# ── initial belief ────────────────────────────────────────────────────────────

def init_belief_cnabu(n_sem: int = 17, prior: float = 1.0):
    """
    Returns:
      belief  : (2×H_BINS, W, D) float32  — uniform prior (all = prior)
      sem_dir : (N_SEM, W, D) float32     — uniform Dirichlet (all = prior)
    """
    belief  = np.full((2 * H_BINS, MAP_W, MAP_D), prior, dtype=np.float32)
    sem_dir = np.full((n_sem, MAP_W, MAP_D),       prior, dtype=np.float32)
    return belief, sem_dir
