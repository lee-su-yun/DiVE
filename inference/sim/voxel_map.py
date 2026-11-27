"""
Voxel map utilities for BeliefMap.

Shelf voxel space: 120(W) x 60(H) x 80(D) at 0.5cm/voxel
  voxel x (0~119): left-right   → world x
  voxel y (0~59):  up-down      → world z  (height, z-up)
  voxel z (0~79):  front-back   → world y  (depth)

Isaac Sim uses Z-up convention.
Shelf in world:
  x: env_origin.x - 0.30  ~  env_origin.x + 0.30
  y: env_origin.y - 0.20  ~  env_origin.y + 0.20  (depth; front=-0.20, back=+0.20)
  z: env_origin.z + 0.80  ~  env_origin.z + 1.10  (height; SHELF_Y=0.80)
"""

import torch
import numpy as np

# Shelf dimensions in voxels
VOXEL_W = 120  # x  (left-right)
VOXEL_H = 60   # y  (up-down / height)
VOXEL_D = 80   # z  (front-back / depth)
VOXEL_SIZE = 0.005  # 0.5 cm per voxel

# World offsets from env_origin to voxel (0,0,0) corner
# voxel (0,0,0) = left / bottom / front corner of shelf space
SHELF_X_OFFSET = -0.30   # world x  : left edge of shelf
SHELF_Z_OFFSET =  0.80   # world z  : shelf surface height (SHELF_Y)
SHELF_Y_OFFSET = -0.20   # world y  : front edge of shelf space


def world_to_voxel(pos_world: torch.Tensor, env_origins: torch.Tensor) -> torch.Tensor:
    """
    World positions → voxel indices (integer).
    pos_world:   (N, 3)  [x, y, z]
    env_origins: (N, 3)
    Returns:     (N, 3) int64  [vx, vy, vz], clamped to valid range
    """
    ox = env_origins[:, 0] + SHELF_X_OFFSET   # world x origin
    oy = env_origins[:, 1] + SHELF_Y_OFFSET   # world y origin (depth)
    oz = env_origins[:, 2] + SHELF_Z_OFFSET   # world z origin (height)

    vx = ((pos_world[:, 0] - ox) / VOXEL_SIZE).long()   # world x  → voxel x
    vy = ((pos_world[:, 2] - oz) / VOXEL_SIZE).long()   # world z  → voxel y (height)
    vz = ((pos_world[:, 1] - oy) / VOXEL_SIZE).long()   # world y  → voxel z (depth)

    voxel = torch.stack([vx, vy, vz], dim=1)
    voxel[:, 0] = voxel[:, 0].clamp(0, VOXEL_W - 1)
    voxel[:, 1] = voxel[:, 1].clamp(0, VOXEL_H - 1)
    voxel[:, 2] = voxel[:, 2].clamp(0, VOXEL_D - 1)
    return voxel




def make_swept_map_from_trajectory(ee_positions: torch.Tensor,
                                   env_origin: torch.Tensor) -> torch.Tensor:
    """
    Swept volume of the whole gripper body along an EE trajectory.

    Palm + rod geometry, measured along the push direction:
      ┌──────────────────────────────────────────────────────┐
      │  EE(tool0)                             rod tip(250mm)│
      │  ├── Palm (0~37mm):   width 90mm, height 40mm        │
      │  └── Rod  (37~250mm): width 30mm, height 30mm        │
      └──────────────────────────────────────────────────────┘

    The push direction (push_dir) and the lateral direction
    (push_dir ⊥ world_Z) are computed in world coordinates, the gripper volume
    is sampled in metric space along those axes, and the samples are then
    converted to voxels — so the far end of the swept map coincides with
    end_voxel (the rod tip).

    ee_positions: (T, 3) world [x, y, z]
    Returns: (W, H, D) bool tensor
    """
    GRIPPER_REACH  = 0.25    # m, EE → rod tip
    PALM_END       = 0.037   # m, end of the palm section from the EE (palm centre 12mm + half-z 25mm)
    PALM_HALF_LAT  = 0.045   # m, palm half-width  (90mm / 2)
    ROD_HALF_LAT   = 0.015   # m, rod half-width   (30mm / 2)
    PALM_HALF_HT   = 0.020   # m, palm half-height (40mm / 2)
    ROD_HALF_HT    = 0.015   # m, rod half-height  (30mm / 2)

    N_PUSH = 40   # samples along the push direction (EE → tip, one per ~6mm)
    N_LAT  = 7    # samples along the lateral direction
    N_HT   = 5    # samples along the height direction

    swept = torch.zeros(VOXEL_W, VOXEL_H, VOXEL_D, dtype=torch.bool)
    T     = ee_positions.shape[0]

    # ── Push / lateral / up directions in world space ───────────────────────
    if T > 1:
        raw  = ee_positions[-1] - ee_positions[0]
        norm = raw.norm()
        push_dir = raw / norm if norm > 1e-6 else torch.tensor([0., 1., 0.])
    else:
        push_dir = torch.tensor([0., 1., 0.])

    world_up = torch.tensor([0., 0., 1.])
    lat_dir  = torch.linalg.cross(push_dir, world_up)
    lat_norm = lat_dir.norm()
    lat_dir  = lat_dir / lat_norm if lat_norm > 1e-6 else torch.tensor([1., 0., 0.])

    # ── Gripper-volume sample offsets on a push × lateral × height lattice ──
    offsets = []
    for push_d in torch.linspace(0.0, GRIPPER_REACH, N_PUSH).tolist():
        if push_d <= PALM_END:
            half_lat = PALM_HALF_LAT
            half_ht  = PALM_HALF_HT
        else:
            half_lat = ROD_HALF_LAT
            half_ht  = ROD_HALF_HT
        for lat_v in torch.linspace(-half_lat, half_lat, N_LAT).tolist():
            for ht_v in torch.linspace(-half_ht, half_ht, N_HT).tolist():
                offsets.append(push_dir * push_d
                                + lat_dir  * lat_v
                                + world_up * ht_v)
    offsets = torch.stack(offsets)   # (N_offsets, 3)

    # ── All sample points: world coords → voxels, in one batched call ───────
    # ee_positions: (T, 3), offsets: (N, 3)  →  all_pos: (T*N, 3)
    all_pos = (ee_positions.unsqueeze(1)          # (T, 1, 3)
               + offsets.unsqueeze(0)             # (1, N, 3)
               ).reshape(-1, 3)                   # (T*N, 3)

    orig   = env_origin.unsqueeze(0).expand(all_pos.shape[0], -1)
    voxels = world_to_voxel(all_pos.float(), orig)   # (T*N, 3)

    # ── Mark the voxels (batched indexing) ──────────────────────────────────
    swept[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = True

    return swept
