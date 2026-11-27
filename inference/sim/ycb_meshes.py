"""
YCB mesh utilities — load USD meshes, voxelize once, cache sample points.
Used by the scene modules in inference/sim/ to build a true mesh-shaped GT
semantic_map (rather than a bbox/OBB approximation).

API:
  load_type_samples(type_id) -> np.ndarray (M, 3) float32
      USD local frame (no standing rotation applied) sample points
      at VOXEL_SIZE pitch covering the *filled* mesh volume.

  voxelize_scene(grid, type_ids, positions, quats_wxyz, env_origin)
      Stamp each (type_id, world pose) onto a (W, H, D) int32 grid in-place.
"""
from __future__ import annotations
import os
import sys
import numpy as np

VOXEL_W, VOXEL_H, VOXEL_D = 120, 60, 80
VOXEL_SIZE = 0.005
SHELF_X_OFFSET = -0.30
SHELF_Z_OFFSET =  0.80
SHELF_Y_OFFSET = -0.20

CACHE_DIR = os.path.expanduser("~/.cache/beliefmap_ycb_samples")


# ── USD mesh extraction (Isaac/Omniverse env required) ─────────────────────

def _extract_usd_mesh(usd_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Open USD, walk every Mesh prim, bake local xform → return (V, F).

    Returns:
      verts: (Nv, 3) float64 in USD root frame
      faces: (Nf, 3) int32 triangulated
    """
    from pxr import Usd, UsdGeom, Gf

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {usd_path}")

    default = stage.GetDefaultPrim() or stage.GetPseudoRoot()

    all_v = []
    all_f = []
    v_offset = 0
    for prim in Usd.PrimRange(default):
        if prim.GetTypeName() != "Mesh":
            continue
        mesh = UsdGeom.Mesh(prim)
        pts_attr = mesh.GetPointsAttr().Get()
        fc_attr  = mesh.GetFaceVertexCountsAttr().Get()
        fi_attr  = mesh.GetFaceVertexIndicesAttr().Get()
        if pts_attr is None or fc_attr is None or fi_attr is None:
            continue

        local_pts = np.asarray(pts_attr, dtype=np.float64)   # (Np, 3)
        # Bake the prim's local-to-stage transform onto verts
        xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
        # Manual matrix-vec: Gf.Matrix4d row-major; verts stored as column 3-vec
        m = np.array([[xf[r][c] for c in range(4)] for r in range(4)], dtype=np.float64)
        # USD convention: row-major, points are row vectors → world_v = (v|1) @ M
        homo = np.hstack([local_pts, np.ones((local_pts.shape[0], 1))])
        world_pts = (homo @ m)[:, :3]
        all_v.append(world_pts)

        counts  = np.asarray(fc_attr, dtype=np.int32)
        indices = np.asarray(fi_attr, dtype=np.int32)
        # Fan-triangulate every polygon
        cursor = 0
        for c in counts:
            for k in range(1, c - 1):
                all_f.append([indices[cursor],
                              indices[cursor + k],
                              indices[cursor + k + 1]])
            cursor += c
        # offset face indices by accumulated vertex count
        nfaces_added = sum(c - 2 for c in counts if c >= 3)
        if nfaces_added:
            tail = np.asarray(all_f[-nfaces_added:], dtype=np.int64)
            tail += v_offset
            all_f[-nfaces_added:] = tail.tolist()
        v_offset += local_pts.shape[0]

    if not all_v:
        raise RuntimeError(f"No mesh found in {usd_path}")
    verts = np.concatenate(all_v, axis=0).astype(np.float64)
    faces = np.asarray(all_f, dtype=np.int64)
    return verts, faces


def _voxelize_mesh(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Trimesh voxelize at VOXEL_SIZE/2, then 3D hole-fill on the voxel grid.

    Sub-voxel pitch (VOXEL_SIZE/2): with a sample pitch equal to the voxel
    pitch, rotating the object misaligns the sample lattice with the voxel
    grid and individual voxels end up with no sample in them. Halving the
    pitch removes the gaps.

    binary_fill_holes: the bottle-shaped USD meshes (bleach_cleanser,
    mustard_bottle, …) are not watertight, so trimesh.voxelized().fill()
    leaves their interior empty. Running a 3D flood fill directly on the
    voxel matrix fills every enclosed cavity.

    Returns: (M, 3) float32 sample points (USD local frame).
    """
    import trimesh
    from scipy.ndimage import binary_fill_holes
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    if not mesh.is_watertight:
        mesh.fill_holes()
    vg = mesh.voxelized(pitch=VOXEL_SIZE / 2.0)
    try:
        vg = vg.fill()
    except Exception:
        pass
    # 3D flood-fill any holes the mesh-level fill missed
    matrix = np.asarray(vg.matrix, dtype=bool)
    matrix = binary_fill_holes(matrix)
    indices = np.argwhere(matrix)                                     # (M, 3) int
    # voxel index → world (mesh local) coord:  origin + (idx + 0.5) * pitch
    origin = np.asarray(vg.origin, dtype=np.float32)
    pts = (indices.astype(np.float32) + 0.5) * float(vg.pitch) + origin
    return pts.astype(np.float32)


# ── Public API ──────────────────────────────────────────────────────────────

_TYPE_SAMPLES_CACHE: dict[int, np.ndarray] = {}


def _cache_path(type_id: int, usd_name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{type_id:02d}_{usd_name}.npy")


def load_type_samples(type_id: int, usd_name: str, usd_full_path: str) -> np.ndarray:
    """Return (M, 3) float32 sample points for this YCB type, USD local frame.

    First call: extract USD → trimesh voxelize → save to disk.
    Subsequent calls: load from disk cache.
    """
    if type_id in _TYPE_SAMPLES_CACHE:
        return _TYPE_SAMPLES_CACHE[type_id]
    cp = _cache_path(type_id, usd_name)
    if os.path.isfile(cp):
        pts = np.load(cp)
    else:
        verts, faces = _extract_usd_mesh(usd_full_path)
        pts = _voxelize_mesh(verts, faces)
        np.save(cp, pts)
        print(f"[ycb_meshes] cached {type_id:>2} {usd_name:>22}  "
              f"M={len(pts):>5}  → {cp}")
    _TYPE_SAMPLES_CACHE[type_id] = pts
    return pts


def precache_all_types(types_iter):
    """Force eager load+cache for every (type_id, usd_name, usd_full_path)."""
    for type_id, usd_name, usd_full_path in types_iter:
        load_type_samples(type_id, usd_name, usd_full_path)


# ── Voxelization for GT semantic map ────────────────────────────────────────

def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),    2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),    1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def voxelize_scene(active_type_ids: list,
                   positions: np.ndarray,        # (N, 3) world xyz
                   quats_wxyz: np.ndarray,       # (N, 4)
                   env_origin: np.ndarray,       # (3,)
                   shape: tuple = (VOXEL_W, VOXEL_H, VOXEL_D),
                   scales_local: np.ndarray = None,  # (N, 3) per-instance local-frame scale
                   ) -> np.ndarray:
    """
    Build a (W, H, D) int32 grid from per-object mesh sample points.
    Voxel axis: 0 = world x, 1 = world z (height), 2 = world y (depth).
    Later objects do NOT overwrite earlier (first-come label keeps).

    scales_local: optional (N, 3) per-instance scale applied to mesh sample
        points in the USD local frame BEFORE the standing+yaw rotation.
        None or all-ones → unscaled (default behavior).
    """
    grid = np.zeros(shape, dtype=np.int32)
    ox = env_origin[0] + SHELF_X_OFFSET
    oy = env_origin[1] + SHELF_Y_OFFSET
    oz = env_origin[2] + SHELF_Z_OFFSET

    for i, (tid, pos, quat) in enumerate(zip(active_type_ids, positions, quats_wxyz)):
        pts = _TYPE_SAMPLES_CACHE.get(int(tid))
        if pts is None:
            continue
        R = quat_wxyz_to_rotmat(quat.astype(np.float64))
        if scales_local is not None:
            s = np.asarray(scales_local[i], dtype=np.float64).reshape(1, 3)
            world_pts = (pts * s) @ R.T + pos.reshape(1, 3)
        else:
            world_pts = pts @ R.T + pos.reshape(1, 3)
        vx = ((world_pts[:, 0] - ox) / VOXEL_SIZE).astype(np.int32)
        vy = ((world_pts[:, 2] - oz) / VOXEL_SIZE).astype(np.int32)
        vz = ((world_pts[:, 1] - oy) / VOXEL_SIZE).astype(np.int32)
        ok = ((vx >= 0) & (vx < shape[0]) &
              (vy >= 0) & (vy < shape[1]) &
              (vz >= 0) & (vz < shape[2]))
        vx, vy, vz = vx[ok], vy[ok], vz[ok]
        # first-come keeps: only mark where empty
        empty = grid[vx, vy, vz] == 0
        grid[vx[empty], vy[empty], vz[empty]] = int(tid)
    return grid
