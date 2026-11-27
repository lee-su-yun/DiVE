"""
BeliefMap high-occlusion YCB scene / data-collection script.

Objects are real YCB mesh USDs from the NVIDIA Isaac assets rather than
primitive geometry.

Objects used (Axis_Aligned, 11 types):
  Tall (h>=13cm): 002_master_chef_can, 003_cracker_box, 004_sugar_box,
                  006_mustard_bottle, 021_bleach_cleanser
  Small (h<13cm): 005_tomato_soup_can, 007_tuna_fish_can, 008_pudding_box,
                  009_gelatin_box, 010_potted_meat_can, 061_foam_brick

Scene: N_ACTIVE = 11 objects, one instance of every type (N_DUP = 0).
  - Tall: all 5 tall types, packed into the front zone of the shelf (up to
    22 cm from the entrance), so they hide what is behind them (hence
    "high occlusion").
  - Small: all 6 small types, placed in the back zone behind the tall row.
  - Optional --shape_ood: each active object's two non-up local axes are
    scaled by a random factor in [--scale_low, --scale_high] (height kept).

Spawning (`_spawn_ycb`):
  - The Axis_Aligned USDs carry no physics, so RigidBodyAPI / MassAPI /
    CollisionAPI (convexHull approximation) are applied by hand.
  - The mesh long axis is stood up along world Z: the bottom face is the one
    with the smallest area, and cylinders end up with their circular faces
    up and down.
  - 006_mustard_bottle and 021_bleach_cleanser have their cap on USD local -Y,
    so they use `_Q_NEG_Y_UP` to end up cap-up.
  - Textures are dropped in favour of a distinct flat colour per obj_N index
    (golden-ratio hue spacing).

On episode reset the random yaw is composed with the standing rotation so that
objects only ever rotate about the vertical axis, and the placement bounds use
yaw-aware half-extents so nothing sticks out of the shelf at any yaw.

Push action (frontier-based, angled):
  - 1 episode = N_SUBSCENES = 5 subscenes (A..E) with 4 transitions.
  - At the start of each subscene, frontier voxels are sampled from that
    subscene's PRE ground truth. Frontier = occupied voxels within
    FRONTIER_BASE_CM = 15 cm of the shelf entrance, widened in 5 cm steps if
    too few columns are found. SUBSCENE_LEN = 20 unique x voxels are sampled,
    giving 20 pushes per subscene.
  - Each push targets the occupied voxel closest to the entrance in that x
    column (the frontier line). The angle is uniform in +-theta_max, with
    theta_max = min(d_cm, ANGLE_MAX_DEG) and d = distance to the nearer x edge.
    A negative angle pushes left-to-right (push direction has a +X component).
  - One random step per subscene is picked as the transition: its post state
    becomes the next subscene's PRE.
  - episode_log.txt records the push specs and transition step of every
    subscene.

--test_set_mode instead saves PRE-only scenes, keeping a scene only if at least
one active object is hidden from all 98 views.

Usage:
    PYTHONPATH=$PWD python -m inference.sim.scene_high_occlusion \
        --num_envs 16 --num_episodes 13920 \
        --save_path data/beliefmap_high_occlusion_ycb_v1
"""

import argparse
import sys
import os
import math
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="BeliefMap High-Occlusion Data Collection")
parser.add_argument("--num_envs",       type=int, default=4)
parser.add_argument("--num_episodes",   type=int, default=100)
parser.add_argument("--save_path",      type=str, default="data/beliefmap_high_occlusion_v2")
parser.add_argument("--start_episode",  type=int, default=0)
parser.add_argument("--test_set_mode",  action="store_true",
                    help="Generate a reusable test set: PRE-only capture per scenario with "
                         "rejection sampling (>=1 active slot must be hidden from all 98 views).")
parser.add_argument("--test_set_target", type=int, default=100,
                    help="Number of accepted scenarios to save in --test_set_mode.")
parser.add_argument("--test_set_seed",   type=int, default=42,
                    help="Master RNG seed for --test_set_mode (deterministic test sets).")
parser.add_argument("--test_set_max_tries", type=int, default=10000,
                    help="Hard cap on randomize iterations before aborting (safety).")
parser.add_argument("--shape_ood",  action="store_true",
                    help="Shape OOD: scale each active object's two non-up local axes "
                         "by a random factor in [--scale_low, --scale_high]. Height axis = 1.0.")
parser.add_argument("--scale_low",  type=float, default=0.8)
parser.add_argument("--scale_high", type=float, default=1.2)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

import sys as _sys
_sys.argv += [
    "--/rtx/post/dlss/enabled=false",
    "--/rtx/post/aa/op=0",
    "--/app/asyncRendering=false",
    "--/app/asyncRenderingLowLatency=false",
]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import torch
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat, quat_inv
from isaaclab.utils import configclass

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.utils import clone as _clone, create_prim, get_current_stage
from isaaclab.sim import schemas as _schemas

from inference.sim.scene_cfg import (BeliefMapSceneCfgNoObj,
                       SHELF_Y, SHELF_H, SHELF_W, SHELF_D)
from inference.sim.voxel_map import (
    VOXEL_W, VOXEL_H, VOXEL_D, VOXEL_SIZE,
    SHELF_X_OFFSET, SHELF_Y_OFFSET, SHELF_Z_OFFSET,
    world_to_voxel,
    make_swept_map_from_trajectory,
)
from inference.sim.ycb_types import (
    YCB_TYPES, N_TYPES, N_SLOTS, SLOT_NAMES, SLOT_TO_TYPE_ID,
    TYPE_BY_ID, USD_BY_ID, ID_BY_USD, HALF_LOCAL, HALF_WORLD,
    ROT_BY_ID, MASS_BY_ID, SHAPE_BY_ID, TALL_IDS, SMALL_IDS,
    TYPE_STANDING_VY, slot_init_pos,
)
from inference.sim import ycb_meshes
from inference.utils.voxel_projection import depth_to_height_map, VOXEL_H as _VOX_H
try:
    from cnabu_reproduce.data.dataset_orig_format import _hms_to_occ_free, _to_their_gt
except ImportError:
    def _to_their_gt(gt_id_np):
        return (gt_id_np > 0).astype(np.float32).transpose(1, 0, 2)

    def _hms_to_occ_free(hms_stack):
        """Surface-only fallback (v1-compatible).
          occupied = observed & (h == hm)
          free     = observed & (h >  hm)
        """
        n = hms_stack.shape[0]
        H = _VOX_H
        h_idx = np.arange(H, dtype=np.int16).reshape(H, 1, 1)
        occ_list, free_list = [], []
        for i in range(n):
            hm = hms_stack[i].astype(np.int16)
            observed = (hm >= 0)[None, :, :]
            hm_exp   = hm[None, :, :]
            occ      = observed & (h_idx == hm_exp)
            free     = observed & (h_idx >  hm_exp)
            occ_list.append(occ.astype(np.float32))
            free_list.append(free.astype(np.float32))
        return np.stack(occ_list), np.stack(free_list)

# ── Camera grid ───────────────────────────────────────────────────────────────
N_COLS  = 14
N_ROWS  = 7
N_VIEWS = N_COLS * N_ROWS   # 98
CAM_DIST = 0.50

# ── Push / subscene definitions ───────────────────────────────────────────
# One episode = N_SUBSCENES subscenes x SUBSCENE_LEN pushes.
# At the start of each subscene, frontier voxels are sampled from that
# subscene's PRE ground truth. Frontier = occupied voxels within
# FRONTIER_BASE_CM of the shelf entrance; if fewer than SUBSCENE_LEN columns
# qualify, the slab is widened by FRONTIER_EXPAND_CM and retried.
# One random step per subscene becomes the transition → the next subscene's PRE.
SUBSCENE_LEN       = 20
N_SUBSCENES        = 5                 # A, B, C, D, E
N_TRANSITIONS      = N_SUBSCENES - 1   # 4
TOTAL_STEPS        = SUBSCENE_LEN * N_SUBSCENES   # 100
SUBSCENE_LABELS    = ['A', 'B', 'C', 'D', 'E']

FRONTIER_BASE_CM   = 15.0
FRONTIER_EXPAND_CM = 5.0
FRONTIER_MAX_CM    = 35.0
PUSH_BEYOND_M      = 0.15  # extra distance pushed past the frontier surface (m)
ANGLE_MAX_DEG      = 30.0  # +-30 deg at the centre (30cm from an edge), 0 deg at the edge

# ── High-occlusion YCB scene config (N_ACTIVE = 11 objects) ─────────────────
# Tall  (5): 002, 003, 004, 006, 021 — one instance each, in the front zone.
# Small (6): 005, 007, 008, 009, 010, 061 — one instance each, in the back zone.
#
# Physical YCB dimensions (Axis_Aligned USD, metres, USD local frame):
#   002_master_chef_can : cyl  r=0.0425 h=0.139  (long axis Y)
#   003_cracker_box     : box  0.160×0.060×0.210 (long axis Y)
#   004_sugar_box       : box  0.095×0.045×0.175 (long axis Y)
#   005_tomato_soup_can : cyl  r=0.0335 h=0.102  (long axis Y)
#   006_mustard_bottle  : box  0.058×0.042×0.191 (long axis Y, cap at -Y)
#   007_tuna_fish_can   : cyl  r=0.0425 h=0.033  (cyl axis Y)
#   008_pudding_box     : box  0.086×0.037×0.077 (long axis X)
#   009_gelatin_box     : box  0.075×0.028×0.110 (long axis X)
#   010_potted_meat_can : box  0.097×0.064×0.084 (long axis X)
#   021_bleach_cleanser : cyl  r=0.042  h=0.252  (long axis Y, cap at -Y)
#   061_foam_brick      : box  0.076×0.051×0.050 (long axis X)

@_clone
def _spawn_ycb(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Spawn a YCB Axis_Aligned USD with physics APIs applied manually.

    Isaac Lab's spawn_from_usd calls modify_rigid_body_properties which only modifies
    *existing* RigidBodyAPI and silently does nothing when the USD has no physics
    (like Axis_Aligned). This function uses define_* to apply RigidBodyAPI to the
    Xform root and CollisionAPI + convexHull to every Mesh in the subtree. It also
    replaces the original textured material with a per-object flat color, since we
    don't need photorealism for belief-map data collection.
    """
    import re
    import colorsys
    from pxr import UsdPhysics, UsdShade, Usd, Sdf, Gf

    stage = get_current_stage()

    if not stage.GetPrimAtPath(prim_path).IsValid():
        create_prim(
            prim_path,
            usd_path=cfg.usd_path,
            translation=translation,
            orientation=orientation,
            scale=cfg.scale,
            stage=stage,
        )

    # Apply RigidBodyAPI + PhysX rigid body properties to root Xform
    if cfg.rigid_props is not None:
        _schemas.define_rigid_body_properties(prim_path, cfg.rigid_props)
    else:
        root_prim = stage.GetPrimAtPath(prim_path)
        if not UsdPhysics.RigidBodyAPI(root_prim):
            UsdPhysics.RigidBodyAPI.Apply(root_prim)

    # Apply MassAPI
    if cfg.mass_props is not None:
        _schemas.define_mass_properties(prim_path, cfg.mass_props)

    # Per-object distinct color from obj_N index (golden-ratio hue spacing)
    m = re.search(r"obj_(\d+)", prim_path)
    idx = int(m.group(1)) if m else 0
    h = (idx * 0.6180339887498949) % 1.0
    color = colorsys.hsv_to_rgb(h, 0.85, 0.95)

    # Create a UsdPreviewSurface material with our flat color, child of the spawn prim
    mat_path = prim_path + "/PreviewMat"
    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    # Apply CollisionAPI + convex hull and rebind every Mesh to the flat-color material
    root_prim = stage.GetPrimAtPath(prim_path)
    for child in Usd.PrimRange(root_prim):
        if child.GetTypeName() == "Mesh":
            if not UsdPhysics.CollisionAPI(child):
                UsdPhysics.CollisionAPI.Apply(child)
            mesh_col = UsdPhysics.MeshCollisionAPI(child)
            if not mesh_col:
                mesh_col = UsdPhysics.MeshCollisionAPI.Apply(child)
            mesh_col.CreateApproximationAttr("convexHull")

            # Replace original textured material with the flat-color material
            mat_rel = child.GetRelationship("material:binding")
            if mat_rel and mat_rel.IsValid():
                mat_rel.SetTargets([])
            UsdShade.MaterialBindingAPI.Apply(child).Bind(material)

    return stage.GetPrimAtPath(prim_path)


# ── Standing rotation quaternions (w, x, y, z) ──────────────────────────────
# The YCB Axis_Aligned USDs use stage upAxis=Z, but the mesh long axis lies on
# Y (or X for a few), so spawning them as-is leaves them lying on their side.
# These quaternions stand the long axis up along world +Z.
#   _Q_Y_UP    : local +Y → world +Z (+90 deg about X). Body on +Y.
#   _Q_NEG_Y_UP: local -Y → world +Z (-90 deg about X). The bottle types whose
#                cap is on -Y (006_mustard_bottle, 021_bleach_cleanser), so
#                that the cap ends up on top.
#   _Q_X_UP    : local +X → world +Z (-90 deg about Y). Boxes whose longest
#                axis is X.
_Q_Y_UP     = (0.7071067811865476,  0.7071067811865476, 0.0, 0.0)
_Q_NEG_Y_UP = (0.7071067811865476, -0.7071067811865476, 0.0, 0.0)
_Q_X_UP     = (0.7071067811865476,  0.0, -0.7071067811865476, 0.0)


def _ycb_cfg(prim_name, usd_name, mass, init_pos, rot):
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/{usd_name}.usd",
            func=_spawn_ycb,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, solver_position_iteration_count=4),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=rot),
    )


# 22 slots = 11 types x 2 instances. The SceneCfg attribute names are
# SLOT_NAMES (obj_0..obj_21). Every slot spawns at its off-shelf default
# position (slot_init_pos); on each episode reset randomize_object_positions
# picks the active slots, moves only those to random positions inside the
# shelf, and leaves the rest at their default (off-shelf) position.
def _build_scene_cfg_class():
    """Dynamically build a configclass with 22 slots.

    Each slot 0..21 has a fixed type (slot 2k, 2k+1 = type k+1).
    """
    body = {}
    for slot_idx, slot_name in enumerate(SLOT_NAMES):
        type_id = SLOT_TO_TYPE_ID[slot_idx]
        rec     = TYPE_BY_ID[type_id]
        body[slot_name] = _ycb_cfg(slot_name, rec["usd"], rec["mass"],
                                   slot_init_pos(slot_idx), rec["rot"])
    return configclass(type("HighOcclusionYCBSceneCfg",
                            (BeliefMapSceneCfgNoObj,), body))


HighOcclusionYCBSceneCfg = _build_scene_cfg_class()


# ── Episode active selection ────────────────────────────────────────────────
# Each episode uses the 11 unique types only (5 tall + 6 small), no duplicates (leaves room for placement).
N_ACTIVE      = 11
N_UNIQUE      = N_TYPES                        # 11
N_DUP         = 0

OBJ_NAMES   = SLOT_NAMES                      # all 22 slots (collection-time)

# Slot lookup helpers — slot index → type properties (single source: ycb_types.py).
def slot_type_id(slot_name):    return SLOT_TO_TYPE_ID[SLOT_NAMES.index(slot_name)]
def slot_half_local(slot_name): return HALF_LOCAL[slot_type_id(slot_name)]
def slot_half_world(slot_name): return HALF_WORLD[slot_type_id(slot_name)]
def slot_shape(slot_name):      return SHAPE_BY_ID[slot_type_id(slot_name)]
def slot_is_tall(slot_name):    return TYPE_BY_ID[slot_type_id(slot_name)]["is_tall"]


# Episode-scoped active selection.  Set/cleared by randomize_object_positions().
# active_slots[e] = list of N_ACTIVE slot names placed inside the shelf for env e.
# Other slots are kept at slot_init_pos() (off-shelf, effectively hidden).
_active_slots: list = []   # length = num_envs
_active_scales: list = []  # length = num_envs; each entry dict[slot_name]=(sx,sy,sz) local-frame


def get_active_slots(env_idx: int):
    """Return slot names placed inside the shelf for env e (after randomize)."""
    return _active_slots[env_idx] if _active_slots else []


def get_active_scales(env_idx: int):
    if not _active_scales or env_idx >= len(_active_scales) or _active_scales[env_idx] is None:
        return {}
    return _active_scales[env_idx]


# ── Shape OOD scale helpers ─────────────────────────────────────────────────
def _sample_scale_local(type_id, low, high):
    """Sample per-axis local scale: two non-up axes ~ U[low,high], up axis = 1."""
    rot = ROT_BY_ID[type_id]
    s0 = random.uniform(low, high)
    s1 = random.uniform(low, high)
    if rot == _Q_X_UP:                 # local +X is up
        return (1.0, s0, s1)
    else:                              # _Q_Y_UP or _Q_NEG_Y_UP: local Y is up
        return (s0, 1.0, s1)


def _abs_rotmat_from_quat(q):
    from inference.sim.ycb_meshes import quat_wxyz_to_rotmat
    return np.abs(quat_wxyz_to_rotmat(np.asarray(q, dtype=np.float64)))


def _half_world_for(slot_name, scale_local=(1.0, 1.0, 1.0)):
    """World-frame yaw=0 OBB half-extents after applying local scale + standing rot."""
    tid = slot_type_id(slot_name)
    hl = np.asarray(HALF_LOCAL[tid], dtype=np.float64) * np.asarray(scale_local, dtype=np.float64)
    Rabs = _abs_rotmat_from_quat(ROT_BY_ID[tid])
    return tuple((Rabs @ hl).tolist())


def _apply_xform_scale(stage, env_idx, slot_name, scale_local):
    from pxr import UsdGeom, Gf
    prim = stage.GetPrimAtPath(f"/World/envs/env_{env_idx}/{slot_name}")
    if not prim.IsValid():
        return
    xf = UsdGeom.Xformable(prim)
    sx, sy, sz = (float(scale_local[0]), float(scale_local[1]), float(scale_local[2]))
    for op in xf.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:scale":
            op.Set(Gf.Vec3d(sx, sy, sz))
            return
    xf.AddScaleOp().Set(Gf.Vec3d(sx, sy, sz))


# ── Camera / capture ──────────────────────────────────────────────────────────

def look_at_quaternion(cam_pos: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    device = cam_pos.device
    N = cam_pos.shape[0]
    fwd = torch.nn.functional.normalize(target - cam_pos, dim=1)
    up_hint = torch.zeros(N, 3, device=device); up_hint[:, 2] = 1.0
    right = torch.nn.functional.normalize(torch.cross(fwd, up_hint, dim=1), dim=1)
    up    = torch.nn.functional.normalize(torch.cross(fwd, right, dim=1), dim=1)
    R     = torch.stack([right, up, fwd], dim=2)
    quat  = torch.zeros(N, 4, device=device)
    t     = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    mA = t > 0
    if mA.any():
        s = 0.5 / torch.sqrt((t[mA] + 1.0).clamp(min=1e-10))
        quat[mA, 0] = 0.25 / s
        quat[mA, 1] = (R[mA, 2, 1] - R[mA, 1, 2]) * s
        quat[mA, 2] = (R[mA, 0, 2] - R[mA, 2, 0]) * s
        quat[mA, 3] = (R[mA, 1, 0] - R[mA, 0, 1]) * s
    mB = (~mA) & (R[:, 0, 0] >= R[:, 1, 1]) & (R[:, 0, 0] >= R[:, 2, 2])
    if mB.any():
        s = 2.0 * torch.sqrt((1.0 + R[mB,0,0] - R[mB,1,1] - R[mB,2,2]).clamp(min=1e-10))
        quat[mB, 0] = (R[mB, 2, 1] - R[mB, 1, 2]) / s; quat[mB, 1] = 0.25 * s
        quat[mB, 2] = (R[mB, 0, 1] + R[mB, 1, 0]) / s; quat[mB, 3] = (R[mB, 0, 2] + R[mB, 2, 0]) / s
    mC = (~mA) & (~mB) & (R[:, 1, 1] >= R[:, 2, 2])
    if mC.any():
        s = 2.0 * torch.sqrt((1.0 + R[mC,1,1] - R[mC,0,0] - R[mC,2,2]).clamp(min=1e-10))
        quat[mC, 0] = (R[mC, 0, 2] - R[mC, 2, 0]) / s; quat[mC, 1] = (R[mC, 0, 1] + R[mC, 1, 0]) / s
        quat[mC, 2] = 0.25 * s; quat[mC, 3] = (R[mC, 1, 2] + R[mC, 2, 1]) / s
    mD = (~mA) & (~mB) & (~mC)
    if mD.any():
        s = 2.0 * torch.sqrt((1.0 + R[mD,2,2] - R[mD,0,0] - R[mD,1,1]).clamp(min=1e-10))
        quat[mD, 0] = (R[mD, 1, 0] - R[mD, 0, 1]) / s; quat[mD, 1] = (R[mD, 0, 2] + R[mD, 2, 0]) / s
        quat[mD, 2] = (R[mD, 1, 2] + R[mD, 2, 1]) / s; quat[mD, 3] = 0.25 * s
    return torch.nn.functional.normalize(quat, dim=1)


def get_viewpoint_positions(env_origins: torch.Tensor) -> torch.Tensor:
    device = env_origins.device
    xs = torch.linspace(-SHELF_W / 2, SHELF_W / 2, N_COLS, device=device)
    zs = torch.linspace(SHELF_Y + 0.03, SHELF_Y + SHELF_H - 0.03, N_ROWS, device=device)
    positions = []
    for zi in range(N_ROWS):
        for xi in range(N_COLS):
            x = env_origins[:, 0] + xs[xi]
            y = env_origins[:, 1] - SHELF_D / 2 - CAM_DIST
            z = env_origins[:, 2] + zs[zi]
            positions.append(torch.stack([x, y, z], dim=1))
    return torch.stack(positions, dim=1)


def capture_all_viewpoints(scene, sim, sim_cfg, home_jpos, arm_joint_ids):
    cam         = scene["camera"]
    robot       = scene["robot"]
    env_origins = scene.env_origins
    view_pos    = get_viewpoint_positions(env_origins)
    shelf_center = env_origins.clone()
    shelf_center[:, 2] += SHELF_Y + SHELF_H / 2
    all_depth, all_quats, all_rgb, all_sem = [], [], [], []
    for v in range(N_VIEWS):
        pos_v  = view_pos[:, v, :]
        quat_v = look_at_quaternion(pos_v, shelf_center)
        cam.set_world_poses(positions=pos_v, orientations=quat_v)
        robot.set_joint_position_target(home_jpos, joint_ids=arm_joint_ids)
        scene.write_data_to_sim(); sim.step(); scene.update(sim_cfg.dt)
        raw_d = cam.data.output["distance_to_camera"]
        if raw_d.ndim == 4: raw_d = raw_d[..., 0]
        depth_v = raw_d.float().cpu()
        depth_v[depth_v > 2.0] = float('nan')
        rgb_v = cam.data.output["rgb"][..., :3].byte().cpu()          # (E, H, W, 3)
        sem_v = cam.data.output["instance_id_segmentation_fast"].cpu() # (E, H, W) int32 instance ids
        # Accumulate idToLabels over every view (each view only lists what it sees; the union covers all)
        build_sem_id_map(cam, verbose=False)
        all_depth.append(depth_v)
        all_quats.append(quat_v.cpu())
        all_rgb.append(rgb_v)
        all_sem.append(sem_v)
    return (torch.stack(all_depth, dim=1),   # (E, N_VIEWS, H, W)
            view_pos.cpu(),                   # (E, N_VIEWS, 3)
            torch.stack(all_quats, dim=1),    # (E, N_VIEWS, 4)
            torch.stack(all_rgb,   dim=1),    # (E, N_VIEWS, H, W, 3)
            torch.stack(all_sem,   dim=1))    # (E, N_VIEWS, H, W[, 4])


def _depth_view_to_voxel_occ(depth_m, cam_pos, cam_quat_wxyz, env_origin):
    """Mark only the first surface voxel each camera ray hits in one view's depth.
    Every valid pixel is back-projected to 3D and its voxel index set to occupied.
    Returns: (H, W, D) = (60, 120, 80) float32, axis-1 = up.
    """
    from scipy.spatial.transform import Rotation
    # camera intrinsics — assumes the same PinholeCameraCfg for every view
    H_img, W_img = depth_m.shape
    FX = FY = 24.0 / 20.955 * W_img
    CX = CY = W_img / 2.0

    us = np.arange(W_img, dtype=np.float32)
    vs = np.arange(H_img, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)

    valid = np.isfinite(depth_m) & (depth_m > 0.01)
    d = np.where(valid, depth_m, 0.0)

    # radial depth → camera-space z
    x_norm = (uu - CX) / FX
    y_norm = (vv - CY) / FY
    z = d / np.sqrt(1.0 + x_norm ** 2 + y_norm ** 2)
    p_cam = np.stack([x_norm * z, y_norm * z, z], axis=-1).reshape(-1, 3)

    # camera → world (Isaac Y-up cam, wxyz quat)
    q_xyzw = np.asarray(cam_quat_wxyz)[[1, 2, 3, 0]]
    R = Rotation.from_quat(q_xyzw).as_matrix()
    p_world = (R @ p_cam.T).T + np.asarray(cam_pos)[None, :]

    ox = env_origin[0] + SHELF_X_OFFSET
    oy = env_origin[1] + SHELF_Y_OFFSET
    oz = env_origin[2] + SHELF_Z_OFFSET

    vx = ((p_world[:, 0] - ox) / VOXEL_SIZE).astype(np.int32)
    vy = ((p_world[:, 2] - oz) / VOXEL_SIZE).astype(np.int32)   # world z → voxel y (height)
    vz = ((p_world[:, 1] - oy) / VOXEL_SIZE).astype(np.int32)   # world y → voxel z (depth)

    valid_flat = valid.reshape(-1)
    in_range = (valid_flat &
                (vx >= 0) & (vx < VOXEL_W) &
                (vy >= 0) & (vy < VOXEL_H) &
                (vz >= 0) & (vz < VOXEL_D))

    occ = np.zeros((VOXEL_H, VOXEL_W, VOXEL_D), dtype=np.float32)
    occ[vy[in_range], vx[in_range], vz[in_range]] = 1.0
    return occ


def depth_stack_to_occ(depth_tensor, cam_pos_tensor, cam_quat_tensor, env_origin_np):
    """98 view per-view occupancy stack.
    For each view, only the voxels where a camera ray first hits a surface are 1.
    Returns: (N_VIEWS, H, W, D) float32.
    """
    depth_np    = depth_tensor.numpy()
    cam_pos_np  = cam_pos_tensor.numpy()
    cam_quat_np = cam_quat_tensor.numpy()
    return np.stack([
        _depth_view_to_voxel_occ(depth_np[i], cam_pos_np[i], cam_quat_np[i], env_origin_np)
        for i in range(N_VIEWS)
    ])

def _save_semantic_map(path, gt_id_map):
    if isinstance(gt_id_map, torch.Tensor):
        sem_np = gt_id_map.cpu().numpy()
    else:
        sem_np = gt_id_map
    np.savez_compressed(path, semantic_map=sem_np.astype(np.int32))

# ── GT map ───────────────────────────────────────────────────────────────────

def get_gt_id_map(scene, env_idx: int) -> torch.Tensor:
    """Mesh-based voxelization — each YCB type contributes its real shape.

    Active slots only (others sit off-shelf and would be out of voxel range).
    Same-type duplicates → same semantic id (1..11).
    """
    env_origin_np = scene.env_origins[env_idx].cpu().numpy().astype(np.float64)
    scales_map = get_active_scales(env_idx)
    type_ids, positions, quats, scales = [], [], [], []
    for name in get_active_slots(env_idx):
        if name not in scene._rigid_objects:
            continue
        type_ids.append(slot_type_id(name))
        positions.append(scene[name].data.root_pos_w[env_idx].cpu().numpy())
        quats.append(scene[name].data.root_quat_w[env_idx].cpu().numpy())
        scales.append(scales_map.get(name, (1.0, 1.0, 1.0)))
    if not type_ids:
        return torch.zeros(VOXEL_W, VOXEL_H, VOXEL_D, dtype=torch.int32)
    scales_arr = np.asarray(scales, dtype=np.float64)
    use_scales = scales_arr if not np.allclose(scales_arr, 1.0) else None
    grid = ycb_meshes.voxelize_scene(
        active_type_ids=type_ids,
        positions=np.asarray(positions, dtype=np.float64),
        quats_wxyz=np.asarray(quats, dtype=np.float64),
        env_origin=env_origin_np,
        scales_local=use_scales,
    )
    return torch.from_numpy(grid.astype(np.int32))


# ── Push action ──────────────────────────────────────────────────────────────

PUSH_Z_OFFSET  = 0.09   # shelf floor + 9cm (gripper height during push) — passes over the top of small objects,
                         # 1cm above a tall object's centre → less torque, fewer chain-reaction spins of small objects.
OUTSIDE_MARGIN = 0.05   # outside shelf front before push start (rod tip reference)
APPROACH_EXTRA = 0.10   # additional approach distance before start
GRIPPER_LENGTH = 0.25   # push rod length: wrist → rod tip (25 cm)


def compute_frontier_pushes(pre_gt_e: torch.Tensor, n: int = SUBSCENE_LEN):
    """Frontier-based push specs sampling for a single env.

    pre_gt_e : (W=120, H=60, D=80) — get_gt_id_map output, axis order (X, Z, Y).
    n        : number of unique x voxels to sample.

    Frontier: an x column counts as a candidate if it has an occupied voxel at
    y ∈ [0, depth_voxels), i.e. within FRONTIER_BASE_CM of the shelf entrance.
    If fewer than n unique columns qualify, the slab depth grows by
    FRONTIER_EXPAND_CM at a time, up to FRONTIER_MAX_CM.

    Returns: list of (x_voxel, y_voxel, angle_deg) tuples, length n.
        - y_voxel: y of the occupied voxel nearest the entrance in that column
        - angle_deg: uniform in ±θ_max, with θ_max = min(d_cm, ANGLE_MAX_DEG)
                     and d_cm = distance to the nearer x edge, in cm.
    """
    depth_cm = FRONTIER_BASE_CM
    candidate_xs: list = []
    while True:
        depth_vox = min(int(round(depth_cm * 0.01 / VOXEL_SIZE)), VOXEL_D)
        slab = pre_gt_e[:, :, :depth_vox]                 # (W, H, depth_vox)
        x_has_obj = (slab > 0).any(dim=2).any(dim=1)      # (W,) bool
        candidate_xs = x_has_obj.nonzero(as_tuple=False).flatten().tolist()
        if len(candidate_xs) >= n or depth_cm >= FRONTIER_MAX_CM:
            break
        depth_cm += FRONTIER_EXPAND_CM

    # Prefer unique columns; pad with replacement when there are too few.
    if len(candidate_xs) >= n:
        sampled_xs = random.sample(candidate_xs, n)
    elif len(candidate_xs) > 0:
        sampled_xs = list(candidate_xs)
        while len(sampled_xs) < n:
            sampled_xs.append(random.choice(candidate_xs))
    else:
        # Fallback for an empty GT (no objects): sample x uniformly.
        sampled_xs = [random.randint(0, VOXEL_W - 1) for _ in range(n)]

    pushes = []
    for x in sampled_xs:
        col = pre_gt_e[x, :, :]                            # (H=Z, D=Y)
        y_has_obj = (col > 0).any(dim=0)                   # (Y,)
        ys = y_has_obj.nonzero(as_tuple=False).flatten()
        y_voxel = int(ys[0]) if len(ys) > 0 else int(VOXEL_D * 0.4)

        x_cm = x * VOXEL_SIZE * 100.0
        d_cm = min(x_cm, (VOXEL_W - 1 - x) * VOXEL_SIZE * 100.0)
        theta_max = min(d_cm, ANGLE_MAX_DEG)
        angle_deg = random.uniform(-theta_max, theta_max)
        pushes.append((int(x), y_voxel, float(angle_deg)))
    return pushes


def get_angled_push_params(x_voxel: int, y_voxel: int, angle_deg: float,
                           env_origin: torch.Tensor,
                           extra_m: float = PUSH_BEYOND_M):
    """Compute wrist approach/start/end world coords for an angled push.

    Push direction (xy plane): rotate +Y by angle_deg around +Z axis.
        push_dir = (-sin θ, cos θ, 0)
        angle_deg < 0  →  +X component, the object is pushed to the right.
        angle_deg > 0  →  -X component, the object is pushed to the left.
    Rod tip start: OUTSIDE_MARGIN in front of the shelf entrance, on the push line.
    Rod tip end:   extra_m past the target voxel along push_dir.
    The wrist trails the rod tip by GRIPPER_LENGTH along push_dir.
    """
    angle_rad = math.radians(angle_deg)
    push_dx = -math.sin(angle_rad)
    push_dy =  math.cos(angle_rad)

    target_wx = float(env_origin[0]) + SHELF_X_OFFSET + (x_voxel + 0.5) * VOXEL_SIZE
    target_wy = float(env_origin[1]) + SHELF_Y_OFFSET + (y_voxel + 0.5) * VOXEL_SIZE
    target_wz = float(env_origin[2]) + SHELF_Y + PUSH_Z_OFFSET

    shelf_front_y = float(env_origin[1]) + SHELF_Y_OFFSET
    rod_tip_init_y = shelf_front_y - OUTSIDE_MARGIN
    if abs(push_dy) < 1e-3:
        push_dy = 1e-3 if push_dy >= 0 else -1e-3
    D_init = (target_wy - rod_tip_init_y) / push_dy
    rod_tip_init_x = target_wx - push_dx * D_init

    # Wrist start = rod tip init - push_dir * GRIPPER_LENGTH
    start_wx = rod_tip_init_x - push_dx * GRIPPER_LENGTH
    start_wy = rod_tip_init_y - push_dy * GRIPPER_LENGTH
    start_wz = target_wz

    # Wrist end = wrist start + push_dir * (D_init + extra_m)
    push_dist = D_init + extra_m
    end_wx = start_wx + push_dx * push_dist
    end_wy = start_wy + push_dy * push_dist
    end_wz = start_wz

    # Approach: wrist start - push_dir * APPROACH_EXTRA
    approach_wx = start_wx - push_dx * APPROACH_EXTRA
    approach_wy = start_wy - push_dy * APPROACH_EXTRA
    approach_wz = start_wz

    start_w    = torch.tensor([start_wx, start_wy, start_wz], dtype=torch.float32)
    end_w      = torch.tensor([end_wx, end_wy, end_wz], dtype=torch.float32)
    approach_w = torch.tensor([approach_wx, approach_wy, approach_wz], dtype=torch.float32)
    orig_t = env_origin.unsqueeze(0)
    start_voxel = world_to_voxel(start_w.unsqueeze(0), orig_t)[0]
    end_voxel   = world_to_voxel(end_w.unsqueeze(0),   orig_t)[0]
    return start_voxel, end_voxel, start_w, end_w, approach_w


def get_per_env_angled_push_params(push_specs_per_env: list, num_envs: int,
                                    env_origins: torch.Tensor):
    """Combine per-env (x, y, angle_deg) tuples into batched tensors for apply_push_action.

    push_specs_per_env: list of (x_voxel, y_voxel, angle_deg) length num_envs.
    """
    start_voxels = torch.zeros(num_envs, 3, dtype=torch.long)
    end_voxels   = torch.zeros(num_envs, 3, dtype=torch.long)
    start_w      = torch.zeros(num_envs, 3)
    end_w        = torch.zeros(num_envs, 3)
    approach_w   = torch.zeros(num_envs, 3)
    angles_deg   = torch.zeros(num_envs)
    for e in range(num_envs):
        x, y, a = push_specs_per_env[e]
        sv, ev, sw, ew, aw = get_angled_push_params(x, y, a, env_origins[e].cpu())
        start_voxels[e] = sv; end_voxels[e] = ev
        start_w[e] = sw; end_w[e] = ew; approach_w[e] = aw
        angles_deg[e] = float(a)
    return start_voxels, end_voxels, start_w, end_w, approach_w, angles_deg


def apply_push_action(scene, sim, start_w_in, end_w_in, approach_w_in, push_steps=40):
    """Straight-Y push for all envs simultaneously."""
    num_envs    = start_w_in.shape[0]
    env_origins = scene.env_origins
    device      = env_origins.device
    robot       = scene["robot"]
    sim_cfg_dt  = 0.01

    ik_pose = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False,
                                    ik_method="dls", ik_params={"lambda_val": 0.05}),
        num_envs=num_envs, device=device,
    )
    ee_body_ids, _ = robot.find_bodies("wrist_3_link")
    ee_idx = ee_body_ids[0]
    arm_joint_ids, _ = robot.find_joints(
        ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint",
         "wrist_1_joint","wrist_2_joint","wrist_3_joint"])
    jacobi_body_idx = ee_idx - 1

    start_w  = start_w_in.to(device).clone()
    end_w    = end_w_in.to(device).clone()
    approach_w = approach_w_in.to(device).clone()

    z_floor = env_origins[:, 2] + SHELF_Y + 0.10
    start_w[:, 2]    = torch.max(start_w[:, 2], z_floor)
    end_w[:, 2]      = torch.max(end_w[:, 2], z_floor)
    approach_w[:, 2] = torch.max(approach_w[:, 2], z_floor)

    # Push direction (straight Y)
    push_vecs = end_w - start_w
    push_vecs = push_vecs / push_vecs.norm(dim=1, keepdim=True).clamp(min=1e-6)

    # Gripper orientation aligned with push direction
    world_up = torch.zeros_like(push_vecs); world_up[:, 2] = 1.0
    ez = push_vecs
    ex = torch.cross(world_up, ez, dim=1)
    ex = ex / ex.norm(dim=1, keepdim=True).clamp(min=1e-6)
    ey = torch.cross(ez, ex, dim=1)
    ey = ey / ey.norm(dim=1, keepdim=True).clamp(min=1e-6)
    R  = torch.stack([ex, ey, ez], dim=2)

    push_quat_b = torch.zeros(num_envs, 4, device=device)
    t = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    mA = t > 0
    if mA.any():
        s = 0.5 / torch.sqrt((t[mA]+1.0).clamp(min=1e-10))
        push_quat_b[mA,0] = 0.25/s; push_quat_b[mA,1] = (R[mA,2,1]-R[mA,1,2])*s
        push_quat_b[mA,2] = (R[mA,0,2]-R[mA,2,0])*s; push_quat_b[mA,3] = (R[mA,1,0]-R[mA,0,1])*s
    mB = (~mA)&(R[:,0,0]>=R[:,1,1])&(R[:,0,0]>=R[:,2,2])
    if mB.any():
        s = 2.0*torch.sqrt((1.0+R[mB,0,0]-R[mB,1,1]-R[mB,2,2]).clamp(min=1e-10))
        push_quat_b[mB,0]=(R[mB,2,1]-R[mB,1,2])/s; push_quat_b[mB,1]=0.25*s
        push_quat_b[mB,2]=(R[mB,0,1]+R[mB,1,0])/s; push_quat_b[mB,3]=(R[mB,0,2]+R[mB,2,0])/s
    mC = (~mA)&(~mB)&(R[:,1,1]>=R[:,2,2])
    if mC.any():
        s = 2.0*torch.sqrt((1.0+R[mC,1,1]-R[mC,0,0]-R[mC,2,2]).clamp(min=1e-10))
        push_quat_b[mC,0]=(R[mC,0,2]-R[mC,2,0])/s; push_quat_b[mC,1]=(R[mC,0,1]+R[mC,1,0])/s
        push_quat_b[mC,2]=0.25*s; push_quat_b[mC,3]=(R[mC,1,2]+R[mC,2,1])/s
    mD = (~mA)&(~mB)&(~mC)
    if mD.any():
        s = 2.0*torch.sqrt((1.0+R[mD,2,2]-R[mD,0,0]-R[mD,1,1]).clamp(min=1e-10))
        push_quat_b[mD,0]=(R[mD,1,0]-R[mD,0,1])/s; push_quat_b[mD,1]=(R[mD,0,2]+R[mD,2,0])/s
        push_quat_b[mD,2]=(R[mD,1,2]+R[mD,2,1])/s; push_quat_b[mD,3]=0.25*s
    push_quat_b = torch.nn.functional.normalize(push_quat_b, dim=1)

    GRIPPER_REACH = 0.0
    start_w    = start_w    - push_vecs * GRIPPER_REACH
    end_w      = end_w      - push_vecs * GRIPPER_REACH
    approach_w = approach_w - push_vecs * GRIPPER_REACH

    def to_base(pos_w):
        R_inv = matrix_from_quat(quat_inv(robot.data.root_quat_w))
        return torch.bmm(R_inv, (pos_w - robot.data.root_pos_w).unsqueeze(-1)).squeeze(-1)

    def get_ee_pose_b():
        return subtract_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w,
            robot.data.body_pos_w[:, ee_idx], robot.data.body_quat_w[:, ee_idx])

    def get_jacobian_b():
        jac_full = robot.root_physx_view.get_jacobians()
        jac = jac_full[:, jacobi_body_idx, :, arm_joint_ids]
        R_inv = matrix_from_quat(quat_inv(robot.data.root_quat_w))
        jac[:,:3,:] = torch.bmm(R_inv, jac[:,:3,:]); jac[:,3:,:] = torch.bmm(R_inv, jac[:,3:,:])
        return jac

    def _sim_step(jdes):
        robot.set_joint_position_target(jdes, joint_ids=arm_joint_ids)
        scene.write_data_to_sim(); sim.step(); scene.update(sim_cfg_dt)

    def ik_step_pose(target_pos_w, target_quat_b):
        safe = target_pos_w.clone(); safe[:, 2] = torch.max(safe[:, 2], z_floor)
        ee_pos_b, ee_quat_b = get_ee_pose_b()
        cmd = torch.cat([to_base(safe), target_quat_b], dim=1)
        ik_pose.set_command(cmd)
        jdes = ik_pose.compute(ee_pos_b, ee_quat_b, get_jacobian_b(),
                               robot.data.joint_pos[:, arm_joint_ids])
        _sim_step(jdes)

    home_des = torch.zeros(num_envs, robot.num_joints, device=device)
    for idx, val in zip(arm_joint_ids, [1.57, -1.2, 2.2, -1.0, 1.57, 0.0]):
        home_des[:, idx] = val
    for _ in range(50):
        robot.set_joint_position_target(home_des[:, arm_joint_ids], joint_ids=arm_joint_ids)
        scene.write_data_to_sim(); sim.step(); scene.update(sim_cfg_dt)

    for _ in range(120): ik_step_pose(approach_w, push_quat_b)
    for _ in range(120): ik_step_pose(start_w, push_quat_b)

    ee_at_start  = robot.data.body_pos_w[:, ee_idx].clone()
    ee_traj_list = []
    for step in range(push_steps):
        frac = (step + 1) / push_steps
        ik_step_pose(ee_at_start + frac * (end_w - ee_at_start), push_quat_b)
        ee_pos_w = robot.data.body_pos_w[:, ee_idx]
        rod_tip_w = ee_pos_w + push_vecs * GRIPPER_LENGTH
        for name in OBJ_NAMES:
            if name not in scene._rigid_objects: continue
            obj = scene[name]
            vel = torch.zeros(num_envs, 6, device=device)
            for e in range(num_envs):
                if (obj.data.root_pos_w[e] - rod_tip_w[e]).norm() < 0.05:
                    vel[e, :3] = push_vecs[e] * 0.15
            obj.write_root_velocity_to_sim(vel)
        ee_traj_list.append(torch.cat([robot.data.body_pos_w[:, ee_idx].cpu(),
                                        robot.data.body_quat_w[:, ee_idx].cpu()], dim=1))

    ee_traj = torch.stack(ee_traj_list, dim=1)

    # Slow retract: reverse push trajectory (end → start → approach)
    ee_at_end = robot.data.body_pos_w[:, ee_idx].clone()
    for step in range(push_steps):
        frac = (step + 1) / push_steps
        ik_step_pose(ee_at_end + frac * (ee_at_start - ee_at_end), push_quat_b)
    for _ in range(120): ik_step_pose(approach_w, push_quat_b)
    for _ in range(60):
        robot.set_joint_position_target(home_des[:, arm_joint_ids], joint_ids=arm_joint_ids)
        scene.write_data_to_sim(); sim.step(); scene.update(sim_cfg_dt)
    return ee_traj


# ── Object placement ─────────────────────────────────────────────────────────

def _obb_overlap(c1, h1, yaw1, c2, h2, yaw2, margin=0.0):
    """2D OBB-OBB overlap test (SAT). c=(x,y) center, h=(hx,hy) half-extents.
    Returns True if the two oriented bounding boxes overlap (with optional safety margin).
    """
    cos1, sin1 = math.cos(yaw1), math.sin(yaw1)
    cos2, sin2 = math.cos(yaw2), math.sin(yaw2)
    axes = [(cos1, sin1), (-sin1, cos1), (cos2, sin2), (-sin2, cos2)]
    dx, dy = c2[0] - c1[0], c2[1] - c1[1]
    for ax, ay in axes:
        r1 = abs(h1[0]*(ax*cos1 + ay*sin1)) + abs(h1[1]*(-ax*sin1 + ay*cos1))
        r2 = abs(h2[0]*(ax*cos2 + ay*sin2)) + abs(h2[1]*(-ax*sin2 + ay*cos2))
        if abs(dx*ax + dy*ay) > r1 + r2 + margin:
            return False    # separating axis → no overlap
    return True


def _pick_active_slots():
    """Choose the N_ACTIVE (= 11) slots active this episode.

      Phase 1 — Unique:    one instance of each of the 11 types (5 tall + 6 small).
      Phase 2 — Duplicate: N_DUP (= 0) extra slots, small types only.
    """
    chosen_slots = []
    slot_used    = {}

    # Phase 1: every type gets one instance
    type_order = list(TYPE_BY_ID.keys())
    random.shuffle(type_order)
    for tid in type_order:
        first = random.randint(0, 1)
        chosen_slots.append(SLOT_NAMES[(tid - 1) * 2 + first])
        slot_used[tid] = first

    # Phase 2: SMALL only duplicate, distinct types
    pool     = list(SMALL_IDS)
    used_dup = set()
    for _ in range(N_DUP):
        avail = [t for t in pool if t not in used_dup]
        if not avail:
            avail = pool
        tid = random.choice(avail)
        used_dup.add(tid)
        other = 1 - slot_used[tid]
        chosen_slots.append(SLOT_NAMES[(tid - 1) * 2 + other])
    return chosen_slots


def randomize_object_positions(scene):
    """
    Random selection every episode.
      - Only N_ACTIVE (= 11) of the 22 slots are active.
      - All 11 types appear once, plus N_DUP (= 0) random duplicates.
      - Active tall types go in the front zone of the shelf, small types in the back zone.
      - Inactive slots stay at their spawn position (off-shelf, slot_init_pos).
    Yaw is random (about the vertical axis); the placement bound r_xy=√(hx²+hy²)
    keeps objects inside the shelf at any yaw.
    """
    global _active_slots, _active_scales
    num_envs = scene.num_envs
    HALF_W   = SHELF_W / 2
    HALF_D   = SHELF_D / 2
    MARGIN   = 0.05
    SAFETY   = 0.005         # SAT margin: at least a 5 mm gap between two OBBs
    MAX_TRIES = 200
    TALL_Y_BACK = -HALF_D + 0.22

    _active_slots  = [None] * num_envs
    _active_scales = [None] * num_envs

    shape_ood = bool(getattr(args_cli, "shape_ood", False))
    s_low, s_high = float(args_cli.scale_low), float(args_cli.scale_high)
    if shape_ood:
        import omni.usd
        _stage_ood = omni.usd.get_context().get_stage()

    for e in range(num_envs):
        orig = scene.env_origins[e]
        ox, oy, oz = float(orig[0]), float(orig[1]), float(orig[2])

        active = _pick_active_slots()
        _active_slots[e]  = list(active)
        _active_scales[e] = {}

        if shape_ood:
            for name in SLOT_NAMES:
                if name not in scene._rigid_objects:
                    continue
                if name in active:
                    sc = _sample_scale_local(slot_type_id(name), s_low, s_high)
                    _active_scales[e][name] = sc
                    _apply_xform_scale(_stage_ood, e, name, sc)
                else:
                    _apply_xform_scale(_stage_ood, e, name, (1.0, 1.0, 1.0))
        else:
            for name in active:
                _active_scales[e][name] = (1.0, 1.0, 1.0)

        # Send every inactive slot to its off-shelf default location.
        inactive_set = set(SLOT_NAMES) - set(active)
        for name in inactive_set:
            if name not in scene._rigid_objects:
                continue
            ix, iy, iz = slot_init_pos(SLOT_NAMES.index(name))
            obj = scene[name]
            s = obj.data.root_state_w.clone()
            s[e, 0] = ox + ix; s[e, 1] = oy + iy; s[e, 2] = oz + iz
            sw, sx, sy, sz = obj.cfg.init_state.rot
            s[e, 3] = sw; s[e, 4] = sx; s[e, 5] = sy; s[e, 6] = sz
            s[e, 7:] = 0.0
            obj.write_root_state_to_sim(s)

        # Split active into tall / small zones
        tall_slots  = [n for n in active if slot_is_tall(n)]
        small_slots = [n for n in active if not slot_is_tall(n)]
        random.shuffle(tall_slots)
        random.shuffle(small_slots)

        placed_obb_tall = []
        placed_obb_all  = []

        def _place(name, zone):
            hx, hy, hz = _half_world_for(name, _active_scales[e].get(name, (1.0, 1.0, 1.0)))
            r_xy = math.sqrt(hx * hx + hy * hy)
            h2d  = (hx, hy)
            x_lo = ox - HALF_W + MARGIN + r_xy
            x_hi = ox + HALF_W - MARGIN - r_xy
            if zone == "tall":
                y_lo = oy + (-HALF_D + MARGIN) + r_xy
                y_hi = oy + TALL_Y_BACK - r_xy
            else:
                y_lo = oy + TALL_Y_BACK + r_xy
                y_hi = oy + (HALF_D - MARGIN) - r_xy
            if x_lo > x_hi: x_lo = x_hi = ox
            if y_lo > y_hi: y_lo = y_hi = (y_lo + y_hi) / 2

            avoid = placed_obb_tall if zone == "tall" else placed_obb_all
            placed_ok = False
            yaw = 0.0
            for _ in range(MAX_TRIES):
                x = random.uniform(x_lo, x_hi)
                y = random.uniform(y_lo, y_hi)
                yaw_try = random.uniform(0, 2 * math.pi)
                ok = True
                for (px, py, ph, pyaw) in avoid:
                    if _obb_overlap((x, y), h2d, yaw_try,
                                    (px, py), ph, pyaw, margin=SAFETY):
                        ok = False; break
                if ok:
                    yaw = yaw_try
                    placed_ok = True
                    break
            if not placed_ok:
                print(f"[WARN] placement failed: env={e} {name} zone={zone} "
                      f"({MAX_TRIES} retries — last sample likely overlap)")
            z = oz + SHELF_Y + hz

            yqw = math.cos(yaw / 2); yqz = math.sin(yaw / 2)
            obj = scene[name]
            sw, sx, sy, sz = obj.cfg.init_state.rot
            qw = yqw * sw - yqz * sz
            qx = yqw * sx - yqz * sy
            qy = yqw * sy + yqz * sx
            qz = yqw * sz + yqz * sw

            placed_obb_all.append((x, y, h2d, yaw))
            if zone == "tall":
                placed_obb_tall.append((x, y, h2d, yaw))
            s = obj.data.root_state_w.clone()
            s[e, 0] = x; s[e, 1] = y; s[e, 2] = z
            s[e, 3] = qw; s[e, 4] = qx; s[e, 5] = qy; s[e, 6] = qz
            s[e, 7:] = 0.0
            obj.write_root_state_to_sim(s)

        for name in tall_slots:
            if name in scene._rigid_objects:
                _place(name, "tall")
        for name in small_slots:
            if name in scene._rigid_objects:
                _place(name, "small")


# ── Friction / gripper ───────────────────────────────────────────────────────

def setup_low_com(stage, num_envs, com_ratio=0.5):
    """Shift each YCB object's centerOfMass down the standing axis by
    com_ratio x the mesh AABB half-extent, so that a standing object has a low
    COM and is harder to knock over domino-style.

      _Q_Y_UP    : local +Y → world +Z, so the COM moves towards local -Y (= down)
      _Q_NEG_Y_UP: local -Y → world +Z, so the COM moves towards local +Y (= down)
      _Q_X_UP    : local +X → world +Z, so the COM moves towards local -X (= down)
    """
    from pxr import UsdPhysics, Gf
    applied = 0
    for e in range(num_envs):
        for slot_idx, name in enumerate(SLOT_NAMES):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{e}/{name}")
            if not prim.IsValid():
                continue
            type_id = SLOT_TO_TYPE_ID[slot_idx]
            rot = ROT_BY_ID[type_id]
            hl  = HALF_LOCAL[type_id]
            if rot == _Q_Y_UP:
                com = (0.0, -hl[1] * com_ratio, 0.0)
            elif rot == _Q_NEG_Y_UP:
                com = (0.0, +hl[1] * com_ratio, 0.0)
            elif rot == _Q_X_UP:
                com = (-hl[0] * com_ratio, 0.0, 0.0)
            else:
                continue
            mass_api = UsdPhysics.MassAPI(prim)
            if not mass_api:
                mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*com))
            applied += 1
    print(f"[INFO] CenterOfMass set on {applied} obj prims (com_ratio={com_ratio})")


def set_object_friction(stage, num_envs, static_friction=0.15, dynamic_friction=0.10):
    from pxr import UsdPhysics, UsdShade, Usd
    mat = UsdShade.Material.Define(stage, "/World/PhysicsMaterials/LowFriction")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr().Set(static_friction)
    api.CreateDynamicFrictionAttr().Set(dynamic_friction)
    api.CreateRestitutionAttr().Set(0.05)
    applied = 0
    for e in range(num_envs):
        for name in OBJ_NAMES + ["shelf_base"]:
            prim = stage.GetPrimAtPath(f"/World/envs/env_{e}/{name}")
            if not prim.IsValid(): continue
            for sub in Usd.PrimRange(prim):
                UsdShade.MaterialBindingAPI.Apply(sub).Bind(
                    mat, UsdShade.Tokens.weakerThanDescendants, "physics")
                applied += 1
    print(f"[INFO] Friction applied to {applied} prims")


def attach_gripper_visuals(stage, num_envs):
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    PARTS = [
        ("palm",         (0.090, 0.040, 0.025), ( 0.000, 0.000, 0.020), (0.15, 0.15, 0.15)),
        ("left_finger",  (0.012, 0.014, 0.080), (-0.006, 0.000, 0.080), (0.12, 0.12, 0.12)),
        ("right_finger", (0.012, 0.014, 0.080), ( 0.006, 0.000, 0.080), (0.12, 0.12, 0.12)),
        ("left_tip",     (0.010, 0.010, 0.012), (-0.002, 0.004, 0.126), (0.25, 0.08, 0.08)),
        ("right_tip",    (0.010, 0.010, 0.012), ( 0.002, 0.004, 0.126), (0.25, 0.08, 0.08)),
    ]
    attached = 0
    for e in range(num_envs):
        robot_prim = stage.GetPrimAtPath(f"/World/envs/env_{e}/robot")
        if not robot_prim.IsValid(): continue
        tool0_path = None
        for prim in Usd.PrimRange(robot_prim):
            if prim.GetName() == "tool0":
                tool0_path = str(prim.GetPath()); break
        if tool0_path is None:
            for prim in Usd.PrimRange(robot_prim):
                if prim.GetName() in ("flange", "wrist_3_link"):
                    tool0_path = str(prim.GetPath()); break
        if tool0_path is None:
            print(f"[WARN] gripper attach: tool0/flange not found in env_{e}"); continue
        scope = tool0_path + "/robotiq_2f85"
        stage.DefinePrim(scope, "Scope")
        for name, (sx, sy, sz), (tx, ty, tz), color in PARTS:
            cube = UsdGeom.Cube.Define(stage, f"{scope}/{name}")
            cube.GetSizeAttr().Set(1.0)
            xf = UsdGeom.Xformable(cube)
            ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
            if "xformOp:translate" in ops: ops["xformOp:translate"].Set(Gf.Vec3d(tx, ty, tz))
            else: xf.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))
            if "xformOp:scale" in ops: ops["xformOp:scale"].Set(Gf.Vec3d(sx, sy, sz))
            else: xf.AddScaleOp().Set(Gf.Vec3d(sx, sy, sz))
            cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        attached += 1
    print(f"[INFO] Robotiq 2F-85 gripper attached to {attached}/{num_envs} envs")


# ── Semantic segmentation setup ─────────────────────────────────────────────

_SEM_ID_MAP: dict = {}        # {pixel_value: type_id_1based}, built after the first capture
_SEM_ID_TO_SLOT: dict = {}    # {pixel_value: slot_idx}, for the test-set visibility check


def setup_semantic_labels(num_envs, slot_names):
    """Give every slot a type-class label. Both slot instances of a type share
    the same class string ('type_<id>'), so the camera segmentation groups them
    under one pixel value. In the YCB USD root xform → child Mesh hierarchy the
    label is not always inherited down to the mesh, which drops it from the
    segmentation, so the label is applied directly to the root and to every
    child Mesh.
    """
    import omni.usd
    from pxr import Usd
    _stage = omni.usd.get_context().get_stage()
    labeled = 0
    mesh_per_slot = {}   # diagnostics: number of meshes found per slot
    for e in range(num_envs):
        for name in slot_names:
            root_prim = _stage.GetPrimAtPath(f"/World/envs/env_{e}/{name}")
            if not root_prim.IsValid():
                continue
            type_id = SLOT_TO_TYPE_ID[SLOT_NAMES.index(name)]
            label = f"type_{type_id}"
            # label both the root and every child Mesh, so nothing is lost to inheritance
            sim_utils.add_labels(root_prim, labels=[label], instance_name="class")
            labeled += 1
            n_mesh = 0
            # AllPrims also traverses instance proxies (prototypes of instanceable references)
            for sub in Usd.PrimRange.AllPrims(root_prim):
                if sub.GetTypeName() == "Mesh":
                    sim_utils.add_labels(sub, labels=[label], instance_name="class")
                    labeled += 1
                    n_mesh += 1
            if e == 0:
                mesh_per_slot[name] = n_mesh
    print(f"[INFO] Semantic labels applied to {labeled} prims "
          f"(root + child mesh, 11 type classes)")
    print(f"[INFO] Per-slot mesh count (env 0): {mesh_per_slot}")

    # Count the prims actually carrying a 'type_<id>' label on the stage (checks for missing LabelsAPI).
    try:
        from pxr import UsdSemantics
        type_count = {f"type_{t}": 0 for t in TYPE_BY_ID}
        for prim in _stage.Traverse():
            for schema in prim.GetAppliedSchemas():
                if not schema.startswith("SemanticsLabelsAPI:"):
                    continue
                inst = schema.split(":", 1)[1]
                if inst != "class":
                    continue
                api = UsdSemantics.LabelsAPI(prim, inst)
                lattr = api.GetLabelsAttr()
                vals = lattr.Get() if lattr else None
                if not vals:
                    continue
                for v in vals:
                    if v in type_count:
                        type_count[v] += 1
        print(f"[INFO] LabelsAPI verification (stage-wide 'class' labels): {type_count}")
    except Exception as exc:
        print(f"[WARN] LabelsAPI verification failed: {exc}")


_SLOT_RE = __import__("re").compile(r"/obj_(\d+)\b")


def build_sem_id_map(cam, verbose=True):
    """Extract '/obj_<i>' from the idToLabels (= prim_path) of
    instance_id_segmentation_fast and map it to a type_id via SLOT_TO_TYPE_ID[i].
    The map is accumulated (union) across calls.
    """
    global _SEM_ID_MAP, _SEM_ID_TO_SLOT
    try:
        info = cam.data.info
        if not info:
            return False
        seg_info = info[0].get("instance_id_segmentation_fast", {})
        if "idToLabels" in seg_info:
            seg_info = seg_info["idToLabels"]
        id_map = {}
        slot_map = {}
        for pix_val, label in seg_info.items():
            path = (label.get("class", "") if isinstance(label, dict) else str(label))
            m = _SLOT_RE.search(path)
            if not m:
                continue
            slot_idx = int(m.group(1))
            if slot_idx in SLOT_TO_TYPE_ID:
                id_map[int(pix_val)] = SLOT_TO_TYPE_ID[slot_idx]
                slot_map[int(pix_val)] = slot_idx
        prev = dict(_SEM_ID_MAP)
        added_keys = set(id_map) - set(prev)
        prev.update(id_map)
        _SEM_ID_MAP = prev
        slot_prev = dict(_SEM_ID_TO_SLOT)
        slot_prev.update(slot_map)
        _SEM_ID_TO_SLOT = slot_prev
        seen_types = sorted(set(_SEM_ID_MAP.values()))
        missing = [t for t in range(1, N_TYPES + 1) if t not in seen_types]
        if verbose or (not missing and added_keys):
            print(f"[INFO] Semantic ID map: {len(_SEM_ID_MAP)} instance-id → "
                  f"types {seen_types}  missing={missing}")
        return len(missing) == 0
    except Exception as exc:
        print(f"[WARN] build_sem_id_map failed: {exc}")
        return False


def apply_sem_id_map(sem_frame):
    """
    sem_frame: (H, W, 1) int32 raw instance IDs (colorize=False)
    → (H, W) int32, 0 = background, 1..N_TYPES = semantic type id
    """
    if isinstance(sem_frame, torch.Tensor):
        arr = sem_frame.numpy()
    else:
        arr = sem_frame
    # colorize=False: shape (H, W, 1) int32 → squeeze to (H, W)
    if arr.ndim == 3:
        arr = arr[..., 0]
    raw = arr.astype(np.int32)
    out = np.zeros_like(raw, dtype=np.int32)
    for pix_id, obj_idx in _SEM_ID_MAP.items():
        out[raw == pix_id] = obj_idx
    return out


def visible_slot_idxs_from_raw(sem_frame):
    """
    sem_frame: (H, W, 1) or (H, W) raw pixel IDs (one view).
    Returns: set[int] of slot_idx that have >=1 pixel in this view.
    Requires _SEM_ID_TO_SLOT to be populated (build_sem_id_map called at least once).
    """
    arr = sem_frame.numpy() if isinstance(sem_frame, torch.Tensor) else sem_frame
    if arr.ndim == 3:
        arr = arr[..., 0]
    seen = set()
    for pid in np.unique(arr):
        slot = _SEM_ID_TO_SLOT.get(int(pid))
        if slot is not None:
            seen.add(slot)
    return seen


def encode_rgb_depth(rgb_tensor, depth_tensor):
    """
    rgb_tensor:   (N, H, W, 3) uint8 torch
    depth_tensor: (N, H, W)    float32 torch [m]
    → rgb_jpeg: object array of uint8 JPEG bytes (N,)
    → depth_u16: (N, H, W) uint16 [mm]
    """
    rgb_np    = rgb_tensor.numpy()
    depth_np  = depth_tensor.numpy()
    rgb_jpeg  = np.empty(len(rgb_np), dtype=object)
    for i, frame in enumerate(rgb_np):
        bgr = frame[..., ::-1].copy()
        _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        rgb_jpeg[i] = np.frombuffer(buf, dtype=np.uint8)
    depth_mm  = np.nan_to_num(depth_np, nan=0.0)
    depth_u16 = np.clip(depth_mm * 1000, 0, 65535).astype(np.uint16)
    return rgb_jpeg, depth_u16


# ── Episode log ──────────────────────────────────────────────────────────────

def save_episode_log(save_base, folder_id, push_specs_per_subscene, transition_steps):
    """Write episode_log.txt into the episode's root folder.

    push_specs_per_subscene: list of N_SUBSCENES lists, each SUBSCENE_LEN tuples
                              (x_voxel, y_voxel, angle_deg).
    transition_steps: list of N_TRANSITIONS ints — the step (0..SUBSCENE_LEN-1) within
                      each subscene that becomes the next subscene's PRE.
    """
    folder = os.path.join(save_base, f"{folder_id:09d}")
    os.makedirs(folder, exist_ok=True)
    lines = [f"episode: {folder_id:09d}", ""]
    for s, label in enumerate(SUBSCENE_LABELS):
        lo = s * SUBSCENE_LEN
        hi = (s + 1) * SUBSCENE_LEN - 1
        lines.append(f"sub_scene_{label}  steps {lo:02d}-{hi:02d}")
        for i, (xv, yv, ang) in enumerate(push_specs_per_subscene[s]):
            lines.append(f"  post_{lo+i:02d}: x_vox={xv:3d}  y_vox={yv:3d}  angle_deg={ang:+7.3f}")
        if s < N_TRANSITIONS:
            ts = transition_steps[s]
            next_label = SUBSCENE_LABELS[s + 1]
            lines.append(
                f"  -> transition_{label}{next_label}: result of post_{lo+ts:02d} becomes PRE_{next_label}"
            )
        lines.append("")
    with open(os.path.join(folder, "episode_log.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ── Scene-based save (one folder per env x super-cycle) ─────────────────────
#
# Folder layout (5 subscenes x 20 steps = 100 post dirs per episode):
#   XXXXXXXXX/
#     pre_gt.npz, occ_pre.npz, pre_rgbd.npz, pre_semantic.npz   ← subscene A PRE
#     semantic_map.npz, camera_poses.npz
#     episode_log.txt   (push specs + transition step of every subscene)
#     post_00/ ~ post_19/   the 20 push results of subscene A
#     post_20/              subscene B PRE (pre_*.npz) + the post_20 push result
#     post_21/ ~ post_39/   the remaining push results of subscene B
#     post_40/              subscene C PRE + the post_40 push result
#     post_41/ ~ post_59/   subscene C
#     post_60/              subscene D PRE + the post_60 push result
#     post_61/ ~ post_79/   subscene D
#     post_80/              subscene E PRE + the post_80 push result
#     post_81/ ~ post_99/   subscene E
#
# action.npz holds: ee_poses, start_voxel, end_voxel, swept_map, push_angle_deg,
#                push_angle_rad.

def save_scene_pre_A(save_base, folder_id, env_orig_np,
                     pre_depth, cam_positions, cam_orientations, pre_gt_map,
                     pre_rgb=None, pre_sem=None):
    folder = os.path.join(save_base, f"{folder_id:09d}")
    os.makedirs(folder, exist_ok=True)
    gt_id_np = pre_gt_map.cpu().numpy().astype(np.int32)
    np.savez_compressed(os.path.join(folder, "pre_gt.npz"),
                        gt=_to_their_gt(gt_id_np))
    _save_semantic_map(os.path.join(folder, "semantic_map.npz"), gt_id_np)
    occ = depth_stack_to_occ(pre_depth, cam_positions, cam_orientations, env_orig_np)
    np.savez_compressed(os.path.join(folder, "occ_pre.npz"), occupied=occ)
    np.savez_compressed(os.path.join(folder, "camera_poses.npz"),
                        positions=cam_positions.numpy().astype(np.float32),
                        orientations=cam_orientations.numpy().astype(np.float32))
    if pre_rgb is not None:
        rgb_jpeg, depth_u16 = encode_rgb_depth(pre_rgb, pre_depth)
        np.savez_compressed(os.path.join(folder, "pre_rgbd.npz"),
                            rgb_jpeg=rgb_jpeg, depth_uint16=depth_u16)
    if pre_sem is not None:
        sem = np.stack([apply_sem_id_map(pre_sem[v]) for v in range(pre_sem.shape[0])])
        np.savez_compressed(os.path.join(folder, "pre_semantic.npz"),
                            semantic=sem.astype(np.int32))


def save_subscene_pre(save_base, folder_id, step_in_scene, env_orig_np,
                      pre_depth, cam_positions, cam_orientations, pre_gt_map,
                      pre_rgb=None, pre_sem=None):
    post_dir = os.path.join(save_base, f"{folder_id:09d}", f"post_{step_in_scene:02d}")
    os.makedirs(post_dir, exist_ok=True)
    gt_id_np = pre_gt_map.cpu().numpy().astype(np.int32)
    np.savez_compressed(os.path.join(post_dir, "pre_gt.npz"),
                        gt=_to_their_gt(gt_id_np))
    _save_semantic_map(os.path.join(post_dir, "semantic_map.npz"), gt_id_np)
    occ = depth_stack_to_occ(pre_depth, cam_positions, cam_orientations, env_orig_np)
    np.savez_compressed(os.path.join(post_dir, "pre_occ.npz"), occupied=occ)
    if pre_rgb is not None:
        rgb_jpeg, depth_u16 = encode_rgb_depth(pre_rgb, pre_depth)
        np.savez_compressed(os.path.join(post_dir, "pre_rgbd.npz"),
                            rgb_jpeg=rgb_jpeg, depth_uint16=depth_u16)
    if pre_sem is not None:
        sem = np.stack([apply_sem_id_map(pre_sem[v]) for v in range(pre_sem.shape[0])])
        np.savez_compressed(os.path.join(post_dir, "pre_semantic.npz"),
                            semantic=sem.astype(np.int32))


def save_post_step(save_base, folder_id, step_in_scene, env_orig_np,
                   post_depth, cam_positions, cam_orientations, post_gt_map,
                   start_voxel, end_voxel, swept_map, ee_traj,
                   push_angle_deg=0.0,
                   post_rgb=None, post_sem=None):
    post_dir = os.path.join(save_base, f"{folder_id:09d}", f"post_{step_in_scene:02d}")
    os.makedirs(post_dir, exist_ok=True)
    gt_id_np = post_gt_map.cpu().numpy().astype(np.int32)
    _save_semantic_map(os.path.join(post_dir, "semantic_map.npz"), gt_id_np)
    np.savez_compressed(os.path.join(post_dir, "gt.npz"),
                        gt=_to_their_gt(gt_id_np))
    occ = depth_stack_to_occ(post_depth, cam_positions, cam_orientations, env_orig_np)
    np.savez_compressed(os.path.join(post_dir, "occ.npz"), occupied=occ)
    np.savez_compressed(os.path.join(post_dir, "action.npz"),
                        ee_poses=ee_traj.numpy(),
                        start_voxel=start_voxel.numpy(),
                        end_voxel=end_voxel.numpy(),
                        swept_map=swept_map.numpy(),
                        push_angle_deg=np.float32(push_angle_deg),
                        push_angle_rad=np.float32(math.radians(push_angle_deg)))
    if post_rgb is not None:
        rgb_jpeg, depth_u16 = encode_rgb_depth(post_rgb, post_depth)
        np.savez_compressed(os.path.join(post_dir, "rgbd.npz"),
                            rgb_jpeg=rgb_jpeg, depth_uint16=depth_u16)
    if post_sem is not None:
        sem = np.stack([apply_sem_id_map(post_sem[v]) for v in range(post_sem.shape[0])])
        np.savez_compressed(os.path.join(post_dir, "semantic.npz"),
                            semantic=sem.astype(np.int32))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    num_envs      = args_cli.num_envs
    num_episodes  = args_cli.num_episodes
    save_base     = args_cli.save_path
    start_episode = args_cli.start_episode
    os.makedirs(save_base, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, render_interval=4)
    sim     = sim_utils.SimulationContext(sim_cfg)
    scene   = InteractiveScene(HighOcclusionYCBSceneCfg(num_envs=num_envs, env_spacing=3.0))

    import omni.usd
    _stage = omni.usd.get_context().get_stage()
    attach_gripper_visuals(_stage, num_envs)
    set_object_friction(_stage, num_envs)
    setup_low_com(_stage, num_envs, com_ratio=0.5)
    # NOTE: instance_id_segmentation_fast is used, so no separate class labels are needed;
    #       obj_<i> is mapped directly from the prim_path of each prim the camera sees.

    # Pre-cache YCB mesh sample points (USD → trimesh → voxelize). One-time.
    ycb_meshes.precache_all_types(
        (tid, USD_BY_ID[tid],
         f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/{USD_BY_ID[tid]}.usd")
        for tid in TYPE_BY_ID
    )

    sim.reset()

    _robot = scene["robot"]
    _arm_ids, _ = _robot.find_joints(
        ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint",
         "wrist_1_joint","wrist_2_joint","wrist_3_joint"])

    _fig9 = [1.57, -1.2, 2.2, -1.0, 1.57, 0.0]
    _home = torch.zeros(num_envs, _robot.num_joints, device=sim.device)
    for i, jid in enumerate(_arm_ids): _home[:, jid] = _fig9[i]
    _zero_vel = torch.zeros_like(_home)

    import math as _math
    _cap_angles = [_fig9[0] + _math.pi] + _fig9[1:]
    _cap_home   = torch.zeros(num_envs, _robot.num_joints, device=sim.device)
    for i, jid in enumerate(_arm_ids): _cap_home[:, jid] = _cap_angles[i]

    _robot.write_joint_state_to_sim(_cap_home, _zero_vel)
    scene.write_data_to_sim()

    print(f"[INFO] High-Occlusion-YCB | Envs:{num_envs} | Target:{num_episodes} | Save:{save_base}")

    def stabilize(steps, hold_jpos=None):
        for _ in range(steps):
            if hold_jpos is not None:
                _robot.set_joint_position_target(hold_jpos, joint_ids=_arm_ids)
            scene.write_data_to_sim(); sim.step(); scene.update(sim_cfg.dt)

    def save_states():
        saved = {}
        for name in OBJ_NAMES:
            if name in scene._rigid_objects:
                saved[name] = scene[name].data.root_state_w.clone()
        return saved

    def restore_states(saved):
        for name, state in saved.items():
            scene[name].write_root_state_to_sim(state)

    def reset_robot_to_cap():
        _robot.write_joint_state_to_sim(_cap_home, _zero_vel)
        scene.write_data_to_sim()
        stabilize(5, hold_jpos=_cap_home[:, _arm_ids])

    # Initial scene setup
    stabilize(5, hold_jpos=_cap_home[:, _arm_ids])
    randomize_object_positions(scene)
    reset_robot_to_cap()
    stabilize(50, hold_jpos=_cap_home[:, _arm_ids])   # let the objects settle after spawning (~0.5s)

    import time as _time

    _sem_map_built = False

    # ──────────────────────────────────────────────────────────────────────
    # Test-set generation mode (PRE-only, rejection-sampled). Mirrors the
    # implementation in scene_low_occlusion.py — bypasses the
    # full super-cycle / push loop.
    # ──────────────────────────────────────────────────────────────────────
    if args_cli.test_set_mode:
        import json
        random.seed(args_cli.test_set_seed)
        np.random.seed(args_cli.test_set_seed)
        torch.manual_seed(args_cli.test_set_seed)

        target  = args_cli.test_set_target
        max_try = args_cli.test_set_max_tries
        out_base = save_base
        os.makedirs(out_base, exist_ok=True)
        manifest_path = os.path.join(out_base, "manifest.json")

        episodes_meta = []
        saved = 0
        rejected = 0
        tries = 0
        print(f"[TEST_SET] target={target}  num_envs={num_envs}  seed={args_cli.test_set_seed}  out={out_base}")

        while saved < target and tries < max_try:
            tries += 1
            scene.reset()
            randomize_object_positions(scene)
            reset_robot_to_cap()
            stabilize(50, hold_jpos=_cap_home[:, _arm_ids])

            pre_d, pre_cp, pre_cq, pre_rgb, pre_sem = capture_all_viewpoints(
                scene, sim, sim_cfg, _cap_home[:, _arm_ids], _arm_ids)
            if not _sem_map_built:
                if build_sem_id_map(scene["camera"]):
                    _sem_map_built = True
            else:
                build_sem_id_map(scene["camera"], verbose=False)

            pre_gt = torch.stack([get_gt_id_map(scene, e) for e in range(num_envs)])

            for e in range(num_envs):
                if saved >= target:
                    break
                active_slot_names = list(get_active_slots(e))
                active_slot_idxs  = set(SLOT_NAMES.index(n) for n in active_slot_names)
                visible = set()
                for v in range(N_VIEWS):
                    visible |= visible_slot_idxs_from_raw(pre_sem[e, v])
                hidden = active_slot_idxs - visible
                if not hidden:
                    rejected += 1
                    continue

                fid = saved
                env_orig_np = scene.env_origins[e].cpu().numpy().astype(np.float32)
                save_scene_pre_A(out_base, fid, env_orig_np,
                                 pre_d[e], pre_cp[e], pre_cq[e], pre_gt[e],
                                 pre_rgb=pre_rgb[e], pre_sem=pre_sem[e])

                # Per-object poses for sim-respawn (test_set_loader.py)
                slot_list = sorted(active_slot_idxs)
                slot_names = [SLOT_NAMES[i] for i in slot_list]
                type_ids   = [SLOT_TO_TYPE_ID[i] for i in slot_list]
                root_states_rel = []
                for i_slot, name in zip(slot_list, slot_names):
                    if name not in scene._rigid_objects:
                        root_states_rel.append(np.zeros(13, dtype=np.float32))
                        continue
                    st = scene[name].data.root_state_w[e].detach().cpu().numpy().astype(np.float32)
                    rel = st.copy()
                    rel[:3] = st[:3] - env_orig_np[:3]
                    root_states_rel.append(rel)
                root_states_rel = np.stack(root_states_rel, axis=0).astype(np.float32)
                scales_arr = np.array(
                    [get_active_scales(e).get(n, (1.0, 1.0, 1.0)) for n in slot_names],
                    dtype=np.float32,
                )
                np.savez_compressed(
                    os.path.join(out_base, f"{fid:09d}", "obj_poses.npz"),
                    slot_idxs       = np.array(slot_list, dtype=np.int32),
                    type_ids        = np.array(type_ids,  dtype=np.int32),
                    slot_names      = np.array(slot_names),
                    root_states_rel = root_states_rel,
                    scales_local    = scales_arr,
                    env_origin_src  = env_orig_np,
                    hidden_slots    = np.array(sorted(hidden), dtype=np.int32),
                )

                with open(os.path.join(out_base, f"{fid:09d}", "episode_log.txt"), "w") as f:
                    f.write(f"episode: {fid:09d}\n")
                    f.write(f"test_set_mode: high_occlusion\n")
                    f.write(f"active_slots: {sorted(active_slot_idxs)}\n")
                    f.write(f"hidden_slots (not visible from any of 98 views): {sorted(hidden)}\n")

                episodes_meta.append({
                    "episode_id": fid,
                    "folder": f"{fid:09d}",
                    "active_slots": sorted(active_slot_idxs),
                    "hidden_slots": sorted(hidden),
                    "active_types": sorted({SLOT_TO_TYPE_ID[i] for i in active_slot_idxs}),
                    "hidden_types": sorted({SLOT_TO_TYPE_ID[i] for i in hidden}),
                })
                saved += 1
                print(f"[TEST_SET] saved {saved}/{target}  (env={e} hidden={sorted(hidden)})  "
                      f"tries={tries} rejected={rejected}")

            if saved % max(1, num_envs) == 0 and episodes_meta:
                with open(manifest_path, "w") as f:
                    json.dump({
                        "schema_version": 1,
                        "test_set_name": os.path.basename(out_base.rstrip("/")),
                        "scene_mode": "high",
                        "n_views": N_VIEWS,
                        "n_active_per_episode": N_ACTIVE,
                        "rejection_criterion": "at_least_one_active_slot_hidden_from_all_98_views",
                        "seed": args_cli.test_set_seed,
                        "saved": saved,
                        "rejected": rejected,
                        "tries": tries,
                        "episodes": episodes_meta,
                    }, f, indent=2)

        with open(manifest_path, "w") as f:
            json.dump({
                "schema_version": 1,
                "test_set_name": os.path.basename(out_base.rstrip("/")),
                "scene_mode": "high",
                "n_views": N_VIEWS,
                "n_active_per_episode": N_ACTIVE,
                "rejection_criterion": "at_least_one_active_slot_hidden_from_all_98_views",
                "seed": args_cli.test_set_seed,
                "saved": saved,
                "rejected": rejected,
                "tries": tries,
                "episodes": episodes_meta,
            }, f, indent=2)
        print(f"[TEST_SET] DONE  saved={saved} rejected={rejected} tries={tries}  manifest={manifest_path}")
        simulation_app.close()
        return

    start_folder   = start_episode // TOTAL_STEPS
    target_folders = (num_episodes - start_episode) // TOTAL_STEPS
    target_scenes  = (target_folders + num_envs - 1) // num_envs

    print(f"[INFO] start_folder={start_folder} | target_folders={target_folders} "
          f"| target_scenes={target_scenes}")

    completed_scenes = 0

    def check_fallen():
        """Return {env_idx: [obj_<type_id>, …]} — active types that dropped below shelf.

        Same-type duplicates collapse into a single 'obj_<type_id>' entry; this
        matches the semantic_map convention (one id per type) used downstream
        by detect_tipped_ycb_v1.py.
        """
        result = {}
        for _e in range(num_envs):
            _shelf_z = float(scene.env_origins[_e, 2]) + SHELF_Y
            _fell_tids = set()
            for _n in get_active_slots(_e):
                if _n not in scene._rigid_objects: continue
                if float(scene[_n].data.root_state_w[_e, 2]) < _shelf_z - 0.10:
                    _fell_tids.add(slot_type_id(_n))
            if _fell_tids:
                result[_e] = [f"obj_{t}" for t in sorted(_fell_tids)]
        return result

    def _tilt_cos(quat_w, slot_name):
        """Cosine of the non-yaw part of the rotation between the current quat
        and the type's standing quat. Pure yaw → 1.0 (standing normally); lying
        over by more than 60 deg → < 0.5.
        yaw_quat = current * standing^{-1}. For yaw only, (qx, qy) ≈ 0.
        cos(tilt) = 1 - 2 * (qx² + qy²)  (R[2][2] of yaw_quat).
        """
        sw, sx, sy, sz = ROT_BY_ID[slot_type_id(slot_name)]
        qw, qx, qy, qz = (float(quat_w[0]), float(quat_w[1]),
                          float(quat_w[2]), float(quat_w[3]))
        # yaw_quat = current * standing^{-1};  standing^{-1} = (sw, -sx, -sy, -sz)
        e, f, g, h = sw, -sx, -sy, -sz
        yqx = qw*f + qx*e + qy*h - qz*g
        yqy = qw*g - qx*h + qy*e + qz*f
        return 1.0 - 2.0 * (yqx*yqx + yqy*yqy)

    def check_tipped():
        """Detect only non-yaw rotation (objects that really fell over)."""
        result = {}
        for _e in range(num_envs):
            _shelf_z = float(scene.env_origins[_e, 2]) + SHELF_Y
            _tipped_tids = set()
            for _n in get_active_slots(_e):
                if _n not in scene._rigid_objects: continue
                state = scene[_n].data.root_state_w[_e]
                if float(state[2]) < _shelf_z - 0.10: continue
                if _tilt_cos(state[3:7], _n) < 0.2:    # tilted >78 deg, i.e. only objects knocked flat domino-style
                    _tipped_tids.add(slot_type_id(_n))
            if _tipped_tids:
                result[_e] = [f"obj_{t}" for t in sorted(_tipped_tids)]
        return result

    while completed_scenes < target_scenes:
        # Transition step per env: one random step in each subscene.
        # transition_steps_per_env[e][s] = the sub_step of subscene s to use
        transition_steps_per_env = [
            [random.randrange(SUBSCENE_LEN) for _ in range(N_TRANSITIONS)]
            for _ in range(num_envs)
        ]

        # Push specs per env, sampled from pre_gt at the start of each subscene
        push_specs_per_env_per_subscene = [[] for _ in range(num_envs)]

        # Buffer carrying the object state / GT from one subscene to the next
        transition_obj_buf = [{} for _ in range(num_envs)]
        transition_gt_buf  = [None] * num_envs

        for s_idx in range(N_SUBSCENES):
            label      = SUBSCENE_LABELS[s_idx]
            start_step = s_idx * SUBSCENE_LEN

            # ── Build the subscene PRE state ─────────────────────────────────
            if s_idx > 0:
                # Restore the state left by the previous subscene transition step
                reset_robot_to_cap()
                for name in OBJ_NAMES:
                    if name not in scene._rigid_objects:
                        continue
                    _ns = scene[name].data.root_state_w.clone()
                    for e in range(num_envs):
                        if name in transition_obj_buf[e]:
                            _ns[e] = transition_obj_buf[e][name]
                            _ns[e, 7:] = 0.0
                    scene[name].write_root_state_to_sim(_ns)
                stabilize(50, hold_jpos=_cap_home[:, _arm_ids])   # let the transition settle

            current_pre_state = save_states()

            # Capture PRE
            reset_robot_to_cap()
            pre_d, pre_cp, pre_cq, pre_rgb, pre_sem = capture_all_viewpoints(
                scene, sim, sim_cfg, _cap_home[:, _arm_ids], _arm_ids)
            # Retry after every capture until all 11 types are mapped, then lock the map.
            if not _sem_map_built:
                if build_sem_id_map(scene["camera"]):
                    _sem_map_built = True

            if s_idx == 0:
                pre_gt = torch.stack([get_gt_id_map(scene, e) for e in range(num_envs)])
            else:
                # Reuse the GT saved at the previous subscene transition step
                pre_gt = torch.stack(transition_gt_buf)

            # Save PRE
            for e in range(num_envs):
                fid = start_folder + completed_scenes * num_envs + e
                env_orig_np = scene.env_origins[e].cpu().numpy().astype(np.float32)
                if s_idx == 0:
                    save_scene_pre_A(save_base, fid, env_orig_np,
                                     pre_d[e], pre_cp[e], pre_cq[e], pre_gt[e],
                                     pre_rgb=pre_rgb[e], pre_sem=pre_sem[e])
                else:
                    save_subscene_pre(save_base, fid, start_step, env_orig_np,
                                      pre_d[e], pre_cp[e], pre_cq[e], pre_gt[e],
                                      pre_rgb=pre_rgb[e], pre_sem=pre_sem[e])

            # Sample this subscene's push specs, per env
            for e in range(num_envs):
                push_specs_per_env_per_subscene[e].append(
                    compute_frontier_pushes(pre_gt[e]))

            # Reset the transition buffer for the next subscene
            transition_obj_buf = [{} for _ in range(num_envs)]
            transition_gt_buf  = [None] * num_envs

            # ── Push loop ─────────────────────────────────────────────────────
            for sub_step in range(SUBSCENE_LEN):
                t0 = _time.time()
                restore_states(current_pre_state)
                stabilize(30, hold_jpos=_cap_home[:, _arm_ids])   # settle right before the push

                push_specs_step = [
                    push_specs_per_env_per_subscene[e][s_idx][sub_step]
                    for e in range(num_envs)
                ]
                sv, ev, sw, ew, aw, angles_deg = get_per_env_angled_push_params(
                    push_specs_step, num_envs, scene.env_origins)
                ee_traj = apply_push_action(scene, sim, sw, ew, aw)
                stabilize(20)
                stabilize(20, hold_jpos=_cap_home[:, _arm_ids])

                global_step = start_step + sub_step

                fallen = check_fallen()
                if fallen:
                    with open(os.path.join(save_base, "fallen_log.txt"), "a") as _fl:
                        for _e, _objs in fallen.items():
                            _fid = start_folder + completed_scenes * num_envs + _e
                            _fl.write(f"episode={_fid:09d}  post_{global_step:02d}  fallen={_objs}\n")
                tipped = check_tipped()
                if tipped:
                    with open(os.path.join(save_base, "tipped_log.txt"), "a") as _tl:
                        for _e, _objs in tipped.items():
                            _fid = start_folder + completed_scenes * num_envs + _e
                            _tl.write(f"episode={_fid:09d}  post_{global_step:02d}  tipped={_objs}\n")

                post_d, _, _, post_rgb, post_sem = capture_all_viewpoints(
                    scene, sim, sim_cfg, _cap_home[:, _arm_ids], _arm_ids)
                post_gt = torch.stack([get_gt_id_map(scene, e) for e in range(num_envs)])
                swept = [make_swept_map_from_trajectory(ee_traj[e, :, :3],
                         scene.env_origins[e].cpu()) for e in range(num_envs)]
                reset_robot_to_cap()

                # Fill the transition buffer (skipped for the last subscene)
                if s_idx < N_TRANSITIONS:
                    for e in range(num_envs):
                        if sub_step == transition_steps_per_env[e][s_idx]:
                            for name in OBJ_NAMES:
                                if name in scene._rigid_objects:
                                    transition_obj_buf[e][name] = (
                                        scene[name].data.root_state_w[e].clone())
                            transition_gt_buf[e] = post_gt[e].clone()

                for e in range(num_envs):
                    fid = start_folder + completed_scenes * num_envs + e
                    env_orig_np = scene.env_origins[e].cpu().numpy().astype(np.float32)
                    save_post_step(save_base, fid, global_step, env_orig_np,
                                   post_d[e], pre_cp[e], pre_cq[e], post_gt[e],
                                   sv[e], ev[e], swept[e], ee_traj[e],
                                   push_angle_deg=float(angles_deg[e]),
                                   post_rgb=post_rgb[e], post_sem=post_sem[e])

                print(f"[{label}/{sub_step+1:02d}] scene {completed_scenes+1}/{target_scenes}"
                      f"  ({_time.time()-t0:.1f}s)")

        # ── All subscenes done → write episode_log.txt ─────────────────────
        for e in range(num_envs):
            fid = start_folder + completed_scenes * num_envs + e
            save_episode_log(save_base, fid,
                             push_specs_per_env_per_subscene[e],
                             transition_steps_per_env[e])

        completed_scenes += 1
        done_folders = start_folder + completed_scenes * num_envs
        print(f"[SCENE {completed_scenes}/{target_scenes}] "
              f"folders {done_folders - num_envs:09d}~{done_folders - 1:09d} done")

        if completed_scenes < target_scenes:
            scene.reset()
            randomize_object_positions(scene)
            reset_robot_to_cap()

    total_folders = start_folder + completed_scenes * num_envs
    print(f"[DONE] {completed_scenes} super-cycles | {total_folders} folders → {save_base}")
    simulation_app.close()


if __name__ == "__main__":
    main()
