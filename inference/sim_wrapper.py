"""BeliefMap SimWrapper — YCB scene (high|low occlusion).

inference-time facade on top of Isaac Lab. Implements the minimal
interface needed by inference/run_eval.py:

    sw = SimWrapper(num_envs=1, scene_mode='low')    # 'high' | 'low' | 'super_high'
    sw.reset(seed=...)                               # only 'low' ships in this release
    obs        = sw.capture_view(cam_idx)            # ViewObs
    sw.apply_push(spec)                              # freeform PushSpec
    gt_id_map  = sw.get_gt_id_map()                  # (W,H,D) int
    sw.close()

Implementation strategy
-----------------------
`inference/sim/scene_low_occlusion.py` is both the data-collection script and the
scene definition, and already contains *every* helper needed here (scene cfg
builder, randomize_object_positions, setup_semantic_labels, build_sem_id_map,
apply_sem_id_map, get_gt_id_map, get_angled_push_params, apply_push_action, …).
Rather than copy ~700 lines of those helpers into this file (a drift hazard), the
wrapper *lazy-imports* the chosen scene module after stubbing out AppLauncher and
neutralizing sys.argv, so the module-level argparse + AppLauncher block becomes a
no-op. The function and class definitions are then reused verbatim, and the scene
module itself is left untouched.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Optional

import numpy as np
import torch

# ── Path setup ───────────────────────────────────────────────────────────────
_PKG_ROOT = Path(__file__).resolve().parents[1]   # repo root
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

# ── Safe imports (no isaaclab side effects) ──────────────────────────────────
from inference.sim.voxel_map import (
    VOXEL_W, VOXEL_H, VOXEL_D, VOXEL_SIZE,
    SHELF_X_OFFSET, SHELF_Y_OFFSET, SHELF_Z_OFFSET,
)
from inference.utils.voxel_projection import (
    depth_to_height_map, height_map_to_obs, depth_semantic_to_voxel,
)

# Shelf dimensions (mirror inference/sim/scene_cfg.py)
SHELF_W = 0.60   # x
SHELF_H = 0.30   # z
SHELF_D = 0.40   # y
SHELF_Y = 0.80   # shelf surface height above env_origin


# ─── Module-level Isaac Lab handle ───────────────────────────────────────────
_simulation_app = None
_isaaclab_args  = None


def init_isaac_lab(headless: bool = True,
                   enable_cameras: bool = True,
                   extra_args: Optional[list] = None):
    """Initialize Isaac Lab AppLauncher. Call FIRST in your main script,
    before constructing SimWrapper. Idempotent."""
    global _simulation_app, _isaaclab_args
    if _simulation_app is not None:
        return _simulation_app

    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    args.headless        = headless
    args.enable_cameras  = enable_cameras

    sys.argv += [
        "--/rtx/post/dlss/enabled=false",
        "--/rtx/post/aa/op=0",
        "--/app/asyncRendering=false",
        "--/app/asyncRenderingLowLatency=false",
    ]
    if extra_args:
        sys.argv += extra_args

    app_launcher = AppLauncher(args)
    _simulation_app = app_launcher.app
    _isaaclab_args  = args
    return _simulation_app


def _require_sim_app():
    if _simulation_app is None:
        raise RuntimeError(
            "Isaac Lab is not initialized. Call sim_wrapper.init_isaac_lab(...) "
            "FIRST in your main script."
        )


# ─── Lazy-import the chosen scene module without triggering its main() ───────

_CD_CACHE: dict[str, ModuleType] = {}


def _import_collect_data(scene_mode: str) -> ModuleType:
    """Import inference.sim.scene_{high|low}_occlusion with AppLauncher stubbed
    and sys.argv neutralized so the module's top-level parser + AppLauncher
    block becomes a no-op. The scene module itself is NOT modified — we only
    intercept those two import-time side effects.
    """
    if scene_mode in _CD_CACHE:
        return _CD_CACHE[scene_mode]

    _require_sim_app()

    if scene_mode == 'high':
        modname = 'inference.sim.scene_high_occlusion'
    elif scene_mode == 'low':
        modname = 'inference.sim.scene_low_occlusion'
    elif scene_mode == 'super_high':
        modname = 'inference.sim.scene_super_high_occlusion'
    else:
        raise ValueError(f"scene_mode must be 'high' | 'low' | 'super_high' (got {scene_mode!r})")

    import importlib
    import isaaclab.app as _ila

    # Stub AppLauncher so the scene module's `app_launcher = AppLauncher(args_cli)`
    # does not re-init Omniverse.
    _real_AppLauncher = _ila.AppLauncher
    class _StubAppLauncher:                                  # noqa: N801
        def __init__(self, *args, **kwargs):
            self.app = _simulation_app
        @staticmethod
        def add_app_launcher_args(parser):
            _real_AppLauncher.add_app_launcher_args(parser)
    _ila.AppLauncher = _StubAppLauncher

    # Strip cmdline args so the scene module's argparse falls back to defaults.
    old_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]

    try:
        if modname in sys.modules:
            # already imported (e.g., from a prior SimWrapper) — reuse
            mod = sys.modules[modname]
        else:
            mod = importlib.import_module(modname)
    finally:
        _ila.AppLauncher = _real_AppLauncher
        sys.argv = old_argv

    _CD_CACHE[scene_mode] = mod
    return mod


# ─── ViewObs dataclass (output of capture_view) ──────────────────────────────
@dataclass
class ViewObs:
    occ_view_dive:     np.ndarray   # (60, 120, 80) float32 in {0,1}, DiVE format (D=z, H=x, W=y)
    sem_voxel_central: np.ndarray   # (120, 60, 80) int32, central format (W=x, H=z, D=y)
    depth:             np.ndarray   # (256, 256) float32 [m]
    rgb:               np.ndarray   # (256, 256, 3) uint8
    semantic_2d:       np.ndarray   # (256, 256) int32   (0=bg, 1..N=type_id)
    cam_pos:           np.ndarray   # (3,) world xyz
    cam_quat_wxyz:     np.ndarray   # (4,) wxyz


# ─── SimWrapper ──────────────────────────────────────────────────────────────
class SimWrapper:
    """Single-env Isaac Lab wrapper for YCB inference (high or low occlusion)."""

    def __init__(self, num_envs: int = 1, dt: float = 0.01,
                 scene_mode: str = 'high',
                 push_beyond_m: float = None):
        _require_sim_app()
        if num_envs != 1:
            raise NotImplementedError("SimWrapper supports num_envs=1 only.")
        if scene_mode not in ('high', 'low', 'super_high'):
            raise ValueError(f"scene_mode must be 'high' | 'low' | 'super_high' (got {scene_mode!r})")

        self.num_envs   = num_envs
        self.dt         = dt
        self.scene_mode = scene_mode

        # Lazy-import the appropriate scene module (function reuse path)
        self._cd = _import_collect_data(scene_mode)

        # Push length = distance travelled past the frontier surface. None keeps
        # the scene module default (PUSH_BEYOND_M = 0.15), i.e. the paper setting.
        # It is applied to BOTH the executed push and the predicted swept map so
        # the candidate representation and the physical action stay consistent.
        self.push_beyond_m = (float(push_beyond_m) if push_beyond_m is not None
                              else float(self._cd.PUSH_BEYOND_M))

        # Bring in isaaclab pieces only after init_isaac_lab()
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene
        self._sim_utils = sim_utils

        # ── sim + scene ─────────────────────────────────────────────
        # NOTE: the scene module does NOT call sim.set_camera_view(), so we
        # leave Isaac Sim's default viewport as-is to match it exactly.
        print('[SimWrapper] building sim_cfg + SimulationContext...', flush=True)
        sim_cfg = sim_utils.SimulationCfg(dt=dt, render_interval=4)
        self.sim = sim_utils.SimulationContext(sim_cfg)

        # Pick the right scene cfg class from the imported scene module
        if scene_mode == 'low':
            SceneCfg = self._cd.LowOcclusionYCBSceneCfg
        else:                       # high and super_high both export 'HighOcclusionYCBSceneCfg'
            SceneCfg = self._cd.HighOcclusionYCBSceneCfg
        print(f'[SimWrapper] scene cfg = {SceneCfg.__name__}', flush=True)
        scene_cfg = SceneCfg(num_envs=num_envs, env_spacing=2.5)
        self.scene = InteractiveScene(scene_cfg)

        self.sim.reset()
        print(f"[SimWrapper] sim + scene up. env_origins={self.scene.env_origins}", flush=True)

        # ── arm joint ids + home pose ───────────────────────────────
        robot = self.scene["robot"]
        self.arm_joint_ids, _ = robot.find_joints([
            "shoulder_pan_joint","shoulder_lift_joint","elbow_joint",
            "wrist_1_joint","wrist_2_joint","wrist_3_joint",
        ])
        device = self.scene.env_origins.device
        self.device       = device
        # Use the scene module's HOME pose constants if defined; else fall back
        home    = getattr(self._cd, 'HOME_JPOS', None)
        capture = getattr(self._cd, 'CAPTURE_JPOS', None)
        if home is None:
            home    = [1.57, -1.2, 2.2, -1.0, 1.57, 0.0]
        if capture is None:
            capture = [home[0] + math.pi] + list(home[1:])
        self.home_jpos    = torch.tensor(home,    device=device).unsqueeze(0).expand(num_envs, -1)
        self.capture_jpos = torch.tensor(capture, device=device).unsqueeze(0).expand(num_envs, -1)

        # ── viewpoint cache (uses the scene module's get_viewpoint_positions) ──
        self.view_pos_w = self._cd.get_viewpoint_positions(self.scene.env_origins)  # (E, 98, 3)
        self.shelf_center_w = self.scene.env_origins.clone()
        self.shelf_center_w[:, 2] += SHELF_Y + SHELF_H / 2
        self.env_origin_np = self.scene.env_origins[0].detach().cpu().numpy()
        self.N_VIEWS = self.view_pos_w.shape[1]    # 98

        # ── one-time YCB mesh precache (so get_gt_id_map can voxelize) ──
        # Mirrors the scene module's main() — populates
        # ycb_meshes._TYPE_SAMPLES_CACHE; without this voxelize_scene returns
        # an empty grid because every per-object loop body hits `if pts is None: continue`.
        if hasattr(self._cd, 'ycb_meshes'):
            ycb_meshes = self._cd.ycb_meshes
        else:
            from inference.sim import ycb_meshes  # noqa: PLC0415
        TYPE_BY_ID  = self._cd.TYPE_BY_ID
        USD_BY_ID   = self._cd.USD_BY_ID
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
        ycb_meshes.precache_all_types(
            (tid, USD_BY_ID[tid],
             f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/{USD_BY_ID[tid]}.usd")
            for tid in TYPE_BY_ID
        )

        # ── one-time scene physics + gripper visuals + semantic labels ──
        import omni.usd
        _stage = omni.usd.get_context().get_stage()
        if hasattr(self._cd, 'set_object_friction'):
            self._cd.set_object_friction(_stage, num_envs)
        if hasattr(self._cd, 'setup_low_com'):
            self._cd.setup_low_com(_stage, num_envs)
        if hasattr(self._cd, 'attach_gripper_visuals'):
            self._cd.attach_gripper_visuals(_stage, num_envs)
        # semantic labels: the underlying module owns _SEM_ID_MAP as module state
        slot_names = getattr(self._cd, 'SLOT_NAMES', None) or self._cd.OBJ_NAMES
        self._slot_names = list(slot_names)
        if hasattr(self._cd, 'setup_semantic_labels'):
            self._cd.setup_semantic_labels(num_envs, self._slot_names)

        # ── stabilize ───────────────────────────────────────────────
        for _ in range(20):
            robot.set_joint_position_target(self.capture_jpos, joint_ids=self.arm_joint_ids)
            self.scene.write_data_to_sim(); self.sim.step(); self.scene.update(self.dt)

    # ── reset / scene setup ───────────────────────────────────────
    def reset(self, seed: Optional[int] = None):
        """Randomize active YCB slots, settle physics, warm-up capture."""
        _require_sim_app()
        if seed is not None:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        robot = self.scene["robot"]
        # ① robot to default root state
        root_state = robot.data.default_root_state.clone()
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        robot.write_joint_state_to_sim(
            robot.data.default_joint_pos.clone(),
            robot.data.default_joint_vel.clone(),
        )

        # ② YCB slot randomization (module-level helper of the scene module)
        self._cd.randomize_object_positions(self.scene)

        # ③ stabilize a bit at capture pose
        self.scene.reset()
        for _ in range(50):
            robot.set_joint_position_target(self.capture_jpos, joint_ids=self.arm_joint_ids)
            self.scene.write_data_to_sim(); self.sim.step(); self.scene.update(self.dt)

        # ④ warm-up: first capture populates module-level _SEM_ID_MAP
        self._capture_one_view_raw(cam_idx=0)
        # try to build sem map until all types observed (best-effort)
        if hasattr(self._cd, 'build_sem_id_map'):
            for _ in range(8):
                done = self._cd.build_sem_id_map(self.scene["camera"], verbose=False)
                if done:
                    break
                self._capture_one_view_raw(cam_idx=random.randint(0, self.N_VIEWS - 1))
        print(f"[SimWrapper] reset done. (seed={seed})", flush=True)

    # ── camera capture ───────────────────────────────────────────
    def _capture_one_view_raw(self, cam_idx: int):
        """Move camera, step sim, return (depth, rgb, sem_2d, pos, quat).
        Robot is teleported to capture pose (shoulder_pan + π) before capture
        so it's out of the camera frustum."""
        cam   = self.scene["camera"]
        robot = self.scene["robot"]

        # teleport robot to capture pose (bypass dynamics)
        jpos = robot.data.joint_pos.clone()
        jpos[:, self.arm_joint_ids] = self.capture_jpos
        zero_vel = torch.zeros_like(robot.data.joint_vel)
        robot.write_joint_state_to_sim(jpos, zero_vel)

        pos_v  = self.view_pos_w[:, cam_idx, :]                        # (E, 3)
        quat_v = self._cd.look_at_quaternion(pos_v, self.shelf_center_w)
        cam.set_world_poses(positions=pos_v, orientations=quat_v)
        robot.set_joint_position_target(self.capture_jpos, joint_ids=self.arm_joint_ids)
        self.scene.write_data_to_sim(); self.sim.step(); self.scene.update(self.dt)

        raw_d = cam.data.output["distance_to_camera"]
        if raw_d.ndim == 4: raw_d = raw_d[..., 0]
        depth = raw_d[0].float().cpu().numpy()
        depth[depth > 2.0] = float('nan')

        rgb_raw = cam.data.output["rgb"][0].cpu().numpy()
        rgb = rgb_raw[..., :3].astype(np.uint8)

        sem_raw = cam.data.output["instance_id_segmentation_fast"][0].cpu()
        # build map (lazy first time)
        if hasattr(self._cd, 'build_sem_id_map'):
            self._cd.build_sem_id_map(cam, verbose=False)
        sem_2d = self._cd.apply_sem_id_map(sem_raw)

        pos_np  = pos_v[0].detach().cpu().numpy()
        quat_np = quat_v[0].detach().cpu().numpy()
        return depth, rgb, sem_2d, pos_np, quat_np

    def capture_view(self, cam_idx: int) -> ViewObs:
        """Capture 1 view at cam_idx ∈ [0, 98). Returns ViewObs."""
        _require_sim_app()
        depth, rgb, sem_2d, cam_pos, cam_quat = self._capture_one_view_raw(cam_idx)

        # depth → height_map → (H=z,W=x,D=y) occupied volume → DiVE format
        depth_obj_only = depth.copy()
        depth_obj_only[sem_2d == 0] = np.nan
        hm = depth_to_height_map(depth_obj_only, cam_pos, cam_quat, self.env_origin_np)
        occupied, _ = height_map_to_obs(hm)             # (H=60, W=120, D=80) float32
        occ_view_dive = occupied.astype(np.float32)     # DiVE: (D=z=60, H=x=120, W=y=80) ≡ (H,W,D)

        # semantic 2D → 3D voxel (central format W,H,D)
        sem_voxel_central = depth_semantic_to_voxel(
            depth, sem_2d, cam_pos, cam_quat, self.env_origin_np,
        ).astype(np.int32)

        return ViewObs(
            occ_view_dive=occ_view_dive,
            sem_voxel_central=sem_voxel_central,
            depth=depth,
            rgb=rgb,
            semantic_2d=sem_2d,
            cam_pos=cam_pos,
            cam_quat_wxyz=cam_quat,
        )

    # ── push action (freeform) ────────────────────────────────────
    def _stabilize(self, steps: int, hold_jpos=None):
        """Mirrors the scene module main()'s local `stabilize` —
        step the sim N times (optionally holding the arm at hold_jpos so it
        returns / stays out of the camera frustum)."""
        robot = self.scene["robot"]
        for _ in range(steps):
            if hold_jpos is not None:
                robot.set_joint_position_target(hold_jpos, joint_ids=self.arm_joint_ids)
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.dt)

    def push_outcome(self) -> dict:
        """Object-level outcome of the scene's current state.

        Uses exactly the criteria the data-collection pipeline applies when it
        builds `fallen_log.txt` / `tipped_log.txt`, so inference-time push
        success is measured on the same definition the training targets used:

          fallen : object centre is more than 10 cm below the shelf surface
          tipped : object is rotated more than ~78 deg away from its canonical
                   standing orientation (yaw excluded)

        Returns {'fallen': [...], 'tipped': [...]} with one entry per affected
        object type.
        """
        _require_sim_app()
        cd = self._cd
        e = 0
        shelf_z = float(self.scene.env_origins[e, 2]) + cd.SHELF_Y

        def _tilt_cos(quat_w, slot_name):
            sw_, sx, sy, sz = cd.ROT_BY_ID[cd.slot_type_id(slot_name)]
            qw, qx, qy, qz = (float(quat_w[0]), float(quat_w[1]),
                              float(quat_w[2]), float(quat_w[3]))
            ee, f, g, h = sw_, -sx, -sy, -sz
            yqx = qw * f + qx * ee + qy * h - qz * g
            yqy = qw * g - qx * h + qy * ee + qz * f
            return 1.0 - 2.0 * (yqx * yqx + yqy * yqy)

        fallen, tipped = set(), set()
        for n in cd.get_active_slots(e):
            if n not in self.scene._rigid_objects:
                continue
            state = self.scene[n].data.root_state_w[e]
            if float(state[2]) < shelf_z - 0.10:
                fallen.add(cd.slot_type_id(n))
            elif _tilt_cos(state[3:7], n) < 0.2:
                tipped.add(cd.slot_type_id(n))
        return {'fallen': sorted(fallen), 'tipped': sorted(tipped)}

    def object_positions(self) -> dict:
        """{slot_name: (x, y, z)} for every active object, in world frame.

        Sampled before and after a push so that execution outcome can be scored
        by what actually moved, rather than only by whether anything toppled.
        """
        _require_sim_app()
        cd = self._cd
        out = {}
        for n in cd.get_active_slots(0):
            if n not in self.scene._rigid_objects:
                continue
            s = self.scene[n].data.root_state_w[0]
            out[n] = (float(s[0]), float(s[1]), float(s[2]))
        return out

    def predict_swept_map_for(self, push_spec, n_steps: int = 40) -> np.ndarray:
        """Inference-time geometric swept map matching the *training distribution*.

        Training data (the scene module's main loop) builds the swept_map
        saved in action.npz via
          swept = make_swept_map_from_trajectory(ee_traj, env_origin)
        where ee_traj is the actual EE position sequence recorded during sim
        push (T≈40 steps). The voxelizer samples the full gripper volume
        (Palm 90×40 mm + Rod 30×30 mm × 40 push × 7 lat × 5 ht).

        At inference we don't have the executed ee_traj for a candidate — we
        approximate it as a linear interpolation from start_w to end_w
        (n_steps samples). The gripper-volume voxelization is identical to
        training so σ_m receives an in-distribution swept_map.

        Returns: (W=120, H=60, D=80) bool numpy array.
        """
        _require_sim_app()
        env_origin_t = self.scene.env_origins[0].cpu()
        _sv, _ev, start_w, end_w, approach_w = self._cd.get_angled_push_params(
            int(push_spec.x_voxel), int(push_spec.y_voxel),
            float(push_spec.angle_deg), env_origin=env_origin_t,
            extra_m=self.push_beyond_m,
        )
        if start_w.dtype != torch.float32:
            start_w = start_w.float(); end_w = end_w.float()

        from inference.sim.voxel_map import make_swept_map_from_trajectory  # noqa: PLC0415
        ts = torch.linspace(0.0, 1.0, n_steps).unsqueeze(1)        # (T, 1)
        ee_traj = start_w.unsqueeze(0) * (1.0 - ts) + end_w.unsqueeze(0) * ts  # (T, 3)
        swept = make_swept_map_from_trajectory(ee_traj, env_origin_t)
        return swept.cpu().numpy().astype(bool)


    def apply_push(self, push_spec, push_steps: int = 40,
                   pre_stabilize: int = 30,
                   post_settle: int = 20,
                   post_cap_settle: int = 20):
        """Execute a freeform push with the same stabilize bracket the scene
        module uses when collecting data.

        Sequence (mirrors the scene module's main loop):
            stabilize(pre_stabilize,  hold=capture_jpos)   # pre-push settle
            apply_push_action(start, end, approach)         # ee trajectory
            stabilize(post_settle)                          # free fall (objects come to rest)
            stabilize(post_cap_settle, hold=capture_jpos)   # arm back to capture pose

        Returns: (start_voxel, end_voxel, ee_traj).
        """
        _require_sim_app()
        if not hasattr(self._cd, 'get_angled_push_params') or not hasattr(self._cd, 'apply_push_action'):
            raise RuntimeError(
                f"scene module {self._cd.__name__} does not expose "
                f"get_angled_push_params / apply_push_action."
            )

        # Pre-push settle at capture pose so the arm starts pose is consistent
        if pre_stabilize > 0:
            self._stabilize(pre_stabilize, hold_jpos=self.capture_jpos)

        # Compute world-frame start/end/approach for env 0
        env_origin = self.scene.env_origins[0].cpu()
        start_voxel, end_voxel, start_w, end_w, approach_w = self._cd.get_angled_push_params(
            int(push_spec.x_voxel), int(push_spec.y_voxel), float(push_spec.angle_deg),
            env_origin=env_origin,
            extra_m=self.push_beyond_m,
        )
        # apply_push_action expects (num_envs, 3) floating-point tensors;
        # get_angled_push_params returns long when its inputs are ints.
        def _sw3(v):
            t = v.float() if v.dtype != torch.float32 else v
            return t.unsqueeze(0) if t.dim() == 1 else t
        ee_traj = self._cd.apply_push_action(
            self.scene, self.sim,
            _sw3(start_w), _sw3(end_w), _sw3(approach_w),
            push_steps=push_steps,
        )

        # Post-push settle: free fall first so objects come to rest, then
        # park the arm at capture pose so it's out of the next camera frustum.
        if post_settle > 0:
            self._stabilize(post_settle)
        if post_cap_settle > 0:
            self._stabilize(post_cap_settle, hold_jpos=self.capture_jpos)
        return start_voxel, end_voxel, ee_traj

    # ── GT id map for monitoring metric ───────────────────────────
    def get_gt_id_map(self, env_idx: int = 0, verbose: bool = False) -> torch.Tensor:
        """(W=120, H=60, D=80) int32, 0=bg, 1..N=type_id (mesh-based via ycb_meshes)."""
        _require_sim_app()
        gt = self._cd.get_gt_id_map(self.scene, env_idx)
        if verbose:
            try:
                active = self._cd.get_active_slots(env_idx)
            except Exception as e:
                active = f'<err {e}>'
            nz = int((gt > 0).sum().item())
            uniq = sorted(np.unique(gt.numpy()).tolist())
            print(f'[GT_DIAG] active_slots(env={env_idx}) (n={len(active) if hasattr(active,"__len__") else "?"}) = {active}',
                  flush=True)
            print(f'[GT_DIAG] gt shape={tuple(gt.shape)} dtype={gt.dtype} '
                  f'nonzero={nz}/{gt.numel()}  unique={uniq}', flush=True)
            # also dump the world positions of the first few active objects to spot
            # if they are inside / outside the voxel grid
            for name in (active[:5] if hasattr(active, '__getitem__') else []):
                if name in self.scene._rigid_objects:
                    p = self.scene[name].data.root_pos_w[env_idx].cpu().numpy()
                    print(f'[GT_DIAG]   {name} world pos = {p}', flush=True)
        return gt

    # ── shutdown ──────────────────────────────────────────────────
    def close(self):
        global _simulation_app
        if _simulation_app is not None:
            try:
                _simulation_app.close()
            except Exception:
                pass
            _simulation_app = None


# ─── End of file ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # quick smoke (requires Isaac Lab installed; note that only the 'low'
    # scene module ships with this release)
    init_isaac_lab(headless=True)
    sw = SimWrapper(num_envs=1, scene_mode='high')
    sw.reset(seed=0)
    obs = sw.capture_view(0)
    print(f"[smoke] view 0 captured  occ_sum={obs.occ_view_dive.sum():.0f}  "
          f"sem_unique={np.unique(obs.semantic_2d).tolist()}")
    gt = sw.get_gt_id_map()
    print(f"[smoke] gt id map shape={tuple(gt.shape)}  occ_voxels={int((gt>0).sum())}")
    sw.close()
