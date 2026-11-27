"""
BeliefMap Simulation Scene

Shelf: 60cm(W) x 30cm(H) x 40cm(D), surface at y=0.8m in world
Camera: movable in x (left-right) and y (up-down), fixed z distance
UR5: for pushing objects on the shelf
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

import os as _os
_SIM_DIR     = _os.path.dirname(_os.path.abspath(__file__))
_GRIPPER_USD = _os.path.join(_SIM_DIR, "assets", "ur5_with_gripper.usd")
UR5_USD_PATH = _os.environ.get("BELIEFMAP_UR5_USD", _GRIPPER_USD)
assert _os.path.isfile(UR5_USD_PATH), f"UR5 USD not found: {UR5_USD_PATH}"

# Shelf dimensions (meters)
SHELF_W = 0.60   # x  (width)
SHELF_H = 0.30   # z  (height)
SHELF_D = 0.40   # y  (depth)
SHELF_Y = 0.80   # shelf surface height in world (per env origin)

# ── UR5 Config ──────────────────────────────────────────────────────────────
UR5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=UR5_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, -0.90, SHELF_Y),
        joint_pos={
            "shoulder_pan_joint":   1.57,
            "shoulder_lift_joint": -1.2,
            "elbow_joint":          2.2,
            "wrist_1_joint":       -1.0,
            "wrist_2_joint":        1.57,
            "wrist_3_joint":        0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)


# ── Scene Config ────────────────────────────────────────────────────────────
@configclass
class BeliefMapSceneCfgNoObj(InteractiveSceneCfg):
    """BeliefMap scene with no objects of its own.

    Contains only ground, light, the shelf panels, the robot and the camera.
    Scene configs that spawn YCB assets subclass this so that they control the
    full object set instead of inheriting one.
    """
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.CuboidCfg(
            size=(20.0, 20.0, 0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)),
    )
    shelf_base = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/shelf_base",
        spawn=sim_utils.CuboidCfg(
            size=(SHELF_W, SHELF_D, 0.01),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.35, 0.15)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, SHELF_Y)),
    )
    shelf_back = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/shelf_back",
        spawn=sim_utils.CuboidCfg(
            size=(SHELF_W, 0.01, SHELF_H),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.35, 0.15)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, SHELF_D / 2, SHELF_Y + SHELF_H / 2)),
    )
    shelf_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/shelf_left",
        spawn=sim_utils.CuboidCfg(
            size=(0.01, SHELF_D, SHELF_H),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.35, 0.15)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=3.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-SHELF_W / 2, 0.0, SHELF_Y + SHELF_H / 2)),
    )
    shelf_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/shelf_right",
        spawn=sim_utils.CuboidCfg(
            size=(0.01, SHELF_D, SHELF_H),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.35, 0.15)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=3.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(SHELF_W / 2, 0.0, SHELF_Y + SHELF_H / 2)),
    )
    shelf_top = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/shelf_top",
        spawn=sim_utils.CuboidCfg(
            size=(SHELF_W, SHELF_D, 0.01),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.35, 0.15)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, SHELF_Y + SHELF_H)),
    )
    robot: ArticulationCfg = UR5_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/camera",
        update_period=0,
        height=256,
        width=256,
        # instance_id_segmentation_fast: raw pixel = stage-wide unique instance id.
        # info.idToLabels = {pixel_value: prim_path}. The obj_<i> in prim_path
        # maps straight to a type_id through SLOT_TO_TYPE_ID, so no separate
        # class label is needed.
        data_types=["rgb", "distance_to_camera", "instance_id_segmentation_fast"],
        colorize_instance_id_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, -1.0, SHELF_Y + SHELF_H / 2),
            rot=(0.707, 0.0, -0.707, 0.0),
            convention="world",
        ),
    )

