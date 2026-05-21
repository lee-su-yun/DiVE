"""Real Piper robot wrapper that implements RobotInterface.

Wraps `piper_sdk` and exposes:
    get_ee_pose()      -> (4,4) T_we in world frame
    move_to(T_we)      -> command the robot to that EE pose
    connect() / disconnect()
    open_gripper() / close_gripper()
    go_to_init_pose()

Convention:
    The Piper SDK reports pose as [x_m, y_m, z_m, rx_rad, ry_rad, rz_rad]
    with R = Rz(rz) @ Ry(ry) @ Rx(rx).  Internally it's stored as
    fixed-point integers (micrometers and milli-degrees) — handled here.

    If `gripper_depth=True` (default), the returned/commanded pose is the
    gripper-tip frame G = E translated by [0, 0, 0.136] m (along the EE
    Z axis).  Make sure your hand-eye T_ec is calibrated relative to the
    same frame.
"""
import time
import numpy as np

from transform_utils import (
    pose6d_to_transform,
    transform_to_pose6d,
    inverse_transform,
    rt_to_transform,
)
from robot_interface import RobotInterface


class PiperRobot(RobotInterface):
    # Default "home" pose: matches ABD's `_init_pose`
    # (raw Piper fixed-point: [x_um, y_um, z_um, rx_mdeg, ry_mdeg, rz_mdeg, gripper])
    DEFAULT_INIT_POSE_RAW = [50000, 0, 350000, 0, 120000, 0, 0]

    # Gripper-tip offset from the joint flange (in EE frame).
    T_EG_DEFAULT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0.136],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    GRIPPER_OPEN = 70000
    GRIPPER_CLOSE = 42000

    def __init__(
        self,
        gripper_depth=True,
        gripper_effort=1000,
        move_settle_s=0.2,
        T_eg=None,
        init_pose_raw=None,
    ):
        """
        Args:
            gripper_depth: If True, get_ee_pose() / move_to() use the
                gripper-tip frame (T_we @ T_eg).  If False, use the flange.
            gripper_effort: passed to piper.GripperCtrl (1000 = pick_place).
            move_settle_s: sleep after each move command.
            T_eg: optional override for the gripper-tip offset (4x4).
            init_pose_raw: optional override for the "home" raw pose list.
        """
        # Lazy import so the rest of the pipeline runs without piper_sdk.
        from piper_sdk import C_PiperInterface_V2, C_PiperForwardKinematics

        self._piper = C_PiperInterface_V2()
        self._fk_interface = C_PiperForwardKinematics()
        self._gripper_depth = bool(gripper_depth)
        self._gripper_effort = int(gripper_effort)
        self._move_settle_s = float(move_settle_s)
        self._T_eg = np.asarray(T_eg if T_eg is not None else self.T_EG_DEFAULT,
                                dtype=np.float64)
        self._init_pose_raw = list(init_pose_raw if init_pose_raw is not None
                                   else self.DEFAULT_INIT_POSE_RAW)
        self._connected = False

    # ---------------- connection ----------------

    def connect(self, go_home=True):
        self._piper.ConnectPort()
        while not self._piper.EnablePiper():
            time.sleep(0.01)
        self._connected = True
        if go_home:
            self.go_to_init_pose()

    def disconnect(self):
        # piper_sdk doesn't expose a clean disconnect; flag only.
        self._connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.disconnect()

    # ---------------- pose I/O ----------------

    def get_ee_pose(self):
        """Current EE pose as a 4x4 in world (robot base) frame."""
        raw = self._piper.GetArmEndPoseMsgs().end_pose
        pose6d = [
            raw.X_axis / 1_000_000.0,
            raw.Y_axis / 1_000_000.0,
            raw.Z_axis / 1_000_000.0,
            np.deg2rad(raw.RX_axis / 1000.0),
            np.deg2rad(raw.RY_axis / 1000.0),
            np.deg2rad(raw.RZ_axis / 1000.0),
        ]
        T_flange = pose6d_to_transform(pose6d)
        if self._gripper_depth:
            return T_flange @ self._T_eg
        return T_flange

    def move_to(self, T_we, wait=True, gripper_val=None):
        """Command the robot to reach the given EE pose.

        Args:
            T_we: (4,4) target EE pose in world frame.  Interpreted as the
                  gripper-tip pose if `gripper_depth=True`, else the flange.
            wait: sleep after sending the command.
            gripper_val: optional gripper opening (0..70000). If None, the
                  current gripper value is preserved.
        """
        T_we = np.asarray(T_we, dtype=np.float64)
        if T_we.shape != (4, 4):
            raise ValueError(f"T_we must be (4,4), got {T_we.shape}")

        # If pose is in gripper-tip frame, back out to flange frame.
        if self._gripper_depth:
            T_flange = T_we @ inverse_transform(self._T_eg)
        else:
            T_flange = T_we

        pose6d = transform_to_pose6d(T_flange)
        # convert to Piper fixed-point ints
        x_um = int(round(pose6d[0] * 1_000_000))
        y_um = int(round(pose6d[1] * 1_000_000))
        z_um = int(round(pose6d[2] * 1_000_000))
        rx_mdeg = int(round(np.rad2deg(pose6d[3]) * 1000))
        ry_mdeg = int(round(np.rad2deg(pose6d[4]) * 1000))
        rz_mdeg = int(round(np.rad2deg(pose6d[5]) * 1000))

        if gripper_val is None:
            gripper_val = int(self.get_gripper())
        else:
            gripper_val = int(gripper_val)

        self._control_end_pose(
            [x_um, y_um, z_um, rx_mdeg, ry_mdeg, rz_mdeg, gripper_val]
        )
        if wait:
            time.sleep(self._move_settle_s)

    def step(self, pose6d_raw, gripper):
        """ABD-style raw-int step. pose6d_raw = [x_um, y_um, z_um,
        rx_mdeg, ry_mdeg, rz_mdeg]. gripper in [0, 70000]."""
        end_pose = [int(x) for x in pose6d_raw] + [int(gripper)]
        self._control_end_pose(end_pose)
        time.sleep(self._move_settle_s)

    def _control_end_pose(self, end_pose):
        """Single end-pose + gripper control cycle (matches ABD's piper.py)."""
        self._piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self._piper.EndPoseCtrl(end_pose[0], end_pose[1], end_pose[2],
                                end_pose[3], end_pose[4], end_pose[5])
        self._piper.GripperCtrl(end_pose[6], self._gripper_effort, 0x01, 0)
        time.sleep(0.01)

    # ---------------- convenience ----------------

    def get_joints_rad(self):
        j = self._piper.GetArmJointMsgs().joint_state
        return np.array([
            np.deg2rad(j.joint_1 / 1000.0),
            np.deg2rad(j.joint_2 / 1000.0),
            np.deg2rad(j.joint_3 / 1000.0),
            np.deg2rad(j.joint_4 / 1000.0),
            np.deg2rad(j.joint_5 / 1000.0),
            np.deg2rad(j.joint_6 / 1000.0),
        ])

    def get_gripper(self):
        g = self._piper.GetArmGripperMsgs().gripper_state.grippers_angle
        return int(np.clip(g, 0, 70000))

    def open_gripper(self):
        self._piper.GripperCtrl(self.GRIPPER_OPEN, 300, 0x01, 0)

    def close_gripper(self):
        self._piper.GripperCtrl(self.GRIPPER_CLOSE, 300, 0x01, 0)

    def go_to_init_pose(self):
        self._control_end_pose(self._init_pose_raw)
        time.sleep(self._move_settle_s)
        self.open_gripper()

    def reset_joints(self):
        """Move all joints to zero (use with caution — large motion)."""
        joints = [0, 0, 0, 0, 0, 0, 0]
        self._piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        self._piper.JointCtrl(joints[0], joints[1], joints[2],
                              joints[3], joints[4], joints[5])
        self._piper.GripperCtrl(joints[6], 300, 0x01, 0)
        time.sleep(0.005)
