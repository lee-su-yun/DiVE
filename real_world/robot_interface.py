"""Abstract robot interface.

Replace `DummyRobot` with a real implementation that talks to your Piper
SDK. The pipeline only needs two methods:

    get_ee_pose() -> (4,4) T_we      # current EE pose in world frame
    move_to(T_we)                    # command the robot to this EE pose
"""
import numpy as np


class RobotInterface:
    """Minimal contract the pipeline depends on."""

    def get_ee_pose(self):
        raise NotImplementedError

    def move_to(self, T_we, wait=True):
        raise NotImplementedError


class DummyRobot(RobotInterface):
    """For dry-runs without a real robot connected.

    `get_ee_pose()` returns a configurable fixed pose; `move_to()` only
    prints the target. Use this to validate the math end-to-end before
    plugging in real hardware.
    """

    def __init__(self, T_we_init=None, verbose=True):
        if T_we_init is None:
            T_we_init = np.eye(4, dtype=np.float64)
        self._T_we = np.asarray(T_we_init, dtype=np.float64).copy()
        self._verbose = verbose

    def get_ee_pose(self):
        return self._T_we.copy()

    def move_to(self, T_we, wait=True):
        T_we = np.asarray(T_we, dtype=np.float64)
        self._T_we = T_we.copy()
        if self._verbose:
            p = T_we[:3, 3]
            print(f"  [DummyRobot] move_to xyz=({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})")
