"""Recover a scene's environment origin from its view-0 camera position."""
import numpy as np


def recover_env_origin(cam_pos_0: np.ndarray) -> np.ndarray:
    """
    Invert the simulator's camera placement to get the voxel-space env_origin.

    Camera placement in the simulator:
      cam_x = env_origin_x + SHELF_X_OFFSET            (xs[0] = -0.30)
      cam_y = env_origin_y + SHELF_Y_OFFSET - CAM_DIST (= env_origin_y - 0.20 - 0.50)
      cam_z = env_origin_z + SHELF_Z_OFFSET + 0.03     (= env_origin_z + 0.80 + 0.03)

    Inverse:
      env_origin_x = cam_x + 0.30
      env_origin_y = cam_y + 0.70
      env_origin_z = cam_z - 0.83
    """
    return np.array([
        cam_pos_0[0] + 0.30,
        cam_pos_0[1] + 0.70,
        cam_pos_0[2] - 0.83,
    ], dtype=np.float32)
