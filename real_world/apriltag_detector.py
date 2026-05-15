"""AprilTag detection: image -> T_ct (tag in camera frame).

Wraps pupil_apriltags. Tag family fixed to tag36h11.
"""
import numpy as np
import cv2
from pupil_apriltags import Detector

from transform_utils import rt_to_transform


_DETECTOR = Detector(
    families="tag36h11",
    nthreads=2,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25,
)


def estimate_T_ct(
    rgb,
    K,
    tag_id,
    tag_size,
    decision_margin_min=20.0,
):
    """Detect a single AprilTag and return its pose in the camera frame.

    Args:
        rgb: (H,W,3) uint8 RGB image (undistorted preferred).
        K:   (3,3) camera intrinsic matrix.
        tag_id: int, the AprilTag ID to look for.
        tag_size: physical edge length of the tag in meters.
        decision_margin_min: detection quality threshold; raise to reject
            blurry/oblique detections, lower if you miss good ones.

    Returns:
        T_ct: (4,4) transform — points in tag frame become camera-frame
              points via p_cam = T_ct @ p_tag.  None if no valid detection.
        margin: float decision margin of the chosen detection, or None.
    """
    rgb = np.asarray(rgb)
    K = np.asarray(K, dtype=np.float64)

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must be (H,W,3), got shape {rgb.shape}")

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    dets = _DETECTOR.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(fx, fy, cx, cy),
        tag_size=float(tag_size),
    )

    best = None
    best_margin = -1.0
    for d in dets:
        if int(d.tag_id) != int(tag_id):
            continue
        margin = float(getattr(d, "decision_margin", 0.0))
        if margin < decision_margin_min:
            continue
        if margin > best_margin:
            best = d
            best_margin = margin

    if best is None:
        return None, None

    R_ct = np.asarray(best.pose_R, dtype=np.float64)
    t_ct = np.asarray(best.pose_t, dtype=np.float64).reshape(3)
    T_ct = rt_to_transform(R_ct, t_ct)
    return T_ct, best_margin


def estimate_T_ct_averaged(
    rgb_list,
    K,
    tag_id,
    tag_size,
    decision_margin_min=20.0,
):
    """Average multiple detections of the same tag for a more stable pose.

    Useful at startup: capture N frames while everything is still and
    average the resulting tag-in-camera poses.
    """
    from scipy.spatial.transform import Rotation as R

    Rs, ts, margins = [], [], []
    for rgb in rgb_list:
        T_ct, m = estimate_T_ct(rgb, K, tag_id, tag_size, decision_margin_min)
        if T_ct is None:
            continue
        Rs.append(T_ct[:3, :3])
        ts.append(T_ct[:3, 3])
        margins.append(m)

    if not Rs:
        return None, None

    R_mean = R.from_matrix(Rs).mean().as_matrix()
    t_mean = np.mean(np.stack(ts, axis=0), axis=0)
    T_ct = rt_to_transform(R_mean, t_mean)
    return T_ct, float(np.mean(margins))
