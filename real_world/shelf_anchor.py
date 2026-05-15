"""Step 1: anchor the voxel-map (shelf) frame S in the robot world W.

One-shot procedure:
    1. Robot is held at some pose where the wrist camera can see the AprilTag.
    2. Capture an RGB frame and read the current EE pose T_we_now.
    3. Detect the tag in the image -> T_ct.
    4. Compose:
          T_wt = T_we_now @ T_ec @ T_ct          (tag in world)
          T_ws = T_wt @ T_tag_voxel              (voxel-map origin in world)

T_ws is constant for the rest of the session as long as the shelf doesn't
move.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

from apriltag_detector import estimate_T_ct, estimate_T_ct_averaged
from transform_utils import rt_to_transform, inverse_transform


def anchor_shelf(
    rgb,
    K,
    T_we_now,
    T_ec,
    T_tag_voxel,
    tag_id,
    tag_size,
    decision_margin_min=20.0,
):
    """Compute T_ws from a single RGB capture.

    Args:
        rgb: (H,W,3) uint8 RGB image showing the AprilTag.
        K:   (3,3) camera intrinsics.
        T_we_now: (4,4) current EE pose in world frame (from robot FK at
                  the moment `rgb` was captured).
        T_ec: (4,4) constant EE -> camera transform (hand-eye calibration).
        T_tag_voxel: (4,4) AprilTag -> voxel-map origin transform
                     (you measure this from where you stuck the tag).
        tag_id, tag_size, decision_margin_min: see apriltag_detector.

    Returns:
        T_ws: (4,4) voxel-map origin -> world transform.
        info: dict with intermediate transforms for debugging.
    """
    T_we_now = np.asarray(T_we_now, dtype=np.float64)
    T_ec = np.asarray(T_ec, dtype=np.float64)
    T_tag_voxel = np.asarray(T_tag_voxel, dtype=np.float64)

    T_ct, margin = estimate_T_ct(rgb, K, tag_id, tag_size, decision_margin_min)
    if T_ct is None:
        raise RuntimeError(
            f"AprilTag id={tag_id} not detected in the image. "
            "Reposition the robot so the wrist camera sees the tag, then retry."
        )

    T_wt = T_we_now @ T_ec @ T_ct
    T_ws = T_wt @ T_tag_voxel

    info = {
        "T_ct": T_ct,
        "T_wt": T_wt,
        "T_ws": T_ws,
        "decision_margin": margin,
    }
    return T_ws, info


def anchor_shelf_averaged(
    rgb_list,
    K,
    T_we_now,
    T_ec,
    T_tag_voxel,
    tag_id,
    tag_size,
    decision_margin_min=20.0,
):
    """Same as `anchor_shelf` but averages over multiple frames.

    Use this when the robot is held still and you grab N frames in a row;
    it reduces tag detection noise.  T_we_now must be the same for all
    frames (robot must not move during capture).
    """
    T_ct, margin = estimate_T_ct_averaged(
        rgb_list, K, tag_id, tag_size, decision_margin_min
    )
    if T_ct is None:
        raise RuntimeError(
            f"AprilTag id={tag_id} not detected in any of the {len(rgb_list)} frames."
        )

    T_we_now = np.asarray(T_we_now, dtype=np.float64)
    T_ec = np.asarray(T_ec, dtype=np.float64)
    T_tag_voxel = np.asarray(T_tag_voxel, dtype=np.float64)

    T_wt = T_we_now @ T_ec @ T_ct
    T_ws = T_wt @ T_tag_voxel

    info = {
        "T_ct": T_ct,
        "T_wt": T_wt,
        "T_ws": T_ws,
        "decision_margin": margin,
        "n_frames": len(rgb_list),
    }
    return T_ws, info


def anchor_shelf_multi_pose(
    captures,
    K,
    T_ec,
    T_tag_voxel,
    tag_id,
    tag_size,
    decision_margin_min=20.0,
):
    """Robust anchoring from multiple wrist-camera viewpoints.

    The wrist camera observes the same tag from N different robot poses.
    Each capture yields an independent estimate of the tag pose in world:
        T_wt_i = T_we_i @ T_ec @ T_ct_i
    We average these in SE(3) and return T_ws = mean(T_wt) @ T_tag_voxel.

    The spread between per-pose tag estimates is a quality signal: if the
    translations disagree by more than ~1-2 cm, your hand-eye T_ec or
    FK or both are off.

    Args:
        captures: list of dicts, each with keys:
            "rgb"      : (H,W,3) RGB image
            "T_we"     : (4,4) EE pose in world AT THE MOMENT of capture
        K, T_ec, T_tag_voxel, tag_id, tag_size, decision_margin_min:
            same as `anchor_shelf`.

    Returns:
        T_ws: (4,4) anchored shelf transform.
        info: dict with per-pose T_wt list and translation spread (m).
    """
    T_ec = np.asarray(T_ec, dtype=np.float64)
    T_tag_voxel = np.asarray(T_tag_voxel, dtype=np.float64)

    T_wt_list = []
    margins = []
    skipped = 0
    for i, cap in enumerate(captures):
        rgb = cap["rgb"]
        T_we_i = np.asarray(cap["T_we"], dtype=np.float64)

        T_ct, m = estimate_T_ct(rgb, K, tag_id, tag_size, decision_margin_min)
        if T_ct is None:
            skipped += 1
            continue
        T_wt_list.append(T_we_i @ T_ec @ T_ct)
        margins.append(m)

    if len(T_wt_list) < 2:
        raise RuntimeError(
            f"Need at least 2 valid tag detections, got {len(T_wt_list)} "
            f"(skipped {skipped}). Add more robot poses or improve lighting."
        )

    # Translation: arithmetic mean.  Rotation: SE3 mean via scipy.
    ts = np.array([T[:3, 3] for T in T_wt_list])
    t_mean = ts.mean(axis=0)
    t_spread = float(np.linalg.norm(ts - t_mean, axis=1).max())

    Rs = [T[:3, :3] for T in T_wt_list]
    R_mean = R.from_matrix(Rs).mean().as_matrix()

    T_wt = rt_to_transform(R_mean, t_mean)
    T_ws = T_wt @ T_tag_voxel

    info = {
        "T_wt": T_wt,
        "T_wt_per_pose": T_wt_list,
        "T_ws": T_ws,
        "decision_margins": margins,
        "translation_spread_m": t_spread,
        "n_poses_used": len(T_wt_list),
        "n_poses_skipped": skipped,
    }
    return T_ws, info
