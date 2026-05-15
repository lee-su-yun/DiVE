"""End-to-end pipeline.

Workflow:
    1. Connect to camera + robot.
    2. Move robot to a pose where the wrist camera sees the AprilTag on the
       shelf (you can do this manually before running, or set HOME_POSE).
    3. Capture one (or several) RGB frames, run AprilTag detection, anchor
       the voxel-map frame S in the world frame W -> T_ws.
    4. Load 98 sim viewpoints from pose.npy.
    5. Compute the corresponding 98 EE poses T_we.
    6. Save them to disk.  Optionally drive the robot through them.

User inputs (edit the constants in `build_user_config` or pass via argparse):
    T_ec         : EE -> camera (constant offset from hand-eye calibration)
    T_tag_voxel  : AprilTag -> voxel-map origin (measured)
    tag_id, tag_size, voxel_size_m, pose_npy_path

채워야하는 변수들
EE_TO_CAMERA_XYZ_M	    -> ee로부터 camera까지 미터
EE_TO_CAMERA_RPY_DEG	-> ee 좌표축으로부터 camera의 각도 (degree)
TAG_TO_VOXEL_XYZ_M	    -> tag로부터 voxel map origin 미터
TAG_TO_VOXEL_RPY_DEG	-> tag로부터 voxel map origin 각도 (degree)
TAG_ID_DEFAULT	        -> tag 종류
TAG_SIZE_M_DEFAULT	    -> tag size
POSE_NPY_DEFAULT	    -> pose 경로

실행 방법
python main.py --pose-num 42
"""
import argparse
import os
import time
import numpy as np

from viewpoint_loader import load_viewpoints, summarize_viewpoints
from shelf_anchor import anchor_shelf_multi_pose
from ee_pose_solver import solve_ee_poses, save_ee_poses
from robot_interface import DummyRobot
from transform_utils import pose6d_to_transform


# ---------------------------------------------------------------------------
# USER CONFIG — edit these to match your setup.
# ---------------------------------------------------------------------------

# (1) EE -> camera transform (constant, measured by ruler / hand-eye calib).
#     Specify as 6 numbers: [tx, ty, tz, rx_deg, ry_deg, rz_deg].
#       tx/ty/tz : translation from EE origin to camera optical center (meters)
#       rx/ry/rz : Euler angles in degrees with Piper's convention
#                  R = Rz(rz) @ Ry(ry) @ Rx(rx).
#     If camera is mounted square to the EE (no rotation), set rotations = 0.
EE_TO_CAMERA_XYZ_M       = [0.00, 0.00, 0.00]    # translation (meters)
EE_TO_CAMERA_RPY_DEG     = [0.0,  0.0,  0.0]     # rotation (degrees)

def build_T_ec():
    pose6d = list(EE_TO_CAMERA_XYZ_M) + list(EE_TO_CAMERA_RPY_DEG)
    return pose6d_to_transform(pose6d, degrees=True)


# (2) AprilTag -> voxel-map origin transform.
#     "Where is the voxel-map origin in the tag frame?"
#     Same 6-number format: translation (m) + Euler (deg).
#     If your tag is stuck directly at the voxel origin with axes aligned,
#     keep all zeros.
TAG_TO_VOXEL_XYZ_M       = [0.0, 0.0, 0.0]
TAG_TO_VOXEL_RPY_DEG     = [0.0, 0.0, 0.0]

def build_T_tag_voxel():
    pose6d = list(TAG_TO_VOXEL_XYZ_M) + list(TAG_TO_VOXEL_RPY_DEG)
    return pose6d_to_transform(pose6d, degrees=True)

# (3) Tag / dataset constants
TAG_ID_DEFAULT = 0
TAG_SIZE_M_DEFAULT = 0.04        # physical edge length of the tag (meters)
VOXEL_SIZE_M_DEFAULT = 0.005     # 1 voxel = 5 mm (matches APOBU dataset)

# (4) Path to the 98-row pose.npy
POSE_NPY_DEFAULT = "/data/APOBU/beliefmap_high_occlusion_ycb_v3/000000000/push_1/pose.npy"

# (5) Where to save the resulting EE poses
OUT_NPZ_DEFAULT = "ee_poses.npz"

# (6) Anchoring: number of *robot poses* to capture, and frames per pose.
#     The wrist camera is moved to N_POSES different positions, each one
#     viewing the AprilTag from a different angle.  At each pose we grab
#     N_FRAMES_PER_POSE images (averaged internally for image-noise).
#     Translation-spread between poses is reported as a quality check.
ANCHOR_N_POSES = 3
ANCHOR_N_FRAMES_PER_POSE = 5
ANCHOR_TRANSLATION_SPREAD_WARN_M = 0.015   # warn if poses disagree by >1.5 cm

# (7) Piper-specific
PIPER_GRIPPER_DEPTH = True       # True: EE frame = gripper tip (T_we @ T_eg).
                                  # T_ec must be defined relative to the same frame.
PIPER_GO_HOME_ON_CONNECT = True   # move to init pose right after connecting
PER_MOVE_SLEEP_S = 0.5            # extra dwell between viewpoints (for capture)

# (8) Which viewpoint to move to.
#     None       : just save all 98 EE poses to file, don't move the robot.
#     int 0..97  : after solving, move the robot to ONLY this single viewpoint.
#     "all"      : move through all 98 viewpoints in order.
#     CLI override: --pose-num N  (or --pose-num all)
TARGET_POSE_NUM = 42

# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pose-npy", default=POSE_NPY_DEFAULT)
    p.add_argument("--out", default=OUT_NPZ_DEFAULT)
    p.add_argument("--tag-id", type=int, default=TAG_ID_DEFAULT)
    p.add_argument("--tag-size", type=float, default=TAG_SIZE_M_DEFAULT)
    p.add_argument("--voxel-size", type=float, default=VOXEL_SIZE_M_DEFAULT)
    p.add_argument("--anchor-poses", type=int, default=ANCHOR_N_POSES,
                   help="Number of robot poses to capture the tag from "
                        "(more = more robust, default 3)")
    p.add_argument("--anchor-frames", type=int, default=ANCHOR_N_FRAMES_PER_POSE,
                   help="Frames per pose (averaged for image noise)")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip camera/robot connection; load pre-saved rgb/intrinsics/T_we_now.")
    p.add_argument("--rgb-path", default=None,
                   help="(dry-run) path to a saved RGB image of the tag (.npy or image file).")
    p.add_argument("--K-path", default=None,
                   help="(dry-run) path to (3,3) intrinsic matrix saved as .npy.")
    p.add_argument("--T-we-now-path", default=None,
                   help="(dry-run) path to current (4,4) EE pose saved as .npy.")
    p.add_argument("--pose-num", default=str(TARGET_POSE_NUM),
                   help="Which viewpoint to drive the robot to. "
                        "Integer 0..97 for a single pose, 'all' for all 98, "
                        "or 'none' to just save the EE poses without moving.")
    p.add_argument("--no-home", action="store_true",
                   help="Don't move the robot to its init pose after connecting.")
    return p.parse_args()


def parse_pose_num(s):
    """Returns 'all', 'none', or an int index."""
    if s is None or s == "None" or s.lower() == "none":
        return "none"
    if s.lower() == "all":
        return "all"
    return int(s)


def _avg_frames(camera, n):
    """Grab n RGB frames; the AprilTag detector will average over them."""
    return [camera.get_rgb() for _ in range(max(1, n))]


def _average_rgb_for_detection(rgb_list):
    """We don't pixel-average images (changes detection); the multi-pose
    anchor just picks one frame per pose.  Use a center-of-burst frame
    instead of the first (auto-exposure may shift on the first)."""
    return rgb_list[len(rgb_list) // 2]


def collect_captures(args, robot, cam):
    """Move the wrist camera to N_POSES manually-selected positions and
    grab tag images at each.  User confirms each pose via stdin.

    Returns a list of dicts: [{"rgb": (H,W,3), "T_we": (4,4)}, ...]
    """
    captures = []
    for i in range(args.anchor_poses):
        prompt = (f"\n[anchor pose {i+1}/{args.anchor_poses}] "
                  f"Move the robot so the wrist camera sees the AprilTag "
                  f"from a *different* angle than before, then press Enter "
                  f"(or 's' to skip): ")
        s = input(prompt).strip().lower()
        if s == "s":
            print("  skipped")
            continue
        T_we_i = robot.get_ee_pose()
        frames = _avg_frames(cam, args.anchor_frames)
        rgb = _average_rgb_for_detection(frames)
        captures.append({"rgb": rgb, "T_we": T_we_i})
        print(f"  captured (T_we[:3,3] = {T_we_i[:3, 3]})")
    return captures


def get_captures_K_robot(args, T_ec):
    """Return (captures, K, robot).

    Real run: opens RealSense + Piper, interactively collects N anchor poses.
    Dry run:  loads a single pre-saved (rgb, K, T_we_now) and replicates it
              `anchor_poses` times (so the math runs but spread is zero).
    """
    if args.dry_run:
        if args.rgb_path is None or args.K_path is None or args.T_we_now_path is None:
            raise ValueError("--dry-run requires --rgb-path, --K-path, --T-we-now-path")
        if args.rgb_path.endswith(".npy"):
            rgb = np.load(args.rgb_path)
        else:
            import cv2
            rgb = cv2.cvtColor(cv2.imread(args.rgb_path), cv2.COLOR_BGR2RGB)
        K = np.load(args.K_path)
        T_we_now = np.load(args.T_we_now_path)
        robot = DummyRobot(T_we_init=T_we_now)
        # Dry-run: only one capture (no real multi-pose data available).
        captures = [{"rgb": rgb, "T_we": T_we_now}]
        return captures, K, robot

    # ---- Real hardware path ----
    from realsense_camera import RealSenseCamera
    from piper_robot import PiperRobot

    robot = PiperRobot(gripper_depth=PIPER_GRIPPER_DEPTH)
    robot.connect(go_home=(PIPER_GO_HOME_ON_CONNECT and not args.no_home))

    cam = RealSenseCamera(width=640, height=480, fps=30)
    K = cam.get_intrinsics()

    captures = collect_captures(args, robot, cam)
    cam.close()
    return captures, K, robot


def main():
    args = parse_args()

    T_ec = build_T_ec()
    T_tag_voxel = build_T_tag_voxel()

    print(f"== Sim2Real viewpoint planner ==")
    print(f"pose.npy   : {args.pose_npy}")
    print(f"tag id     : {args.tag_id}, size: {args.tag_size} m")
    print(f"voxel size : {args.voxel_size} m / voxel")
    print(f"T_ec (EE->camera): xyz_m={EE_TO_CAMERA_XYZ_M}  rpy_deg={EE_TO_CAMERA_RPY_DEG}")
    print(f"T_tag_voxel      : xyz_m={TAG_TO_VOXEL_XYZ_M}  rpy_deg={TAG_TO_VOXEL_RPY_DEG}")
    print()

    # Step 0: load viewpoints
    print("[1/4] Loading sim viewpoints")
    T_sc_list = load_viewpoints(args.pose_npy, voxel_size_m=args.voxel_size)
    summarize_viewpoints(T_sc_list)
    print()

    # Step 1: capture from multiple robot poses + anchor
    print("[2/4] Collecting tag captures from multiple robot poses")
    captures, K, robot = get_captures_K_robot(args, T_ec)
    print(f"  K =\n{K}")
    print(f"  collected {len(captures)} capture(s)")
    print()

    print("[3/4] Anchoring shelf in world via AprilTag (multi-pose average)")
    T_ws, info = anchor_shelf_multi_pose(
        captures=captures,
        K=K,
        T_ec=T_ec,
        T_tag_voxel=T_tag_voxel,
        tag_id=args.tag_id,
        tag_size=args.tag_size,
    )
    spread = info["translation_spread_m"]
    print(f"  poses used      : {info['n_poses_used']}  (skipped {info['n_poses_skipped']})")
    print(f"  margins         : {[f'{m:.1f}' for m in info['decision_margins']]}")
    print(f"  T_wt spread     : {spread*1000:.1f} mm "
          f"({'OK' if spread < ANCHOR_TRANSLATION_SPREAD_WARN_M else 'WARN — check T_ec / FK'})")
    print(f"  T_ws =\n{T_ws}")
    print()

    # Step 2: solve all EE poses
    print("[4/4] Solving 98 EE target poses")
    T_we_list = solve_ee_poses(T_ws, T_sc_list, T_ec)

    out_path = os.path.abspath(args.out)
    save_ee_poses(out_path, T_we_list, T_ws, T_ec, T_tag_voxel)
    print(f"  saved {len(T_we_list)} EE poses -> {out_path}")
    print()

    mode = parse_pose_num(args.pose_num)
    N = len(T_we_list)

    if mode == "none":
        print("(--pose-num=none) Skipping robot motion. EE poses are saved on disk.")
        return

    if mode == "all":
        print(f"== Executing all {N} viewpoints in order ==")
        input("Press Enter to begin (Ctrl+C to abort): ")
        for i, T in enumerate(T_we_list):
            pp = T[:3, 3]
            print(f"  viewpoint {i+1}/{N}  "
                  f"target xyz=({pp[0]:+.3f}, {pp[1]:+.3f}, {pp[2]:+.3f})")
            robot.move_to(T, wait=True)
            time.sleep(PER_MOVE_SLEEP_S)
        print("done.")
        return

    # Single viewpoint
    i = int(mode)
    if i < 0 or i >= N:
        raise ValueError(f"--pose-num must be in [0, {N-1}], got {i}")
    T = T_we_list[i]
    pp = T[:3, 3]
    print(f"== Moving to viewpoint {i}/{N-1} ==")
    print(f"  target xyz=({pp[0]:+.3f}, {pp[1]:+.3f}, {pp[2]:+.3f})")
    input("Press Enter to move (Ctrl+C to abort): ")
    robot.move_to(T, wait=True)
    print("done.")


if __name__ == "__main__":
    main()
