"""Minimal RealSense RGB camera wrapper.

If pyrealsense2 isn't installed or no device is connected, the rest of the
pipeline still works — you can feed any (rgb, K) into shelf_anchor.py.
"""
import numpy as np

try:
    import pyrealsense2 as rs
    _RS_AVAILABLE = True
except ImportError:
    _RS_AVAILABLE = False


class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30, serial=None):
        if not _RS_AVAILABLE:
            raise RuntimeError(
                "pyrealsense2 not installed. `pip install pyrealsense2` "
                "or supply your own (rgb, K) directly."
            )
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        if serial is not None:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        profile = self._pipeline.start(cfg)

        color_profile = profile.get_stream(rs.stream.color)
        intr = color_profile.as_video_stream_profile().get_intrinsics()
        self._K = np.array([
            [intr.fx, 0.0,     intr.ppx],
            [0.0,     intr.fy, intr.ppy],
            [0.0,     0.0,     1.0],
        ], dtype=np.float64)
        self._width = width
        self._height = height

        # Throw away the first few frames so auto-exposure stabilizes.
        for _ in range(10):
            self._pipeline.wait_for_frames()

    def get_intrinsics(self):
        return self._K.copy()

    def get_rgb(self):
        frames = self._pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("Failed to grab color frame from RealSense")
        rgb = np.asanyarray(color.get_data()).copy()  # (H,W,3) uint8 RGB
        return rgb

    def close(self):
        try:
            self._pipeline.stop()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
