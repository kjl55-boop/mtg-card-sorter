"""
Camera interface module.
Provides a simple abstraction for Picamera2 (and fallback to OpenCV VideoCapture for desktop testing).
"""

import cv2
from typing import Any, Tuple
from pathlib import Path
from app import config, utils

try:
    from picamera2 import Picamera2
    from libcamera import controls
    HAVE_PICAM = True
except Exception:
    HAVE_PICAM = False

def init_camera(preview_size=None):
    """Return a camera handle. Use Picamera2 when available, otherwise fallback to cv2.VideoCapture(0)."""
    preview_size = preview_size or config.CAMERA_PREVIEW_SIZE
    if HAVE_PICAM:
        cam = Picamera2()
        config_obj = cam.create_preview_configuration(main={"size": preview_size})
        cam.configure(config_obj)
        try:
            cam.set_controls({
                "AfMode": controls.AfModeEnum.Continuous,
                "AwbEnable": True
            })
        except Exception:
            pass
        cam.start()
        return ("picamera2", cam)
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, preview_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, preview_size[1])
        return ("opencv", cap)

def grab_frame(cam_handle):
    """Return a BGR numpy array frame from camera handle returned by init_camera."""
    kind, cam = cam_handle
    if kind == "picamera2":
        arr = cam.capture_array()
        # Picamera2 returns RGB ndarray
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        ret, frame = cam.read()
        if not ret:
            raise RuntimeError("Failed to read frame from OpenCV capture")
        return frame

def close_camera(cam_handle):
    kind, cam = cam_handle
    if kind == "picamera2":
        try:
            cam.stop()
        except Exception:
            pass
    else:
        cam.release()
