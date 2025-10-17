# pipeline/camera/api.py

from pathlib import Path
import numpy as np
from config.config import CONFIG
from pipeline import utils
from .core import Camera, CameraError
from typing import Optional

log = utils.get_logger("camera")

def capture_frame(cam_index: int = 0, timeout: float = 2.0) -> np.ndarray:
    """
    Capture a single frame using the best available backend.
    Automatically starts and stops the camera.
    """
    cam = Camera(preview_size=CONFIG.preview_size, warmup_sec=CONFIG.camera_warmup_sec, use_background_thread=False, cam_index=cam_index)
    cam.start()
    try:
        img = cam.read(timeout=timeout)
        if img is None:
            raise RuntimeError("capture_frame: no frame captured")
        return img
    finally:
        cam.stop()

def init_camera(preview_size: Optional[tuple] = None) -> Camera:
    c = Camera(preview_size=preview_size or CONFIG.preview_size, warmup_sec=CONFIG.camera_warmup_sec, use_background_thread=True)
    c.start()
    return c

def grab_frame(cam: Camera, timeout: float = 1.0) -> Optional[np.ndarray]:
    try:
        return cam.read(timeout=timeout)
    except Exception as exc:
        utils.log_exception(log, exc, "grab_frame failed")
        return None

def close_camera(cam: Camera) -> None:
    try:
        cam.stop()
    except Exception as exc:
        utils.log_exception(log, exc, "close_camera failed")
