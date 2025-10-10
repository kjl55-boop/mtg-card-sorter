import subprocess
import shutil
import tempfile
from pathlib import Path
import cv2
import os
from time import sleep

def _capture_with_libcamera_jpeg(timeout_ms=2000):
    """
    Use libcamera-jpeg to capture a single frame and return the BGR image (or None).
    Requires libcamera-apps installed and access to /dev/video* (typical on Pi OS).
    """
    if not shutil.which("libcamera-jpeg"):
        return None
    tmp = Path(tempfile.gettempdir()) / f"libcam_capture_{os.getpid()}.jpg"
    if tmp.exists():
        tmp.unlink()
    cmd = ["libcamera-jpeg", "-o", str(tmp), "-n", "--timeout", str(timeout_ms)]
    try:
        # libcamera-jpeg exits after capturing; ensure it runs quickly
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=(timeout_ms/1000.0 + 2))
        img = cv2.imread(str(tmp))
        if tmp.exists():
            try: tmp.unlink()
            except: pass
        return img
    except Exception:
        if tmp.exists():
            try: tmp.unlink()
            except: pass
        return None

def _capture_with_picamera2():
    """
    Use Picamera2 (if installed) to capture a single frame as BGR numpy array.
    Returns None if Picamera2 is not present or fails.
    """
    try:
        from picamera2 import Picamera2
        import numpy as np
    except Exception:
        return None
    try:
        pc2 = Picamera2()
        config = pc2.create_preview_configuration({"size": (1280, 720)})
        pc2.configure(config)
        pc2.start()
        # give camera a moment to warm up
        sleep(0.3)
        arr = pc2.capture_array()
        pc2.stop()
        # arr is RGB; convert to BGR for OpenCV compatibility
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return bgr
    except Exception:
        return None

def capture_frame(cam_index: int = 0, timeout: float = 2.0):
    """
    Robust capture_frame: try Picamera2, then libcamera-jpeg CLI, then OpenCV CAP_V4L2.
    Returns BGR numpy array or raises RuntimeError.
    """
    # 1) Picamera2 (preferred when available)
    img = _capture_with_picamera2()
    if img is not None:
        return img

    # 2) libcamera-jpeg CLI fallback (works for CSI cameras)
    img = _capture_with_libcamera_jpeg(int(timeout * 1000))
    if img is not None:
        return img

    # 3) OpenCV fallback (try CAP_V4L2 backend)
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open camera index {cam_index} with OpenCV")
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError("Failed to read frame from OpenCV capture")
    return frame
