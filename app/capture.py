"""
app.capture

Robust frame capture (Picamera2, libcamera-jpeg, OpenCV fallback)
and helpers to normalize/save debug crops and compute phash.

Exports:
 - capture_frame(cam_index=0, timeout=2.0) -> BGR numpy array
 - normalize_and_save(frame_bgr, filename) -> pathlib.Path
 - compute_phash_bgr(frame_bgr) -> hex phash string
"""
from pathlib import Path
import tempfile
import shutil
import subprocess
import os
from time import sleep

import cv2
from PIL import Image
import imagehash
import numpy as np

from .crop import normalize_card_image
from .config import DEBUG_DIR, NORMALIZED_SIZE

DEBUG_DIR.mkdir(parents=True, exist_ok=True)
OUT_W, OUT_H = NORMALIZED_SIZE


# --- Pi / libcamera capture helpers ------------------------------------------------

def _capture_with_picamera2():
    """
    Try Picamera2 (if installed). Returns BGR numpy array or None.
    """
    try:
        from picamera2 import Picamera2
    except Exception:
        return None
    try:
        pc2 = Picamera2()
        # preview config is usually fine; size can be adjusted
        config = pc2.create_preview_configuration({"size": (1280, 720)})
        pc2.configure(config)
        pc2.start()
        sleep(0.25)
        arr = pc2.capture_array()
        pc2.stop()
        # Picamera2 returns RGB; convert to BGR
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return bgr
    except Exception:
        return None


def _capture_with_libcamera_jpeg(timeout_ms=2000):
    """
    Use libcamera-jpeg CLI to capture a single frame. Returns BGR numpy array or None.
    Requires libcamera-jpeg installed and in PATH.
    """
    if not shutil.which("libcamera-jpeg"):
        return None
    tmp = Path(tempfile.gettempdir()) / f"libcam_capture_{os.getpid()}.jpg"
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass
    cmd = ["libcamera-jpeg", "-o", str(tmp), "-n", "--timeout", str(timeout_ms)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=(timeout_ms / 1000.0 + 3))
        img = cv2.imread(str(tmp))
        try:
            tmp.unlink()
        except Exception:
            pass
        return img
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
        return None


# --- OpenCV fallback ----------------------------------------------------------------

def _capture_with_opencv(cam_index=0, timeout=2.0, backend=cv2.CAP_V4L2):
    """
    Attempt OpenCV VideoCapture using the requested backend.
    Returns BGR array or None.
    """
    cap = cv2.VideoCapture(cam_index, backend)
    try:
        if not cap.isOpened():
            return None
        # optional warm-up: read a couple frames
        for _ in range(2):
            ret, frame = cap.read()
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        return frame
    finally:
        cap.release()


# --- Public API --------------------------------------------------------------------

def capture_frame(cam_index: int = 0, timeout: float = 2.0):
    """
    Robust capture function:
     1) Picamera2 (if available)
     2) libcamera-jpeg CLI (if available)
     3) OpenCV VideoCapture (CAP_V4L2)
    Returns BGR numpy array or raises RuntimeError.
    """
    # 1) Picamera2
    img = _capture_with_picamera2()
    if img is not None:
        return img

    # 2) libcamera-jpeg CLI
    img = _capture_with_libcamera_jpeg(int(timeout * 1000))
    if img is not None:
        return img

    # 3) OpenCV fallback
    img = _capture_with_opencv(cam_index, timeout=timeout, backend=cv2.CAP_V4L2)
    if img is not None:
        return img

    # Try other OpenCV backends as last resort
    for be in (cv2.CAP_GSTREAMER, cv2.CAP_FFMPEG, cv2.CAP_ANY):
        try:
            img = _capture_with_opencv(cam_index, timeout=timeout, backend=be)
            if img is not None:
                return img
        except Exception:
            continue

    raise RuntimeError("Failed to read frame from any available capture backend")


def normalize_and_save(frame_bgr: np.ndarray, filename: str) -> Path:
    """
    Normalize (deskew + crop to card), save to DEBUG_DIR and return the Path.
    Uses normalize_card_image from app.crop which produces a BGR image sized to NORMALIZED_SIZE.
    """
    norm = normalize_card_image(frame_bgr, size=(OUT_W, OUT_H))
    p = Path(DEBUG_DIR) / filename
    # write with OpenCV; ensure parent exists
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), norm)
    return p


def compute_phash_bgr(frame_bgr: np.ndarray) -> str:
    """
    Compute perceptual hash (imagehash.phash) from an OpenCV BGR image and return hex string.
    """
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    return str(imagehash.phash(pil))
