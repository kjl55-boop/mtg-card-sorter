"""
Capture helper wrappers and debug saving.

Expose:
 - capture_frame(cam) -> BGR frame (wrapper around OpenCV VideoCapture)
 - normalize_and_save(frame, filename) -> saved Path
 - compute_phash_bgr(frame) -> hex string
"""

from pathlib import Path
import cv2
from PIL import Image
import imagehash
from .crop import normalize_card_image
from .config import DEBUG_DIR, NORMALIZED_SIZE
import numpy as np

DEBUG_DIR.mkdir(parents=True, exist_ok=True)
OUT_W, OUT_H = NORMALIZED_SIZE


def capture_frame(cam_index: int = 0, timeout: float = 2.0):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open camera index {cam_index}")
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError("Failed to read frame from OpenCV capture")
    return frame


def normalize_and_save(frame_bgr: np.ndarray, filename: str) -> Path:
    norm = normalize_card_image(frame_bgr, size=(OUT_W, OUT_H))
    p = Path(DEBUG_DIR) / filename
    cv2.imwrite(str(p), norm)
    return p


def compute_phash_bgr(frame_bgr: np.ndarray) -> str:
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return str(imagehash.phash(Image.fromarray(img_rgb)))
