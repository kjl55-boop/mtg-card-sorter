"""
Small utility helpers: image conversions, safe save, logging helper.
"""

import cv2
import os
from pathlib import Path
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("card_inspector")

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)

def pil_to_bgr(pil_img):
    """Convert PIL Image to OpenCV BGR numpy array."""
    arr = np.array(pil_img)
    # PIL uses RGB
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def safe_imwrite(path, img):
    p = Path(path)
    ensure_dir(p.parent)
    cv2.imwrite(str(p), img)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))
