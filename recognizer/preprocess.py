"""
Pure image preprocessing utilities for pHash-based recognition.

Public:
- preprocess_for_phash(img_bgr, out_size=256, clahe=True, blur_ksize=(3,3), crop_margin_pct=0.02, highpass=False, debug=False)
    -> gray or PreprocessResult(final, gray, clahe, blurred, highpass)
- simple dataclass PreprocessResult for optional traces
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np
from pipeline import utils

log = utils.get_logger("preprocess")

@dataclass
class PreprocessResult:
    final: np.ndarray
    gray: Optional[np.ndarray] = None
    clahe: Optional[np.ndarray] = None
    blurred: Optional[np.ndarray] = None
    highpass: Optional[np.ndarray] = None
    metadata: Optional[dict] = None

def _remove_margin(img_gray: np.ndarray, crop_margin_pct: float) -> np.ndarray:
    h, w = img_gray.shape[:2]
    m = int(min(h, w) * crop_margin_pct)
    if m > 0 and h - 2*m > 0 and w - 2*m > 0:
        return img_gray[m:h-m, m:w-m]
    return img_gray

def _maybe_highpass(gray: np.ndarray) -> np.ndarray:
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    lap = cv2.convertScaleAbs(lap)
    return cv2.addWeighted(gray, 0.9, lap, 0.1, 0)

def preprocess_for_phash(img_bgr: np.ndarray,
                         out_size: int = 256,
                         clahe: bool = True,
                         blur_ksize: Tuple[int,int] = (3,3),
                         crop_margin_pct: float = 0.02,
                         highpass: bool = False,
                         debug: bool = False):
    if img_bgr is None:
        raise ValueError("input image is None")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = _remove_margin(gray, crop_margin_pct)
    gray = cv2.resize(gray, (out_size, out_size), interpolation=cv2.INTER_AREA)

    blurred = cv2.GaussianBlur(gray, blur_ksize, 0) if blur_ksize else None
    base = blurred if blurred is not None else gray

    clahe_img = None
    if clahe:
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_img = clahe_obj.apply(base)
        base = clahe_img

    hp = _maybe_highpass(base) if highpass else None
    final = hp if highpass else base

    if debug:
        log.debug("preprocess_for_phash: out_size=%s clahe=%s highpass=%s", out_size, clahe, highpass)
        return PreprocessResult(final=final, gray=gray, clahe=clahe_img, blurred=blurred, highpass=hp,
                                metadata={"out_size": out_size, "crop_margin_pct": crop_margin_pct})
    return final
