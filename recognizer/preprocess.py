"""
Pure image preprocessing utilities.

Public:
- preprocess_for_phash(img_bgr, out_size=256, clahe=True, blur_ksize=(3,3), crop_margin_pct=0.02, highpass=False, debug=False)
    -> gray or PreprocessResult(final, gray, clahe, blurred, highpass)
- preprocess_for_ocr(img_bgr_or_gray, scale=2.0, blur_ksize=(3,3), adaptive_block=31, adaptive_C=6, debug=False)
    -> binarized or PreprocessResult(final, gray, resized, binarized)
- simple dataclass PreprocessResult for optional traces
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Any
import cv2
import numpy as np
from . import utils
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
    # Convert and crop small margins
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = _remove_margin(gray, crop_margin_pct)
    # Resize to canonical square
    gray = cv2.resize(gray, (out_size, out_size), interpolation=cv2.INTER_AREA)
    blurred = None
    if blur_ksize:
        blurred = cv2.GaussianBlur(gray, blur_ksize, 0)
    base = blurred if blurred is not None else gray
    clahe_img = None
    if clahe:
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_img = clahe_obj.apply(base)
        base = clahe_img
    hp = None
    if highpass:
        hp = _maybe_highpass(base)
        base = hp
    if debug:
        log.debug("preprocess_for_phash: out_size=%s clahe=%s highpass=%s", out_size, clahe, highpass)
        return PreprocessResult(final=base, gray=gray, clahe=clahe_img, blurred=blurred, highpass=hp,
                                metadata={"out_size": out_size, "crop_margin_pct": crop_margin_pct})
    return base

def preprocess_for_ocr(img_bgr_or_gray,
                       scale: float = 2.0,
                       blur_ksize: Tuple[int,int] = (3,3),
                       adaptive_block: int = 31,
                       adaptive_C: int = 6,
                       debug: bool = False):
    # Accept BGR or already-grayscale input
    if img_bgr_or_gray is None:
        raise ValueError("input image is None")
    gray = img_bgr_or_gray
    if len(getattr(gray, "shape", ())) == 3:
        gray = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    # Upscale for small text
    resized = gray
    if scale != 1.0:
        resized = cv2.resize(gray, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    blurred = None
    if blur_ksize:
        blurred = cv2.GaussianBlur(resized, blur_ksize, 0)
    src = blurred if blurred is not None else resized
    # Adaptive threshold for OCR
    bin_img = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, adaptive_block, adaptive_C)
    if debug:
        return PreprocessResult(final=bin_img, gray=gray, blurred=blurred, metadata={"scale": scale})
    return bin_img
