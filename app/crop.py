"""
Crop and deskew utilities.

Provides deterministic, padding-free normalization of card images.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

# Normalized size default; kept here so this module is self-contained
NORMALIZED_SIZE: Tuple[int, int] = (400, 560)
OUT_W, OUT_H = NORMALIZED_SIZE


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts.astype("float32"), dst)
    warped = cv2.warpPerspective(image, M, (out_w, out_h))
    return warped


def _find_largest_quad(gray: np.ndarray, min_area: int = 2000) -> Optional[np.ndarray]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours[:20]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > min_area:
            return approx.reshape(4, 2)
    return None


def normalize_card_image(frame_bgr: np.ndarray, size: Tuple[int, int] = (OUT_W, OUT_H)) -> Optional[np.ndarray]:
    """
    Return a deskewed, tightly cropped BGR image sized to `size`.
    No padding is added. If detection fails, return a centered crop resized to size.
    """
    if frame_bgr is None:
        return None
    out_w, out_h = size
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    quad = _find_largest_quad(gray)
    if quad is not None:
        ordered = _order_points(quad)
        warped = _four_point_transform(frame_bgr, ordered, out_w, out_h)
        return warped
    # fallback center-crop preserving output aspect
    H, W = frame_bgr.shape[:2]
    target_ratio = out_w / out_h
    current_ratio = W / H
    if current_ratio > target_ratio:
        new_w = int(target_ratio * H)
        x0 = max(0, (W - new_w) // 2)
        crop = frame_bgr[:, x0:x0 + new_w]
    else:
        new_h = int(W / target_ratio)
        y0 = max(0, (H - new_h) // 2)
        crop = frame_bgr[y0:y0 + new_h, :]
    resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return resized
