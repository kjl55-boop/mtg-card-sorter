"""
Card detection and cropping utilities.

Public API:
- find_card_contour(frame, min_area=5000) -> (box, contour) or (None, None)
- crop_card_from_box(frame, box, pad_x_pct=0.08, pad_y_pct=0.08, return_transform=False)
    -> cropped_bgr or (cropped_bgr, transform)
- extract_snippets(card_img, top_pct, mid_start_pct, mid_end_pct, bot_pct)
    -> list of (label, snippet_img)

All functions are pure image-processing helpers and perform no I/O or hardware actions.
"""

from typing import Optional, Tuple, List
from pathlib import Path

import cv2
import numpy as np
from app import utils
log = utils.get_logger("crop")

# -----------------------
# Private helpers
# -----------------------
def _box_to_rect(box: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """Convert 4x2 box points to cv2.minAreaRect representation (center, size, angle)."""
    rect = cv2.minAreaRect(box.astype(np.float32))
    return rect  # (center (x,y), (w,h), angle)

def _normalize_angle_and_size(size: Tuple[float, float], angle: float) -> Tuple[Tuple[float, float], float]:
    """
    Normalize orientation so that returned size is (w <= h) and angle rotates to portrait.
    Returns (size, angle).
    """
    w, h = size
    if w > h:
        w, h = h, w
        angle += 90.0
    # Normalize angle to range [-90, 90)
    if angle <= -90:
        angle += 180
    if angle > 90:
        angle -= 180
    return (w, h), angle

def _create_rotated_canvas(frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Return a canvas twice the size of frame with frame centered, plus the new center coords.
    Using a larger canvas avoids clipping when rotating.
    """
    h, w = frame.shape[:2]
    canvas = np.zeros((h * 2, w * 2, 3), dtype=frame.dtype)
    canvas[h // 2:h // 2 + h, w // 2:w // 2 + w] = frame
    new_center = (w, h)
    return canvas, new_center

def _clamp_box(x: int, y: int, w: int, h: int, max_w: int, max_h: int) -> Tuple[int, int, int, int]:
    x = max(0, min(int(x), max_w - 1))
    y = max(0, min(int(y), max_h - 1))
    w = max(0, min(int(w), max_w - x))
    h = max(0, min(int(h), max_h - y))
    return x, y, w, h

# -----------------------
# Public API
# -----------------------
def find_card_contour(frame: np.ndarray, min_area: int = 5000) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Find the largest rectangular contour likely to be the card.
    Returns (box, contour) where box is 4x2 int32 points in image coords, or (None, None) on failure.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    log.debug("find_card_contour: found %d contours > min_area=%d", len(contours), min_area)
    if not contours:
        log.debug("find_card_contour: no contours above threshold")
        return None, None
    card_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(card_contour)
    box = cv2.boxPoints(rect).astype(np.int32)
    return box, card_contour

def crop_card_from_box(frame: np.ndarray, box: np.ndarray,
                       pad_x_pct: float = 0.08, pad_y_pct: float = 0.08,
                       return_transform: bool = False) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Deskew and crop a card from the frame using a 4-point box.
    - pad_x_pct/pad_y_pct are fractions of the detected box width/height to expand the crop.
    - If return_transform True, returns (cropped_image, M) where M is the 2x3 affine transform used.
    - Returns None (or (None, None) when return_transform is True) if cropping fails.
    """
    if box is None:
        return (None, None) if return_transform else None

    try:
        # Derive rectangle params and normalize orientation
        rect = _box_to_rect(box)
        (cx, cy), (w_rect, h_rect), angle = rect
        (w_rect, h_rect), angle = _normalize_angle_and_size((w_rect, h_rect), angle)

        # Build a canvas to avoid clipping during rotation
        canvas, new_center = _create_rotated_canvas(frame)
        M = cv2.getRotationMatrix2D(new_center, angle, 1.0)
        rotated = cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_CUBIC)

        # Re-detect contours on rotated canvas for robust cropping
        gray_r = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        blurred_r = cv2.GaussianBlur(gray_r, (5, 5), 0)
        edges_r = cv2.Canny(blurred_r, 50, 150)
        contours_r, _ = cv2.findContours(edges_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_r = [c for c in contours_r if cv2.contourArea(c) > 5000]
        if not contours_r:
            # fallback: crop by projecting original rect center into rotated canvas
            # compute projected center
            pts = np.array([[cx, cy]], dtype=np.float32)
            pts = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
            proj = (M @ pts.T).T
            proj_x, proj_y = int(proj[0, 0]), int(proj[0, 1])
            # use rect width/height as box
            w_box = int(w_rect)
            h_box = int(h_rect)
            x = proj_x - w_box // 2
            y = proj_y - h_box // 2
        else:
            card_contour = max(contours_r, key=cv2.contourArea)
            x, y, w_box, h_box = cv2.boundingRect(card_contour)

        # apply padding
        pad_x = int(w_box * pad_x_pct)
        pad_y = int(h_box * pad_y_pct)
        x = x - pad_x
        y = y - pad_y
        w_box = w_box + 2 * pad_x
        h_box = h_box + 2 * pad_y

        # clamp to image boundaries
        x, y, w_box, h_box = _clamp_box(x, y, w_box, h_box, rotated.shape[1], rotated.shape[0])
        if w_box <= 0 or h_box <= 0:
            return (None, None) if return_transform else None

        cropped = rotated[y:y + h_box, x:x + w_box].copy()

        if return_transform:
            return cropped, M
        return cropped
    except Exception:
        return (None, None) if return_transform else None

def extract_snippets(card_img: np.ndarray, top_pct: float, mid_start_pct: float,
                     mid_end_pct: float, bot_pct: float) -> List[Tuple[str, np.ndarray]]:
    """
    Return a list of (label, snippet_img) for Top, Middle, Bottom bands.
    Indices are clamped to valid ranges; returns empty list if input invalid.
    """
    if card_img is None:
        return []
    h, w = card_img.shape[:2]
    top_h = max(1, int(top_pct * h))
    mid_s = max(0, min(int(mid_start_pct * h), h - 1))
    mid_e = max(mid_s + 1, min(int(mid_end_pct * h), h))
    bot_y = max(0, min(int(bot_pct * h), h - 1))

    top = card_img[0:top_h, :].copy()
    middle = card_img[mid_s:mid_e, :].copy()
    bottom = card_img[bot_y:, :].copy()
    return [("Top", top), ("Middle", middle), ("Bottom", bottom)]

# -----------------------
# Simple CLI test and examples
# -----------------------
if __name__ == "__main__":
    # Quick manual smoke test if invoked directly.
    # Usage: python -m app.crop path/to/sample.jpg
    import sys
    sample = None
    if len(sys.argv) > 1:
        sample = sys.argv[1]
    if sample:
        img = cv2.imread(sample)
        box, _ = find_card_contour(img, min_area=2000)
        if box is None:
            print("No card contour found")
            exit(1)
        cropped = crop_card_from_box(img, box)
        if cropped is None:
            print("Crop failed")
            exit(2)
        cv2.imshow("Cropped", cv2.resize(cropped, (0, 0), fx=0.6, fy=0.6))
        for lbl, sn in extract_snippets(cropped, 0.12, 0.55, 0.63, 0.78):
            cv2.imshow(lbl, cv2.resize(sn, (0, 0), fx=0.6, fy=0.6))
        cv2.waitKey(0)
