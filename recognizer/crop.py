"""
Card detection and cropping utilities.

Public API:
- find_card_contour(frame, min_area=5000) -> (box, contour) or (None, None)
- crop_card_from_box(frame, box, pad_x_pct=0.08, pad_y_pct=0.08, return_transform=False)
    -> cropped_bgr or (cropped_bgr, transform)
- extract_snippets(card_img) -> list of (label, snippet_img)

All functions are pure image-processing helpers and perform no I/O or hardware actions.
"""

from typing import Optional, Tuple, List
import cv2
import numpy as np
from pipeline import utils
from recognizer.card_slicer import CardSlicer
from config.config import CONFIG

log = utils.get_logger("crop")

# -----------------------
# Private helpers
# -----------------------
def _box_to_rect(box: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    rect = cv2.minAreaRect(box.astype(np.float32))
    return rect

def _normalize_angle_and_size(size: Tuple[float, float], angle: float) -> Tuple[Tuple[float, float], float]:
    w, h = size
    if w > h:
        w, h = h, w
        angle += 90.0
    if angle <= -90:
        angle += 180
    if angle > 90:
        angle -= 180
    return (w, h), angle

def _create_rotated_canvas(frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
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
    if box is None:
        return (None, None) if return_transform else None

    try:
        rect = _box_to_rect(box)
        (cx, cy), (w_rect, h_rect), angle = rect
        (w_rect, h_rect), angle = _normalize_angle_and_size((w_rect, h_rect), angle)

        canvas, new_center = _create_rotated_canvas(frame)
        M = cv2.getRotationMatrix2D(new_center, angle, 1.0)
        rotated = cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_CUBIC)

        gray_r = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        blurred_r = cv2.GaussianBlur(gray_r, (5, 5), 0)
        edges_r = cv2.Canny(blurred_r, 50, 150)
        contours_r, _ = cv2.findContours(edges_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_r = [c for c in contours_r if cv2.contourArea(c) > 5000]

        if not contours_r:
            pts = np.array([[cx, cy]], dtype=np.float32)
            pts = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
            proj = (M @ pts.T).T
            proj_x, proj_y = int(proj[0, 0]), int(proj[0, 1])
            w_box = int(w_rect)
            h_box = int(h_rect)
            x = proj_x - w_box // 2
            y = proj_y - h_box // 2
        else:
            card_contour = max(contours_r, key=cv2.contourArea)
            x, y, w_box, h_box = cv2.boundingRect(card_contour)

        pad_x = int(w_box * pad_x_pct)
        pad_y = int(h_box * pad_y_pct)
        x = x - pad_x
        y = y - pad_y
        w_box = w_box + 2 * pad_x
        h_box = h_box + 2 * pad_y

        x, y, w_box, h_box = _clamp_box(x, y, w_box, h_box, rotated.shape[1], rotated.shape[0])
        if w_box <= 0 or h_box <= 0:
            return (None, None) if return_transform else None

        cropped = rotated[y:y + h_box, x:x + w_box].copy()
        return (cropped, M) if return_transform else cropped

    except Exception:
        return (None, None) if return_transform else None

def slice_regions(card_img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Legacy wrapper for region slicing. Uses CardSlicer internally.
    """
    if card_img is None:
        return []
    slicer = CardSlicer(region_defs=CONFIG.game_profile.crop_regions)
    return [(name, crop) for name, crop in slicer.crop_all(card_img).items()]

def detect_orientation(card: np.ndarray) -> str:
    regions = slice_regions(card)
    title = next((r for l, r in regions if l == "title"), None)
    text = next((r for l, r in regions if l == "text"), None)

    if title is None or text is None:
        return "unknown"

    title_density = cv2.countNonZero(cv2.cvtColor(title, cv2.COLOR_BGR2GRAY))
    text_density = cv2.countNonZero(cv2.cvtColor(text, cv2.COLOR_BGR2GRAY))

    return "upside_down" if title_density < text_density else "upright"
