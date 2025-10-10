"""
Card detection and cropping functions.
Pure image processing functions that are easy to unit test.
"""

import cv2
import numpy as np

from .config import NORMALIZED_SIZE

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
        return rotated

    card_contour = max(contours, key=cv2.contourArea)
    x, y, w_box, h_box = cv2.boundingRect(card_contour)

    pad_x = int(w_box * pad_x_pct / 2)
    pad_y = int(h_box * pad_y_pct / 2)
    x = max(x - pad_x, 0)
    y = max(y - pad_y, 0)
    w_box = min(w_box + 2*pad_x, rotated.shape[1] - x)
    h_box = min(h_box + 2*pad_y, rotated.shape[0] - y)

    cropped = rotated[y:y+h_box, x:x+w_box].copy()
    return cropped

def extract_snippets(card_img, top_pct, mid_start_pct, mid_end_pct, bot_pct):
    """
    Return list of (label, snippet_img).
    Clamps indices to valid ranges.
    """
    h, w = card_img.shape[:2]
    top_h = max(1, int(top_pct * h))
    mid_s = int(mid_start_pct * h)
    mid_e = int(mid_end_pct * h)
    bot_y = int(bot_pct * h)

    top = card_img[0:top_h, :]
    middle = card_img[mid_s:mid_e, :]
    bottom = card_img[bot_y:, :]
    return [("Top", top), ("Middle", middle), ("Bottom", bottom)]
