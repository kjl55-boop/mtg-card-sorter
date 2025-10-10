"""
Card detection and cropping functions.
Pure image processing functions that are easy to unit test.
"""

import cv2
import numpy as np

def find_card_contour(frame, min_area=5000):
    """
    Find the largest rectangular contour likely to be the card.
    Returns box (4x2 int32 points) and contour or (None, None).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    if not contours:
        return None, None
    card_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(card_contour)
    box = cv2.boxPoints(rect).astype(np.int32)
    return box, card_contour

def crop_card_from_box(frame, box, pad_x_pct=0.08, pad_y_pct=0.08):
    """
    Rotate on an expanded canvas to avoid clipping, re-detect contour and crop with padding.
    Returns cropped card BGR image.
    """
    rect = cv2.minAreaRect(box.astype(np.float32))
    center, size, angle = rect
    if angle < -45:
        angle += 90
    # force portrait orientation w <= h
    if size[0] > size[1]:
        size = (size[1], size[0])
        angle += 90

    h, w = frame.shape[:2]
    canvas = np.zeros((h*2, w*2, 3), dtype=np.uint8)
    canvas[h//2:h//2+h, w//2:w//2+w] = frame
    new_center = (w, h)

    M = cv2.getRotationMatrix2D(new_center, angle, 1.0)
    rotated = cv2.warpAffine(canvas, M, (w*2, h*2), flags=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 5000]
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
