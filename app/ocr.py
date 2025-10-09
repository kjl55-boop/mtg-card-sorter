"""
OCR helpers and title-band tight-crop logic for fallback use.
This module is optional in the hash-based pipeline but useful for diagnostics.
"""

import cv2
import numpy as np
import pytesseract

def preprocess_for_ocr(img, scale=2.0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 15, 6)
    return gray

def crop_title_band(card_img, init_top_pct=0.12, expand_px=6):
    """
    Heuristic cropping to tightly isolate the title band.
    Returns the tight crop (BGR) or the initial band if detection fails.
    """
    h, w = card_img.shape[:2]
    init_h = max(10, int(h * init_top_pct))
    band = card_img[0:init_h, :]

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.resize(gray, (gray.shape[1]*2, gray.shape[0]*2))
    gray_r = cv2.GaussianBlur(gray_r, (3,3), 0)
    th = cv2.adaptiveThreshold(gray_r, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 6)

    proj = np.sum(th > 0, axis=1)
    rows = np.where(proj > (0.05 * th.shape[1]))[0]
    if rows.size == 0:
        return band
    top_row = max(0, int(rows[0]//2) - expand_px)
    bot_row = min(init_h, int(rows[-1]//2) + expand_px)
    return card_img[top_row:bot_row, :]

def ocr_image(img, config="--oem 1 --psm 7"):
    """
    Return extracted text and a simple average confidence.
    """
    data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
    texts = []
    confs = []
    for i, txt in enumerate(data.get("text", [])):
        t = txt.strip()
        if t:
            texts.append(t)
            try:
                confs.append(int(data["conf"][i]))
            except Exception:
                pass
    text = " ".join(texts)
    avg_conf = int(sum(confs)/len(confs)) if confs else 0
    return text, avg_conf
