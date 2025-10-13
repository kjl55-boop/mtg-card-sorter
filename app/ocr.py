"""
OCR helpers that depend on preprocess utilities for consistent image correction.

Public:
- crop_title_band(card_img, init_top_pct=0.12, expand_px=6) -> tight BGR crop (or fallback)
- ocr_image(img, tesseract_config="--oem 1 --psm 7") -> (text, avg_conf)
- preprocess_for_ocr is re-exported from app.preprocess.preprocess_for_ocr for convenience
"""

from typing import Tuple
import cv2
import numpy as np
import pytesseract

from .preprocess import preprocess_for_ocr

def crop_title_band(card_img: np.ndarray, init_top_pct: float = 0.12, expand_px: int = 6) -> np.ndarray:
    h, w = card_img.shape[:2]
    init_h = max(10, int(h * init_top_pct))
    band = card_img[0:init_h, :].copy()
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    # upscale slightly for detection stability
    gray_r = cv2.resize(gray, (gray.shape[1]*2, gray.shape[0]*2), interpolation=cv2.INTER_LINEAR)
    gray_r = cv2.GaussianBlur(gray_r, (3,3), 0)
    th = cv2.adaptiveThreshold(gray_r, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 6)
    proj = np.sum(th > 0, axis=1)
    rows = np.where(proj > (0.05 * th.shape[1]))[0]
    if rows.size == 0:
        return band
    top_row = max(0, int(rows[0]//2) - expand_px)
    bot_row = min(init_h, int(rows[-1]//2) + expand_px)
    return card_img[top_row:bot_row, :].copy()

def ocr_image(img, tesseract_config: str = "--oem 1 --psm 7") -> Tuple[str, int]:
    # Accept grayscale or BGR; tesseract expects proper binary/grayscale for best results
    if img is None:
        return "", 0
    gray = img
    if len(getattr(img, "shape", ())) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # minimal blurring and binarization to improve tesseract
    proc = preprocess_for_ocr(gray, scale=1.5, debug=False)
    try:
        data = pytesseract.image_to_data(proc, config=tesseract_config, output_type=pytesseract.Output.DICT)
    except Exception:
        return "", 0
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
