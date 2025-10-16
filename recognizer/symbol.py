import cv2
import numpy as np
from typing import List

def isolate_mana_symbols(band_img: np.ndarray, debug: bool = False) -> List[np.ndarray]:
    """
    Detect and extract individual mana symbols from a cropped mana cost band.
    Returns a list of grayscale symbol crops.
    """
    gray = cv2.cvtColor(band_img, cv2.COLOR_BGR2GRAY) if band_img.ndim == 3 else band_img
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    symbol_crops = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h
        area = cv2.contourArea(cnt)

        # Filter: roughly circular, reasonable size
        if 0.8 < aspect < 1.2 and 100 < area < 2000:
            crop = gray[y:y+h, x:x+w]
            resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_CUBIC)
            symbol_crops.append(resized)

            if debug:
                cv2.imshow(f"Symbol {len(symbol_crops)}", resized)

    if debug:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Sort left to right
    symbol_crops.sort(key=lambda img: cv2.boundingRect(cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0])[0])
    return symbol_crops
