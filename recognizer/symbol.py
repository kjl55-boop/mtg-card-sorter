import cv2
import numpy as np
from typing import List, Tuple

def pad_crop(crop: np.ndarray, target_size: int = 64) -> np.ndarray:
    """
    Pads a symbol crop to target_size x target_size, centering the original.
    """
    h, w = crop.shape
    pad_y = max(0, (target_size - h) // 2)
    pad_x = max(0, (target_size - w) // 2)
    padded = cv2.copyMakeBorder(crop, pad_y, pad_y, pad_x, pad_x,
                                borderType=cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

def isolate_mana_symbols(band_img: np.ndarray, debug: bool = False) -> List[np.ndarray]:
    """
    Detect and extract individual mana symbols from a cropped mana cost band.
    Returns a list of padded grayscale symbol crops.
    """
    gray = cv2.cvtColor(band_img, cv2.COLOR_BGR2GRAY) if band_img.ndim == 3 else band_img
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    symbols: List[Tuple[int, np.ndarray]] = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h
        area = cv2.contourArea(cnt)

        # Filter: roughly circular, reasonable size
        if 0.8 < aspect < 1.2 and 100 < area < 2000:
            crop = gray[y:y+h, x:x+w]
            padded = pad_crop(crop)
            symbols.append((x, padded))

            if debug:
                cv2.imshow(f"Symbol {len(symbols)}", padded)

    if debug:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Sort left to right by x position
    symbols.sort(key=lambda tup: tup[0])
    return [img for _, img in symbols]
