import cv2
import numpy as np
from typing import List, Tuple

def center_crop(crop: np.ndarray, target_size: int = 64) -> np.ndarray:
    """
    Centers a symbol crop in a square canvas before resizing.
    """
    h, w = crop.shape
    size = max(h, w)
    canvas = np.zeros((size, size), dtype=np.uint8)
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = crop
    return cv2.resize(canvas, (target_size, target_size), interpolation=cv2.INTER_CUBIC)


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

        if debug:
            overlay = band_img.copy()
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.imshow("Contours", overlay)


        # Filter: roughly circular, reasonable size, not too narrow
        if w > 10 and h > 10 and 0.8 < aspect < 1.2 and 100 < area < 2000:
            crop = gray[y:y+h, x:x+w]
            padded = center_crop(crop)
            symbols.append((x, padded))

            if debug:
                cv2.imshow(f"Symbol {len(symbols)}", padded)

    if debug:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Sort left to right by x position
    symbols.sort(key=lambda tup: tup[0])
    return [img for _, img in symbols]
