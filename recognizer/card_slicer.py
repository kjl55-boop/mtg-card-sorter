#	Slice known card image into semantic regions

from dataclasses import dataclass
import cv2
import numpy as np
from typing import Dict, Optional, Tuple

@dataclass
class CropRegion:
    name: str
    x_start_pct: float
    x_end_pct: float
    y_start_pct: float
    y_end_pct: float


DEFAULT_REGIONS = {
    "title": (0.05, 0.95, 0.02, 0.12),
    "mana": (0.75, 0.98, 0.02, 0.12),
    "text": (0.05, 0.95, 0.55, 0.75),
    "bottom": (0.05, 0.95, 0.78, 0.98)
}

class CardSlicer:
    def __init__(self, region_defs: Optional[Dict[str, Tuple[float, float, float, float]]] = None):
        region_defs = region_defs or DEFAULT_REGIONS
        self.regions = {
            name: CropRegion(name, *bounds)
            for name, bounds in region_defs.items()
        }


    def crop(self, card_img: np.ndarray, region_name: str) -> np.ndarray:
        if card_img is None or region_name not in self.regions:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        region = self.regions[region_name]
        h, w = card_img.shape[:2]
        x1 = int(w * region.x_start_pct)
        x2 = int(w * region.x_end_pct)
        y1 = int(h * region.y_start_pct)
        y2 = int(h * region.y_end_pct)
        return card_img[y1:y2, x1:x2].copy()

    def crop_all(self, card_img: np.ndarray) -> Dict[str, np.ndarray]:
        return {name: self.crop(card_img, name) for name in self.regions}
