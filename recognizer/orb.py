import cv2
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from recognizer.phash import verify_with_orb
from recognizer.crop import crop_mana_cost

def load_symbol_db(symbol_dir: Path) -> Dict[str, np.ndarray]:
    """
    Load grayscale mana symbol images from a directory.
    Returns a dict mapping symbol ID (from filename) to image.
    """
    db = {}
    for img_path in symbol_dir.glob("*.png"):
        symbol_id = img_path.stem
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            db[symbol_id] = img
    return db

def match_mana_symbols(card_img: np.ndarray, symbol_db: Dict[str, np.ndarray],
                       min_matches: int = 8) -> List[Tuple[str, int]]:
    """
    Crop mana cost region from card image and match against reference symbols using ORB.
    Returns a list of (symbol_id, inlier_count), sorted by descending inliers.
    """
    crop = ensure_gray(crop_mana_cost(card_img))
    matches = []
    for symbol_id, symbol_img in symbol_db.items():
        symbol_img = ensure_gray(symbol_img)
        good, inliers = verify_with_orb(crop, symbol_img, min_matches=min_matches)
        if inliers >= min_matches:
            matches.append((symbol_id, inliers))
    matches.sort(key=lambda x: -x[1])
    return matches


def ensure_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
