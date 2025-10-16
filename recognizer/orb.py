import cv2
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from recognizer.phash import verify_with_orb
from recognizer.crop import crop_mana_cost
from config.config import CONFIG

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
        good, inliers = verify_symbol_orb(crop, symbol_img, min_matches=min_matches)
        if inliers >= min_matches:
            matches.append((symbol_id, inliers))
    matches.sort(key=lambda x: -x[1])
    return matches

def verify_symbol_orb(img1: np.ndarray,
                    img2: np.ndarray,
                    min_matches: int = 8) -> Tuple[int, int]:
    def ensure_gray(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    gray1 = ensure_gray(img1)
    gray2 = ensure_gray(img2)

    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    if des1 is None or des2 is None:
        return 0, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if len([m, n]) == 2 and m.distance < 0.75 * n.distance]
    if len(good) < min_matches:
        return len(good), 0
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    try:
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return len(good), int(mask.sum()) if mask is not None else 0
    except Exception:
        return len(good), 0


def ensure_gray(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale only if image has 3 channels."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
