"""
pHash compute, index build/load, matching, and ORB verification utilities.

Public:
- compute_phash_from_gray(gray_img, phash_size=8) -> str (hex string)
- match_phash(query_phash, index, top_k, threshold) -> list of candidates (key, meta, dist)
- verify_with_orb(img_query_bgr, img_candidate_bgr, min_matches=8) -> (good_matches, inliers)
- match_card(card_bgr, index=None, preprocess_fn=..., preprocess_kwargs=..., top_k=5, threshold=10)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Callable, Optional, Dict, List, Tuple, Any
from PIL import Image
import imagehash
import pickle
from imagehash import hex_to_hash
from . import utils

# Logging
log = utils.get_logger("phash")
INDEX_PATH = Path("data/descriptors/phash_index.pkl")

# Configurable defaults
DEFAULT_PHASH_SIZE = 8
DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 10
DEFAULT_ORB_MIN_MATCHES = 8

# --- Core pHash logic ---

def load_index(path: Path = INDEX_PATH) -> dict:
    """Load phash index from pickle file."""
    try:
        with open(path, "rb") as f:
            index = pickle.load(f)
        log.info("Loaded phash index with %d entries from %s", len(index), path)
        return index
    except Exception as e:
        log.warning("Failed to load index from %s: %s", path, e)
        return {}

def compute_phash_from_gray(gray, phash_size=DEFAULT_PHASH_SIZE) -> str:
    """Compute perceptual hash from grayscale image and return as hex string."""
    pil_img = Image.fromarray(gray)
    return str(imagehash.phash(pil_img, hash_size=phash_size))

def match_phash(query_phash: str,
                index: Dict[str, Dict],
                top_k: int = DEFAULT_TOP_K,
                threshold: int = DEFAULT_THRESHOLD) -> List[Tuple[str, Dict, int]]:
    """Match query phash (hex string) against index of hex strings."""
    query_hash = hex_to_hash(query_phash)
    candidates = []
    log.debug("Query shape: %s, DB shape: %s", query_hash.hash.shape, db_hash.hash.shape)
    for card_id, rec in index.items():
        db_phash = rec["phash"]
        db_hash = hex_to_hash(db_phash)
        if query_hash.hash.shape != db_hash.hash.shape:
            log.warning("Shape mismatch: query %s vs db %s", query_hash.hash.shape, db_hash.hash.shape)
            continue
        dist = query_hash - db_hash
        if dist <= threshold:
            candidates.append((card_id, rec, dist))

    candidates.sort(key=lambda x: x[2])
    return candidates[:top_k]

# --- ORB verification ---

def verify_with_orb(img1_bgr: np.ndarray,
                    img2_bgr: np.ndarray,
                    min_matches: int = DEFAULT_ORB_MIN_MATCHES) -> Tuple[int, int]:
    orb = cv2.ORB_create(2000)
    gray1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)
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

# --- Full match pipeline ---

def match_card(card_bgr: np.ndarray,
               index: Optional[Dict[str, Dict]] = None,
               preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
               preprocess_kwargs: Optional[Dict] = None,
               top_k: int = DEFAULT_TOP_K,
               threshold: int = DEFAULT_THRESHOLD,
               verify_orb: bool = True) -> Optional[Dict[str, Any]]:
    if index is None:
        raise ValueError("index must be provided")
    preprocess_kwargs = preprocess_kwargs or {}
    gray = preprocess_fn(card_bgr, **preprocess_kwargs) if preprocess_fn else cv2.cvtColor(card_bgr, cv2.COLOR_BGR2GRAY)
    qph = compute_phash_from_gray(gray)
    log.debug("Query phash: %s", qph)
    candidates = match_phash(qph, index, top_k=top_k, threshold=threshold)
    if not candidates:
        return None
    best_key, best_rec, best_dist = candidates[0]
    good, inliers = 0, 0
    if verify_orb:
        try:
            db_path = Path(best_rec["meta"]["path"])
            db_img = cv2.imread(str(db_path)) if db_path.exists() else None
            if db_img is not None:
                good, inliers = verify_with_orb(card_bgr, db_img)
        except Exception:
            pass
    return {
        "id": best_key,
        "meta": best_rec.get("meta", {}),
        "dist": int(best_dist),
        "orb_matches": int(good),
        "inliers": int(inliers)
    }
