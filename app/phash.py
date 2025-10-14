"""
pHash compute, index build/load, matching, and ORB verification utilities.

Public:
- compute_phash_from_gray(gray_img, phash_size=32) -> np.ndarray uint8 (phash bytes)
- hamming_distance(a, b) -> int
- build_index_from_folder(images_folder, out_path, preprocess_fn, preprocess_kwargs)
- load_index(path) -> dict
- match_phash(query_phash, index, top_k, threshold) -> list of candidates (key, meta, dist)
- verify_with_orb(img_query_bgr, img_candidate_bgr, min_matches=8) -> (good_matches, inliers)
- match_card(card_bgr, index=None, preprocess_fn=..., preprocess_kwargs=..., top_k=5, threshold=10)
"""

import cv2
import numpy as np
import pickle
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List, Tuple
from . import utils

#log config
log = utils.get_logger("phash")
INDEX_PATH = Path("data/descriptors/phash_index.pkl")

# Configurable defaults (move to config.py if you prefer)
DEFAULT_PHASH_SIZE = 32
DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 10
DEFAULT_ORB_MIN_MATCHES = 8

from PIL import Image
import imagehash

def compute_phash_from_gray(gray, phash_size=32):
    pil_img = Image.fromarray(gray)
    return str(imagehash.phash(pil_img, hash_size=phash_size))
'''
def compute_phash_from_gray(gray_img: np.ndarray, phash_size: int = DEFAULT_PHASH_SIZE) -> np.ndarray:
    if gray_img is None:
        raise ValueError("gray image is None")
    ph = cv2.img_hash.PHash_create(hash_size=phash_size)
    h = ph.compute(gray_img)
    if h is None:
        raise RuntimeError("phash computation failed")
    arr = np.asarray(h).flatten().astype(np.uint8)
    return arr
'''

'''
'''

def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    a = np.asarray(a, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8)
    if a.shape != b.shape:
        raise ValueError("hash shapes differ")
    return int(np.unpackbits(np.bitwise_xor(a, b)).sum())
    
def _serialize_phash(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def _deserialize_phash(b: bytes, length: int) -> np.ndarray:
    return np.frombuffer(b, dtype=np.uint8)[:length]

def save_index(index: dict, path: Path = INDEX_PATH):
    ok = utils.safe_pickle_save(path, index)
    if ok:
        log.info("Saved phash index to %s", path)
    else:
        log.error("Failed to save phash index to %s", path)
    return ok

def load_index(path: Path = INDEX_PATH) -> dict:
    idx = utils.safe_pickle_load(path)
    if idx is None:
        log.info("No phash index at %s (returning empty index)", path)
        return {}
    log.info("Loaded phash index with %d entries from %s", len(idx), path)
    return idx

def build_index_from_folder(images_folder: Path,
                            out_path: Path,
                            preprocess_fn: Callable[[np.ndarray], np.ndarray],
                            preprocess_kwargs: Optional[dict] = None,
                            phash_size: int = DEFAULT_PHASH_SIZE) -> dict:
    preprocess_kwargs = preprocess_kwargs or {}
    images_folder = Path(images_folder)
    index = {}
    for img_path in images_folder.glob("*.png"):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            gray = preprocess_fn(img, **preprocess_kwargs) if preprocess_fn else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ph = compute_phash_from_gray(gray, phash_size=phash_size)
            idx = img_path.stem
            index[idx] = {"phash": _serialize_phash(ph), "phash_len": ph.size, "meta": {"path": str(img_path)}}
        except Exception:
            # keep building; caller may log exceptions
            continue
    save_index(index, Path(out_path))
    return index

def match_phash(query_phash, index, top_k=5, threshold=10):
    from imagehash import hex_to_hash

    query_hash = hex_to_hash(query_phash)
    candidates = []

    for card_id, rec in index.items():
        db_phash = rec["phash"]
        dist = query_hash - hex_to_hash(db_phash)
        if dist <= threshold:
            candidates.append((card_id, rec, dist))

    candidates.sort(key=lambda x: x[2])
    return candidates[:top_k]

'''
def match_phash(query_phash: np.ndarray,
                index: dict,
                top_k: int = DEFAULT_TOP_K,
                threshold: int = DEFAULT_THRESHOLD) -> List[Tuple[str, dict, int]]:
    if not index:
        return []
    candidates = []
    for key, rec in index.items():
        db_ph = _deserialize_phash(rec["phash"], rec["phash_len"])
        dist = hamming_distance(query_phash, db_ph)
        candidates.append((key, rec, int(dist)))
    candidates.sort(key=lambda x: x[2])
    return candidates[:top_k]
'''
    
def verify_with_orb(img1_bgr: np.ndarray, img2_bgr: np.ndarray, min_matches: int = DEFAULT_ORB_MIN_MATCHES) -> Tuple[int, int]:
    orb = cv2.ORB_create(2000)
    gray1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    if des1 is None or des2 is None:
        return 0, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for mn in matches:
        if len(mn) != 2:
            continue
        m, n = mn
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < min_matches:
        return len(good), 0
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    try:
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is None:
            return len(good), 0
        inliers = int(mask.sum())
        return len(good), inliers
    except Exception:
        return len(good), 0

def match_card(card_bgr: np.ndarray,
               index: Optional[dict] = None,
               preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
               preprocess_kwargs: Optional[dict] = None,
               top_k: int = DEFAULT_TOP_K,
               threshold: int = DEFAULT_THRESHOLD,
               verify_orb: bool = True):
    preprocess_kwargs = preprocess_kwargs or {}
    if index is None:
        raise ValueError("index must be provided")
    gray = preprocess_fn(card_bgr, **preprocess_kwargs) if preprocess_fn else cv2.cvtColor(card_bgr, cv2.COLOR_BGR2GRAY)
    qph = compute_phash_from_gray(gray)
    candidates = match_phash(qph, index, top_k=top_k, threshold=threshold)
    if not candidates:
        return None
    best_key, best_rec, best_dist = candidates[0]
    if best_dist <= threshold:
        good, inliers = 0, 0
        if verify_orb:
            try:
                db_path = Path(best_rec["meta"]["path"])
                db_img = cv2.imread(str(db_path)) if db_path.exists() else None
                if db_img is not None:
                    good, inliers = verify_with_orb(card_bgr, db_img)
            except Exception:
                good, inliers = 0, 0
        return {"id": best_key, "meta": best_rec.get("meta", {}), "dist": int(best_dist), "orb_matches": int(good), "inliers": int(inliers)}
    return None
