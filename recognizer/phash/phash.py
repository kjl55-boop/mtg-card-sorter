"""
pHash compute, index build/load, matching, and optional region verification utilities.

Public:
- compute_phash_from_gray(gray_img, phash_size=8) -> str (hex string)
- match_phash(query_phash, index, top_k, threshold) -> list of candidates (key, meta, dist)
- match_card(card_bgr, index=None, preprocess_fn=..., preprocess_kwargs=..., top_k=5, threshold=10, verify_title=False) -> dict
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Callable, Optional, Dict, List, Tuple, Any
from PIL import Image
import imagehash
import pickle
from imagehash import hex_to_hash
from pipeline import utils
from recognizer.card_slicer import CardSlicer
from recognizer.phash.phash_tools import PhashComparator

# Logging
log = utils.get_logger("phash")
INDEX_PATH = Path("data/descriptors/phash_index.pkl")

# Configurable defaults
DEFAULT_PHASH_SIZE = 8
DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 10

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
    query_hash = hex_to_hash(query_phash)
    candidates = []

    for card_id, rec in index.items():
        db_phash = rec.get("phash")
        try:
            db_hash = hex_to_hash(db_phash)
        except Exception as e:
            log.warning("Failed to parse phash for %s: %s", card_id, e)
            continue

        if query_hash.hash.shape != db_hash.hash.shape:
            log.warning("Shape mismatch: query %s vs db %s", query_hash.hash.shape, db_hash.hash.shape)
            continue

        dist = query_hash - db_hash
        log.debug("Query shape: %s, DB shape: %s, dist: %d", query_hash.hash.shape, db_hash.hash.shape, dist)

        if dist <= threshold:
            candidates.append((card_id, rec, dist))

    candidates.sort(key=lambda x: x[2])
    return candidates[:top_k]

# --- Full match pipeline with optional title verification ---

def match_card(card_bgr: np.ndarray,
               index: Optional[Dict[str, Dict]] = None,
               preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
               preprocess_kwargs: Optional[Dict] = None,
               top_k: int = DEFAULT_TOP_K,
               threshold: int = DEFAULT_THRESHOLD,
               verify_title: bool = False) -> Optional[Dict[str, Any]]:
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
    title_dist = None

    if verify_title:
        try:
            db_path = Path(best_rec["meta"].get("path", ""))
            db_img = cv2.imread(str(db_path)) if db_path.exists() else None
            if db_img is not None:
                slicer = CardSlicer()
                cmp = PhashComparator()
                query_title = slicer.crop(card_bgr, "title")
                db_title = slicer.crop(db_img, "title")
                title_dist = cmp.compare(query_title, db_title)
        except Exception as e:
            log.warning("Title verification failed for %s: %s", best_key, e)

    return {
        "id": best_key,
        "meta": best_rec.get("meta", {}),
        "dist": int(best_dist),
        "title_dist": title_dist
    }
