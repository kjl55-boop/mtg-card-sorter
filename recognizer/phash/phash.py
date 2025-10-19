# recognizer/phash/phash.py

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
from config.config import CONFIG

log = utils.get_logger("phash")

class Matcher:
    def __init__(self,
                 index_path: Optional[Path] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.index_path = index_path or CONFIG.game_profile.index_path
        self.config = config or {}
        self.phash_size = self.config.get("phash_size", CONFIG.phash_size)
        self.top_k = self.config.get("top_k", CONFIG.match_top_k)
        self.threshold = self.config.get("phash_threshold", CONFIG.phash_threshold)
        self.verify_title = self.config.get("verify_title", False)

        log.info("Matcher initialized with phash_size=%d, top_k=%d, threshold=%d", self.phash_size, self.top_k, self.threshold)
        log.info("Using index path: %s", self.index_path)

        self.index = self.load_index()
        if not self.index:
            log.warning("Phash index is empty or failed to load.")



    def load_index(self) -> Dict[str, Any]:
        try:
            with open(self.index_path, "rb") as f:
                index = pickle.load(f)
            log.info("Loaded phash index with %d entries from %s", len(index), self.index_path)
            return index
        except Exception as e:
            log.warning("Failed to load index from %s: %s", self.index_path, e)
            return {}


    def compute_phash_from_gray(self, gray: np.ndarray) -> str:
        pil_img = Image.fromarray(gray)
        return str(imagehash.phash(pil_img, hash_size=self.phash_size))

    def match_phash(self, query_phash: str) -> List[Tuple[str, Dict, int]]:
        query_hash = hex_to_hash(query_phash)
        candidates = []

        for card_id, rec in self.index.items():
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
            #log.debug("Query shape: %s, DB shape: %s, dist: %d", query_hash.hash.shape, db_hash.hash.shape, dist)
            log.debug("Candidate %s: dist=%d", card_id, dist)

            if dist <= self.threshold:
                candidates.append((card_id, rec, dist))

        candidates.sort(key=lambda x: x[2])
        return candidates[:self.top_k]

    def match_with_policy(self,
                          card_bgr: np.ndarray,
                          preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                          preprocess_kwargs: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        preprocess_kwargs = preprocess_kwargs or {}
        gray = preprocess_fn(card_bgr, **preprocess_kwargs) if preprocess_fn else cv2.cvtColor(card_bgr, cv2.COLOR_BGR2GRAY)
        qph = self.compute_phash_from_gray(gray)
        log.debug("Query phash: %s", qph)

        candidates = self.match_phash(qph)
        if not candidates:
            return None

        best_key, best_rec, best_dist = candidates[0]
        log.info("Best candidate: %s (dist=%d)", best_key, best_dist)
        title_dist = None

        if self.verify_title:
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
