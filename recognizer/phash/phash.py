# recognizer/phash/phash.py
"""
Matcher for pHash-based recognition.
Uses recognizer.preprocess.preprocess_for_phash by default to ensure parity with indexer.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Callable, Optional, Dict, List, Tuple, Any
from PIL import Image
import pickle
from imagehash import hex_to_hash
import imagehash
from datetime import datetime

from pipeline import utils
from recognizer.card_slicer import CardSlicer
from recognizer.phash.phash_tools import PhashComparator
from recognizer.preprocess import preprocess_for_phash, PreprocessResult
from config.config import CONFIG

log = utils.get_logger("phash")

class Matcher:
    def __init__(self,
                 index_path: Optional[Path] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.index_path = index_path or CONFIG.game_profile.index_path
        self.config = config or {}

        # phash parameters (priority: provided config -> CONFIG -> loader defaults)
        self.phash_size = int(self.config.get("phash_size", getattr(CONFIG, "phash_size", 16)))
        self.top_k = int(self.config.get("top_k", getattr(CONFIG, "phash_top_k", getattr(CONFIG, "match_top_k", 12))))
        self.threshold = int(self.config.get("phash_threshold", getattr(CONFIG, "phash_threshold", 16)))
        self.verify_title = bool(self.config.get("verify_title", False))

        # preprocessing defaults (merge game_profile.preprocess_config and loader-level CONFIG)
        gp = getattr(CONFIG, "game_profile", None)
        gp_pp = gp.preprocess_config if gp and hasattr(gp, "preprocess_config") else {}
        self.pp_defaults = {
            "out_size": int(self.config.get("phash_out_size", getattr(CONFIG, "phash_out_size", gp_pp.get("out_size", 256)))),
            "clahe": bool(self.config.get("phash_clahe", getattr(CONFIG, "phash_clahe", gp_pp.get("clahe", True)))),
            "blur_ksize": tuple(self.config.get("phash_blur_ksize", getattr(CONFIG, "phash_blur_ksize", gp_pp.get("blur_ksize", (3,3))))),
            "crop_margin_pct": float(self.config.get("phash_crop_margin_pct", getattr(CONFIG, "phash_crop_margin_pct", gp_pp.get("crop_margin_pct", 0.02)))),
            "highpass": bool(self.config.get("phash_highpass", getattr(CONFIG, "phash_highpass", gp_pp.get("highpass", False)))),
            "debug": False
        }

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

    def save_debug(self, img: np.ndarray, tag: str) -> Path:
        out_dir = Path(CONFIG.logs_dir) / "diagnostics" / "debug_images"
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tag}.png"
        out_path = out_dir / filename
        cv2.imwrite(str(out_path), img)
        return out_path

    def compute_phash_from_gray(self, gray: np.ndarray) -> str:
        if gray is None:
            raise ValueError("compute_phash_from_gray: input is None")
        # ensure uint8
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(gray)
        return str(imagehash.phash(pil_img, hash_size=self.phash_size))

    def match_phash(self, query_phash: str) -> List[Tuple[str, Dict, int]]:
        query_hash = hex_to_hash(query_phash)
        candidates: List[Tuple[str, Dict, int]] = []

        for card_id, rec in self.index.items():
            db_phash = rec.get("phash")
            if not db_phash:
                continue
            try:
                db_hash = hex_to_hash(db_phash)
            except Exception as e:
                log.debug("Failed to parse phash for %s: %s", card_id, e)
                continue

            # if shapes differ, skip (likely different phash_size)
            if query_hash.hash.shape != db_hash.hash.shape:
                log.debug("Shape mismatch for %s: query %s vs db %s", card_id, query_hash.hash.shape, db_hash.hash.shape)
                continue

            dist = int(query_hash - db_hash)
            candidates.append((card_id, rec, dist))

        candidates.sort(key=lambda x: x[2])
        return candidates[:self.top_k]

    def match_with_policy(self,
                        card_bgr: np.ndarray,
                        preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                        preprocess_kwargs: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        preprocess_kwargs = preprocess_kwargs or {}

        # build effective preprocess function if none provided
        if preprocess_fn is None:
            merged = dict(self.pp_defaults)
            # allow call-time overrides
            merged.update(preprocess_kwargs)
            preprocess_fn = lambda im, **kw: preprocess_for_phash(im, **merged)

        try:
            pp_out = preprocess_fn(card_bgr, **(preprocess_kwargs or {}))
            gray_img = pp_out.final if hasattr(pp_out, "final") else pp_out
        except Exception as e:
            log.warning("Preprocessing failed: %s — falling back to cvtColor gray", e)
            gray_img = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2GRAY)

        qph = self.compute_phash_from_gray(gray_img)
        log.debug("Query phash: %s", qph)

        candidates = self.match_phash(qph)
        if not candidates:
            return None

        best_key, best_rec, best_dist = candidates[0]
        log.info("Best candidate: %s (dist=%d)", best_key, best_dist)

        # Ensure variable always defined
        title_dist = None

        if self.verify_title:
            try:
                db_path = Path(best_rec.get("meta", {}).get("path", ""))
                db_img = cv2.imread(str(db_path)) if db_path.exists() else None
                if db_img is not None:
                    slicer = CardSlicer()
                    cmp = PhashComparator(hash_size=self.phash_size)
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


    '''
       candidates = self.match_phash(qph)
        if not candidates:
            return None

        best_key, best_rec, best_dist = candidates[0]
        log.info("Best candidate: %s (dist=%d)", best_key, best_dist)
        title_dist = None

        if self.verify_title:
            try:
                db_path = Path(best_rec.get("meta", {}).get("path", ""))
                db_img = cv2.imread(str(db_path)) if db_path.exists() else None
                if db_img is not None:
                    slicer = CardSlicer()
                    cmp = PhashComparator(hash_size=self.phash_size)
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
        '''
