"""
High-level orchestrator that composes crop, preprocess, phash, and OCR modules
and implements retry / fallback policy for card recognition.

Public API:
- Matcher(phash_index=None, preprocess_fn=None, ocr_fn=None, config=None)
    - match_once(card_bgr) -> MatchResult | None
    - match_with_policy(card_bgr) -> MatchResult
    - match_from_camera(camera, attempts_per_card=3, inter_frame_delay=0.08) -> MatchResult
    - reload_index()
- MatchResult dataclass: structured outcome and diagnostics
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List
from pathlib import Path
import time
import uuid
import cv2
import numpy as np

import app as app_pkg
from . import crop
# default component implementations; these modules should exist in the package
from .preprocess import preprocess_for_phash
from .phash import (
    compute_phash_from_gray,
    match_phash,
    verify_with_orb,
    load_index,
)
from .ocr import crop_title_band, ocr_image

from . import utils
log = utils.get_logger("matcher")

# Configuration defaults (override via Matcher config param)
DEFAULTS = {
    "phash_size": 32,
    "phash_threshold": 10,
    "top_k": 5,
    "orb_min_matches": 8,
    "orb_inlier_threshold": 6,
    "attempts_per_card": 3,
    "inter_frame_delay": 0.08,
    "ocr_on_failure": True,
    "ocr_threshold_margin": 6,  # run OCR if best_dist <= threshold + margin
    "save_debug_on_failure": True,
}

DEBUG_DIR = Path(app_pkg.DEBUG_DIR)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class MatchResult:
    success: bool
    id: Optional[str] = None
    dist: Optional[int] = None
    orb_matches: Optional[int] = None
    inliers: Optional[int] = None
    ocr_text: Optional[str] = None
    ocr_conf: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    debug_files: Dict[str, str] = field(default_factory=dict)
    attempts: int = 0
    elapsed: float = 0.0

class Matcher:
    """
    Compose low-level primitives into a recognition policy.
    Accepts optional injected functions/modules for easier testing.
    """

    def __init__(
        self,
        phash_index: Optional[Dict[str, Any]] = None,
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = None,
        ocr_fn: Callable[[np.ndarray], tuple] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.config = {**DEFAULTS, **(config or {})}
        self._index = phash_index or load_index(Path(app_pkg.DESCRIPTORS_DIR) / "phash_index.pkl")
        # injection points (use defaults if not provided)
        self.preprocess_fn = preprocess_fn or (lambda img, **kw: preprocess_for_phash(img, **kw))
        self.ocr_fn = ocr_fn or (lambda img: ocr_image(img))
        # cached last-match debug id for quick investigation
        self._last_debug_id = None

        log.info("Matcher initialized with phash_size=%s top_k=%s", self.config["phash_size"], self.config["top_k"])

    def reload_index(self, path: Optional[Path] = None):
        path = Path(path) if path else Path(app_pkg.DESCRIPTORS_DIR) / "phash_index.pkl"
        self._index = load_index(path)

    def _save_debug(self, img: np.ndarray, tag: str) -> str:
        path = utils.save_debug_image(img, tag=tag, directory=Path(app_pkg.DEBUG_DIR))
        if path:
            log.debug("Saved debug image %s -> %s", tag, path)
        else:
            log.warning("Failed to save debug image for %s", tag)
        return path or ""

    def match_once(self, card_bgr: np.ndarray) -> Optional[MatchResult]:
        """
        One-shot attempt: preprocess -> compute phash -> find top candidates -> (no policy retries).
        Returns MatchResult on decision (success True or False) or None if index empty.
        """
        if not self._index:
            return None

        start = time.time()
        cfg = self.config
        # preprocess to canonical gray used by phash
        gray = self.preprocess_fn(card_bgr, out_size=256, clahe=True, blur_ksize=(3,3), crop_margin_pct=0.02)
        qph = compute_phash_from_gray(gray, phash_size=cfg["phash_size"])

        candidates = match_phash(qph, self._index, top_k=cfg["top_k"], threshold=cfg["phash_threshold"])
        result = MatchResult(success=False, attempts=1, elapsed=0.0)

        if not candidates:
            result.elapsed = time.time() - start
            return result

        best_key, best_rec, best_dist = candidates[0]
        result.dist = int(best_dist)
        result.meta = best_rec.get("meta", {})

        # Accept if within threshold; optionally verify with ORB
        if best_dist <= cfg["phash_threshold"]:
            good, inls = 0, 0
            if cfg["orb_min_matches"] > 0:
                try:
                    db_path = Path(best_rec["meta"].get("path", ""))
                    db_img = cv2.imread(str(db_path)) if db_path.exists() else None
                    if db_img is not None:
                        good, inls = verify_with_orb(card_bgr, db_img, min_matches=cfg["orb_min_matches"])
                except Exception:
                    good, inls = 0, 0
            result.success = True
            result.id = best_key
            result.orb_matches = int(good)
            result.inliers = int(inls)
            result.elapsed = time.time() - start
            return result

        # Not within threshold; keep best candidate info for fallback decision
        result.elapsed = time.time() - start
        return result

    def match_with_policy(self, card_bgr: np.ndarray) -> MatchResult:
        """
        High-level policy: attempt phash matching multiple times, then optionally OCR fallback.
        Always returns a MatchResult (success True/False).
        """
        cfg = self.config
        attempts = cfg["attempts_per_card"]
        inter_delay = cfg["inter_frame_delay"]
        start_total = time.time()

        best_overall: Optional[MatchResult] = None

        for attempt in range(1, attempts + 1):
            mr = self.match_once(card_bgr)
            if mr is None:
                # index empty or other fatal; return failure result
                return MatchResult(success=False, attempts=attempt, elapsed=time.time() - start_total)
            mr.attempts = attempt
            if best_overall is None or (mr.dist is not None and (best_overall.dist is None or mr.dist < best_overall.dist)):
                best_overall = mr
            if mr.success:
                best_overall.elapsed = time.time() - start_total
                return best_overall
            # not success, wait and allow caller to provide new frame if desired
            time.sleep(inter_delay)

        # after retries, consider OCR fallback if configured and candidate near threshold+margin
        final = best_overall or MatchResult(success=False, attempts=attempts, elapsed=time.time() - start_total)
        run_ocr = cfg["ocr_on_failure"] and final.dist is not None and final.dist <= (cfg["phash_threshold"] + cfg["ocr_threshold_margin"])

        if run_ocr:
            try:
                title_crop = crop_title_band(card_bgr)
                text, conf = self.ocr_fn(title_crop)
                if text and conf > 0:
                    # simple heuristic: treat OCR confirmation as success
                    final.success = True
                    final.ocr_text = text
                    final.ocr_conf = conf
                    final.elapsed = time.time() - start_total
                    return final
            except Exception as exc:
                utils.log_exception(log, exc, "Exception while running OCR fallback")

        # save debug artifacts when configured
        if cfg["save_debug_on_failure"]:
            try:
                dbg_query = self._save_debug(card_bgr, "query")
                final.debug_files["query"] = dbg_query
                if final.meta.get("path"):
                    cand_img = cv2.imread(final.meta["path"])
                    if cand_img is not None:
                        final.debug_files["candidate"] = self._save_debug(cand_img, "candidate")
            except Exception:
                pass

        final.elapsed = time.time() - start_total
        return final

    def match_from_camera(self, camera, attempts_per_card: Optional[int] = None, inter_frame_delay: Optional[float] = None) -> MatchResult:
        """
        Convenience: read frames from a persistent camera handle and apply match_with_policy.
        'camera' must expose read(timeout) and is_open()/stop() methods (e.g., Camera class).
        This method will read a frame, run the policy, and return the MatchResult.
        """
        ap = attempts_per_card or self.config["attempts_per_card"]
        idelay = inter_frame_delay if inter_frame_delay is not None else self.config["inter_frame_delay"]

        # read one frame as starting frame; caller can manage conveyor and repeated reads externally
        frame = camera.read(timeout=1.0)
        if frame is None:
            return MatchResult(success=False, attempts=0, elapsed=0.0)
        return self.match_with_policy(frame)
