"""
High-level orchestrator that composes crop, preprocess, and phash modules
to implement card recognition with optional region verification.


Public API:
- Matcher(phash_index=None, preprocess_fn=None, config=None)
    - match_once(card_bgr) -> MatchResult | None
    - match_with_policy(card_bgr) -> MatchResult
    - match_from_camera(camera, attempts_per_card=3, inter_frame_delay=0.08) -> MatchResult
    - reload_index()
- MatchResult dataclass: structured outcome and diagnostics
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any
from pathlib import Path
import time
import cv2
import numpy as np

from config.config import CONFIG
from . import crop
from .preprocess import preprocess_for_phash
from .phash.phash import compute_phash_from_gray, match_phash, load_index
from pipeline import utils

log = utils.get_logger("matcher")

DEFAULTS = {
    "phash_size": 8,
    "phash_threshold": 10,
    "top_k": 5,
    "attempts_per_card": 3,
    "inter_frame_delay": 0.08,
    "save_debug_on_failure": True,
}

DEBUG_DIR = Path(CONFIG.debug_dir)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)
profile = CONFIG.game_profile

@dataclass
class MatchResult:
    success: bool
    id: Optional[str] = None
    dist: Optional[int] = None
    phash: Optional[str] = None  # Add this
    meta: Dict[str, Any] = field(default_factory=dict)
    debug_files: Dict[str, str] = field(default_factory=dict)
    attempts: int = 0
    elapsed: float = 0.0


class Matcher:
    def __init__(
        self,
        phash_index: Optional[Dict[str, Any]] = None,
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.config = {**DEFAULTS, **(config or {})}
        self._index = phash_index or load_index(Path(profile.index_path) / "phash_index.pkl")
        self.preprocess_fn = preprocess_fn or (lambda img, **kw: preprocess_for_phash(img, **kw))
        self._last_debug_id = None

        log.info("Matcher initialized with phash_size=%s top_k=%s", self.config["phash_size"], self.config["top_k"])

    def reload_index(self, path: Optional[Path] = None):
        path = Path(path) if path else Path(profile.index_path) / "phash_index.pkl"
        self._index = load_index(path)

    def _save_debug(self, img: np.ndarray, tag: str) -> str:
        path = utils.save_debug_image(img, tag=tag, directory=Path(CONFIG.debug))
        if path:
            log.debug("Saved debug image %s -> %s", tag, path)
        else:
            log.warning("Failed to save debug image for %s", tag)
        return path or ""

    def match_once(self, card_bgr: np.ndarray) -> Optional[MatchResult]:
        if not self._index:
            return None

        start = time.time()
        cfg = self.config
        gray = self.preprocess_fn(card_bgr, out_size=256, clahe=True, blur_ksize=(3,3), crop_margin_pct=0.02)
        qph = compute_phash_from_gray(gray, phash_size=cfg["phash_size"])
        log.debug("Query phash: %s", qph)
        log.debug("Matching against %d index entries", len(self._index))
        result.phash = qph

        candidates = match_phash(qph, self._index, top_k=cfg["top_k"], threshold=cfg["phash_threshold"])
        for candidate in candidates:
            key, rec, dist = candidate
            log.debug("Candidate %s: dist=%d", key, dist)
        result = MatchResult(success=False, attempts=1, elapsed=0.0)

        if not candidates:
            result.elapsed = time.time() - start
            return result

        best_key, best_rec, best_dist = candidates[0]
        result.dist = int(best_dist)
        result.meta = best_rec.get("meta", {})

        if best_dist <= cfg["phash_threshold"]:
            result.success = True
            result.id = best_key
            result.elapsed = time.time() - start
            return result

        result.elapsed = time.time() - start
        return result

    def match_with_policy(self, card_bgr: np.ndarray) -> MatchResult:
        cfg = self.config
        attempts = cfg["attempts_per_card"]
        inter_delay = cfg["inter_frame_delay"]
        start_total = time.time()

        best_overall: Optional[MatchResult] = None

        for attempt in range(1, attempts + 1):
            mr = self.match_once(card_bgr)
            if mr is None:
                return MatchResult(success=False, attempts=attempt, elapsed=time.time() - start_total)
            mr.attempts = attempt
            if best_overall is None or (mr.dist is not None and (best_overall.dist is None or mr.dist < best_overall.dist)):
                best_overall = mr
            if mr.success:
                best_overall.elapsed = time.time() - start_total
                return best_overall
            time.sleep(inter_delay)

        final = best_overall or MatchResult(success=False, attempts=attempts, elapsed=time.time() - start_total)

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
        ap = attempts_per_card or self.config["attempts_per_card"]
        idelay = inter_frame_delay if inter_frame_delay is not None else self.config["inter_frame_delay"]
        frame = camera.read(timeout=1.0)
        if frame is None:
            return MatchResult(success=False, attempts=0, elapsed=0.0)
        return self.match_with_policy(frame)
