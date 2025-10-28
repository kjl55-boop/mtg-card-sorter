#!/usr/bin/env python3
"""
scripts/controller.py

Small controller API for UI and scripts to:
- capture a single card image
- run matcher on the capture (with retries)
- save capture + metadata to disk (simple file-backed store)
- list recent captures
- set accept/reject on capture records

This intentionally keeps the persistence simple (filesystem + JSON) so you can
iterate quickly. Replace persistence with tools/store.py (SQLite) later if desired.
"""

import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.config import CONFIG
from pipeline import utils
from pipeline.camera import api as camera_api
from recognizer import crop
from recognizer.phash.phash import Matcher

# configure logging same as other scripts
utils.configure_logging(level=CONFIG.log_level)
log = utils.get_logger("controller")

# store layout
CAPTURE_ROOT = Path(CONFIG.data_dir or "data") / "captures"
CAPTURE_ROOT.mkdir(parents=True, exist_ok=True)

# camera lock to avoid concurrent access from UI threads
_camera_lock = threading.Lock()
_shared_camera = None


def _get_camera(shared: bool = True, preview_size: Optional[Tuple[int, int]] = None):
    """
    Return a camera instance. If shared True, reuse a singleton camera object.
    Caller should not close shared camera; use camera_api.close_camera on app teardown.
    """
    global _shared_camera
    if shared:
        with _camera_lock:
            if _shared_camera is None:
                _shared_camera = camera_api.init_camera(preview_size=preview_size or CONFIG.preview_size)
            return _shared_camera
        # unreachable
    else:
        return camera_api.init_camera(preview_size=preview_size or CONFIG.preview_size)


def capture_once(camera=None, pad_x_pct: Optional[float] = None, pad_y_pct: Optional[float] = None, timeout: float = 1.0) -> Optional[np.ndarray]:
    """
    Grab one frame from camera, find card contour, crop it and return the card image (BGR).
    Returns None on failure.
    """
    cam = camera or _get_camera(shared=True)
    with _camera_lock:
        frame = camera_api.grab_frame(cam, timeout=timeout)
    if frame is None:
        log.info("capture_once: no frame")
        return None

    pad_x_pct = CONFIG.pad_x_pct if pad_x_pct is None else pad_x_pct
    pad_y_pct = CONFIG.pad_y_pct if pad_y_pct is None else pad_y_pct

    box, _ = crop.find_card_contour(frame, min_area=CONFIG.min_area)
    if box is None:
        log.debug("capture_once: no card contour found")
        return None

    card = crop.crop_card_from_box(frame, box, pad_x_pct=pad_x_pct, pad_y_pct=pad_y_pct)
    if card is None or card.size == 0:
        log.debug("capture_once: crop failed")
        return None
    return card


def _make_matcher(config_override: Optional[Dict[str, Any]] = None) -> Matcher:
    cfg = dict(vars(CONFIG))  # shallow copy
    if config_override:
        cfg.update(config_override)
    return Matcher(config=cfg)


def capture_and_match(camera=None, matcher: Optional[Matcher] = None, attempts: int = None, dist_threshold: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Capture once (using capture_once) and run matcher.match_with_policy with retries.
    Returns a dict with keys:
      - card (np.ndarray) optional (only included if include_image True)
      - match: dict returned by matcher.match_with_policy
      - ts_iso, id (short)
    Returns None if capture fails or no candidate found across attempts.
    """
    attempts = attempts or CONFIG.match_attempts
    dist_threshold = dist_threshold if dist_threshold is not None else CONFIG.phash_threshold
    cam = camera or _get_camera(shared=True)
    matcher = matcher or _make_matcher()

    for i in range(attempts):
        card = capture_once(camera=cam)
        if card is None:
            log.debug("capture_and_match: attempt %d no card", i + 1)
            continue

        res = matcher.match_with_policy(card)
        if not res:
            log.info("capture_and_match: attempt %d no candidate", i + 1)
            continue

        dist = res.get("dist")
        accepted_flag = bool(res.get("accepted", dist is not None and dist <= dist_threshold))
        log.info("capture_and_match: attempt %d id=%s dist=%s accepted=%s", i + 1, res.get("id"), dist, accepted_flag)

        # prefer explicit accepted, else use threshold
        if accepted_flag:
            out = {
                "ts_iso": datetime.utcnow().isoformat(),
                "result": res,
                "card": card
            }
            return out
        else:
            # still return the best candidate on final attempt for inspection
            if i == attempts - 1:
                out = {
                    "ts_iso": datetime.utcnow().isoformat(),
                    "result": res,
                    "card": card
                }
                return out

    return None


# ---------------------------
# Simple filesystem persistence
# ---------------------------

def _thumb_from_card(card_img: np.ndarray, max_dim: int = 200) -> np.ndarray:
    h, w = card_img.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        return card_img.copy()
    return cv2.resize(card_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def save_capture_record(card_img: np.ndarray, match_result: Dict[str, Any], base_dir: Path = CAPTURE_ROOT) -> str:
    """
    Persist a capture and metadata to data/captures/{id}/
    Returns the record id (uuid string).
    """
    if card_img is None or card_img.size == 0:
        raise ValueError("card_img missing")

    rec_id = uuid.uuid4().hex
    rec_dir = Path(base_dir) / rec_id
    rec_dir.mkdir(parents=True, exist_ok=True)

    # file names
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    img_path = rec_dir / f"{ts}_{rec_id[:8]}_card.png"
    thumb_path = rec_dir / f"{ts}_{rec_id[:8]}_thumb.png"
    meta_path = rec_dir / "meta.json"

    # write image
    ok = utils.safe_imwrite(str(img_path), card_img)
    if not ok:
        log.warning("save_capture_record: failed to write image %s", img_path)

    # write thumbnail
    thumb = _thumb_from_card(card_img)
    ok2 = utils.safe_imwrite(str(thumb_path), thumb)
    if not ok2:
        log.warning("save_capture_record: failed to write thumb %s", thumb_path)

    # serialize match_result but drop raw images if present
    serializable_result = dict(match_result)
    if "card" in serializable_result:
        serializable_result.pop("card", None)

    meta = {
        "id": rec_id,
        "timestamp": datetime.utcnow().isoformat(),
        "image_path": str(img_path.relative_to(Path.cwd())),
        "thumb_path": str(thumb_path.relative_to(Path.cwd())),
        "match": serializable_result,
        "config_snapshot": {k: getattr(CONFIG, k) for k in ("phash_threshold", "phash_out_size", "phash_clahe") if hasattr(CONFIG, k)}
    }

    with open(meta_path, "w", encoding="utf8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)

    log.info("Saved capture record %s (img=%s)", rec_id, img_path)
    return rec_id


def list_recent_captures(limit: int = 50, base_dir: Path = CAPTURE_ROOT) -> List[Dict[str, Any]]:
    """
    Return a list of captures ordered by timestamp desc (reads meta.json files).
    """
    records = []
    for rec_dir in sorted(base_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        meta = rec_dir / "meta.json"
        if not meta.exists():
            continue
        try:
            with open(meta, "r", encoding="utf8") as fh:
                data = json.load(fh)
            records.append(data)
            if len(records) >= limit:
                break
        except Exception:
            log.exception("list_recent_captures: failed to read %s", meta)
            continue
    return records


def set_capture_accept(record_id: str, accepted: Optional[bool] = True, note: Optional[str] = None, base_dir: Path = CAPTURE_ROOT) -> bool:
    """
    Mark an existing capture accepted/rejected and update the meta.json with accepted flag and note.
    Returns True if successful.
    """
    rec_dir = Path(base_dir) / record_id
    meta_path = rec_dir / "meta.json"
    if not meta_path.exists():
        return False
    try:
        with open(meta_path, "r+", encoding="utf8") as fh:
            meta = json.load(fh)
            meta["accepted"] = bool(accepted)
            if note:
                meta.setdefault("notes", [])
                meta["notes"].append({"ts": datetime.utcnow().isoformat(), "note": note})
            fh.seek(0)
            json.dump(meta, fh, indent=2, sort_keys=True)
            fh.truncate()
        return True
    except Exception:
        log.exception("set_capture_accept: failed for %s", record_id)
        return False
