#!/usr/bin/env python3
"""
tools/store.py

SQLite-backed capture store for mtg-card-sorter.

Public API:
- init_store(db_path: Optional[str]) -> sqlite3.Connection
- save_capture_record(card_img: np.ndarray, match_result: dict, conn: sqlite3.Connection) -> str
- list_recent_captures(limit: int, conn: sqlite3.Connection) -> List[dict]
- get_capture(record_id: str, conn: sqlite3.Connection) -> Optional[dict]
- set_capture_accept(record_id: str, accepted: Optional[bool], note: Optional[str], conn: sqlite3.Connection) -> bool
- close_store(conn: sqlite3.Connection) -> None

Behavior:
- Persists full image and thumbnail under data/captures/{id}/
- Stores metadata and candidate list in SQLite table `captures`
- Thread-safe-ish: caller should serialize concurrent writes if needed
"""

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from config.config import CONFIG
from pipeline import utils

# storage locations
DATA_DIR = Path(getattr(CONFIG, "data_dir", "data"))
CAPTURES_DIR = DATA_DIR / "captures"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_DB_PATH = DATA_DIR / "captures.db"

# simple lock for db file and FS writes
_store_lock = threading.Lock()

# SQL schema
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS captures (
    id TEXT PRIMARY KEY,
    ts_iso TEXT NOT NULL,
    image_path TEXT NOT NULL,
    thumb_path TEXT NOT NULL,
    phash_full TEXT,
    phash_title TEXT,
    phash_size INTEGER,
    best_candidate_id TEXT,
    best_distance INTEGER,
    title_distance INTEGER,
    candidates_json TEXT,
    accepted INTEGER,
    config_snapshot TEXT,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_ts ON captures(ts_iso DESC);
CREATE INDEX IF NOT EXISTS idx_candidate ON captures(best_candidate_id);
"""

def _thumb_from_card(card_img: np.ndarray, max_dim: int = 200) -> np.ndarray:
    h, w = card_img.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        return card_img.copy()
    return cv2.resize(card_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def init_store(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Initialize and return an sqlite3.Connection connected to db_path.
    """
    db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    with conn:
        conn.executescript(_CREATE_TABLE_SQL)
    return conn

def _write_image_files(rec_id: str, card_img: np.ndarray, base_dir: Optional[Path] = None) -> Dict[str, str]:
    """
    Write full image and thumbnail to disk under base_dir/rec_id/.
    Returns dict with 'image_path' and 'thumb_path' (absolute strings).
    """
    base_dir = Path(base_dir or CAPTURES_DIR)
    rec_dir = base_dir / rec_id
    rec_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    img_name = f"{ts}_{rec_id[:8]}_card.png"
    thumb_name = f"{ts}_{rec_id[:8]}_thumb.png"
    img_path = rec_dir / img_name
    thumb_path = rec_dir / thumb_name

    ok = utils.safe_imwrite(str(img_path), card_img)
    if not ok:
        raise IOError(f"Failed to write image to {img_path}")

    thumb = _thumb_from_card(card_img)
    ok2 = utils.safe_imwrite(str(thumb_path), thumb)
    if not ok2:
        raise IOError(f"Failed to write thumb to {thumb_path}")

    return {"image_path": str(img_path), "thumb_path": str(thumb_path)}

def save_capture_record(card_img: np.ndarray, match_result: Dict[str, Any], conn: sqlite3.Connection, base_dir: Optional[Path] = None) -> str:
    """
    Persist a capture and metadata into the SQLite store and filesystem.
    Returns the record id (uuid hex).
    """
    if card_img is None or card_img.size == 0:
        raise ValueError("card_img missing or empty")
    if conn is None:
        raise ValueError("Database connection required")

    rec_id = uuid.uuid4().hex
    with _store_lock:
        paths = _write_image_files(rec_id, card_img, base_dir=base_dir)
        # prepare fields
        phash_full = None
        phash_title = None
        phash_size = None
        best_candidate_id = None
        best_distance = None
        title_distance = None
        candidates_json = None

        # try to extract structured fields from match_result
        try:
            res = match_result.get("result") if "result" in match_result else match_result
            # res expected to be dict returned by Matcher.match_with_policy
            best_candidate_id = res.get("id")
            best_distance = res.get("dist")
            title_distance = res.get("title_dist")
            # if match_result contains phashes, ingest them
            phash_full = res.get("phash_full") or res.get("phash")
            phash_title = res.get("phash_title")
            # candidates list
            candidates = res.get("candidates") or res.get("top_k")
            if candidates is None and isinstance(res.get("candidates_json"), str):
                candidates = json.loads(res.get("candidates_json"))
            if candidates is not None:
                candidates_json = json.dumps(candidates)
        except Exception:
            candidates_json = json.dumps(match_result)

        config_snapshot = {k: getattr(CONFIG, k) for k in ("phash_threshold", "phash_out_size", "phash_clahe") if hasattr(CONFIG, k)}
        notes = None

        ts_iso = datetime.utcnow().isoformat()

        with conn:
            conn.execute(
                """
                INSERT INTO captures (
                    id, ts_iso, image_path, thumb_path, phash_full, phash_title, phash_size,
                    best_candidate_id, best_distance, title_distance, candidates_json,
                    accepted, config_snapshot, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec_id, ts_iso, paths["image_path"], paths["thumb_path"], phash_full, phash_title, phash_size,
                    best_candidate_id, best_distance, title_distance, candidates_json,
                    None, json.dumps(config_snapshot), notes
                )
            )
    return rec_id

def list_recent_captures(limit: int = 50, conn: sqlite3.Connection = None) -> List[Dict[str, Any]]:
    """
    Return recent captures ordered by ts_iso desc.
    Each record is a dict with DB fields and parsed JSON fields where applicable.
    """
    if conn is None:
        conn = init_store()

    rows = []
    with _store_lock:
        cur = conn.execute("SELECT id, ts_iso, image_path, thumb_path, phash_full, phash_title, phash_size, best_candidate_id, best_distance, title_distance, candidates_json, accepted, config_snapshot, notes FROM captures ORDER BY ts_iso DESC LIMIT ?", (limit,))
        cols = [c[0] for c in cur.description]
        for r in cur.fetchall():
            rec = dict(zip(cols, r))
            # parse JSON fields
            if rec.get("candidates_json"):
                try:
                    rec["candidates"] = json.loads(rec["candidates_json"])
                except Exception:
                    rec["candidates"] = rec["candidates_json"]
            if rec.get("config_snapshot"):
                try:
                    rec["config_snapshot"] = json.loads(rec["config_snapshot"])
                except Exception:
                    pass
            rows.append(rec)
    return rows

def get_capture(record_id: str, conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
    """
    Return a single capture record by id or None if not found.
    """
    with _store_lock:
        cur = conn.execute("SELECT id, ts_iso, image_path, thumb_path, phash_full, phash_title, phash_size, best_candidate_id, best_distance, title_distance, candidates_json, accepted, config_snapshot, notes FROM captures WHERE id = ?", (record_id,))
        row = cur.fetchone()
    if not row:
        return None
    cols = [c[0] for c in cur.description]
    rec = dict(zip(cols, row))
    if rec.get("candidates_json"):
        try:
            rec["candidates"] = json.loads(rec["candidates_json"])
        except Exception:
            rec["candidates"] = rec["candidates_json"]
    if rec.get("config_snapshot"):
        try:
            rec["config_snapshot"] = json.loads(rec["config_snapshot"])
        except Exception:
            pass
    return rec

def set_capture_accept(record_id: str, accepted: Optional[bool], note: Optional[str], conn: sqlite3.Connection) -> bool:
    """
    Mark an existing capture accepted/rejected. Note persisted as JSON list of entries.
    Returns True on success.
    """
    with _store_lock:
        cur = conn.execute("SELECT notes FROM captures WHERE id = ?", (record_id,))
        row = cur.fetchone()
        if not row:
            return False
        notes_raw = row[0]
        notes_list = []
        if notes_raw:
            try:
                notes_list = json.loads(notes_raw)
            except Exception:
                notes_list = []
        entry = {"ts": datetime.utcnow().isoformat(), "note": note or "", "accepted": accepted}
        notes_list.append(entry)
        notes_json = json.dumps(notes_list)
        try:
            with conn:
                conn.execute("UPDATE captures SET accepted = ?, notes = ? WHERE id = ?", (None if accepted is None else int(bool(accepted)), notes_json, record_id))
            return True
        except Exception:
            return False

def close_store(conn: sqlite3.Connection) -> None:
    try:
        conn.close()
    except Exception:
        pass
