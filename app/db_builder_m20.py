#!/usr/bin/env python3
"""
Builder for M20 scryfall DB + descriptors with:
- Scryfall paging download saved to data/scryfall_db/raw_scryfall/
- Logging to logs/ (timestamped file + current.log)
- Defensive NumPy checks and full traceback logging
Replace or adapt feature-extraction bits with your real extractor.
"""

import sys
import traceback
from pathlib import Path
import sqlite3
import json
import os
import logging
from datetime import datetime
import requests
import time
import tempfile

# third-party libs used by original builder (ensure installed in venv)
import numpy as np
from PIL import Image
import imagehash

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]  # repo root (one level above app/)
DATA_DIR = ROOT / "data" / "scryfall_db"
RAW_DIR = DATA_DIR / "raw_scryfall"
DESCRIPTORS_DIR = DATA_DIR / "descriptors"
DB_PATH = DATA_DIR / "cards.db"
LOG_DIR = ROOT / "logs"
SCRYFALL_M20_JSON = DATA_DIR / "scryfall_m20.json"  # final merged JSON file

# Ensure directories exist
for d in (DATA_DIR, RAW_DIR, DESCRIPTORS_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Logging setup ---
timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
logfile = LOG_DIR / f"builder_{timestamp}.log"
current_log = LOG_DIR / "builder_current.log"

logger = logging.getLogger("db_builder_m20")
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

# File handler (timestamped)
fh = logging.FileHandler(filename=str(logfile), encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(fmt)
logger.addHandler(fh)

# Current file handler (rotates by overwrite)
fh2 = logging.FileHandler(filename=str(current_log), encoding="utf-8", mode="w")
fh2.setLevel(logging.DEBUG)
fh2.setFormatter(fmt)
logger.addHandler(fh2)

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(fmt)
logger.addHandler(ch)


# --- Utilities ---
def safe_array_is_nonempty(a):
    """Return True if a is a numpy array-like and non-empty, or a non-empty sequence."""
    try:
        if a is None:
            return False
        if isinstance(a, np.ndarray):
            return a.size > 0
        if hasattr(a, "__len__"):
            return len(a) > 0
    except Exception:
        return bool(a)
    return False


def phash_for_image_path(img_path):
    """Compute perceptual hash string for an image file path."""
    im = Image.open(img_path).convert("RGB")
    return str(imagehash.phash(im))


def save_descriptor(card_id, kps, des):
    """Save descriptor .npz for card_id atomically and robustly (Windows-safe)."""
    target = DESCRIPTORS_DIR / f"{card_id}.npz"
    DESCRIPTORS_DIR.mkdir(parents=True, exist_ok=True)
    # Normalize arrays safely
    try:
        kps_arr = np.array(kps) if not isinstance(kps, np.ndarray) else kps
    except Exception:
        kps_arr = np.array([])
    try:
        des_arr = np.array(des) if not isinstance(des, np.ndarray) else des
    except Exception:
        des_arr = np.array([])

    # Create a temp file in the same directory to avoid cross-filesystem/permission rename issues
    try:
        with tempfile.NamedTemporaryFile(dir=str(DESCRIPTORS_DIR), prefix=f"{card_id}_", suffix=".npz", delete=False) as tf:
            tmp_path = Path(tf.name)
            # np.savez_compressed writes to a filename, so close the NamedTemporaryFile and use numpy to write directly
            pass
        # Use numpy to write to the temp path (this ensures correct .npz structure)
        np.savez_compressed(str(tmp_path), kps=kps_arr, des=des_arr)
        # Ensure file is flushed to disk before rename
        tmp_path_stat = tmp_path.stat()
        # Atomically replace target (use os.replace which works on Windows)
        os.replace(str(tmp_path), str(target))
        logger.debug("Wrote descriptor %s (%d bytes)", target.name, target.stat().st_size)
    except Exception:
        logger.exception("Failed to write descriptor for %s", card_id)
        # cleanup any leftover temp file
        try:
            if 'tmp_path' in locals() and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to remove temp file for %s", card_id)
        raise



# --- Scryfall fetch (with paging) ---
def fetch_scryfall_set(set_code="m20", pause_between=0.1):
    """
    Fetch all pages for a Scryfall set search and save pages into RAW_DIR.
    Returns merged list of card dicts.
    """
    logger.info("Fetching Scryfall set %s", set_code)
    base = "https://api.scryfall.com/cards/search"
    params = {"q": f"set:{set_code}", "unique": "prints", "order": "set"}
    url = base
    all_cards = []
    page = 0
    while url:
        page += 1
        logger.info("Requesting page %d: %s", page, url)
        try:
            r = requests.get(url, params=params if url == base else None, timeout=30)
        except Exception:
            logger.exception("HTTP request failed for %s", url)
            raise
        if r.status_code != 200:
            logger.error("Scryfall returned %d: %s", r.status_code, r.text[:400])
            r.raise_for_status()
        data = r.json()
        # save raw page
        raw_file = RAW_DIR / f"scryfall_page_{page}.json"
        with raw_file.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        logger.debug("Saved raw page to %s", raw_file)
        # collect data
        page_cards = data.get("data", [])
        logger.info("Page %d contains %d cards", page, len(page_cards))
        all_cards.extend(page_cards)
        if data.get("has_more"):
            # follow next_page link exactly
            url = data.get("next_page")
            params = None
            time.sleep(pause_between)
        else:
            url = None
    # merge into single JSON file
    with SCRYFALL_M20_JSON.open("w", encoding="utf-8") as fh:
        json.dump(all_cards, fh, ensure_ascii=False, indent=2)
    logger.info("Fetched total %d cards; merged saved to %s", len(all_cards), SCRYFALL_M20_JSON)
    return all_cards


# --- Core processing ---
def process_card(card):
    """
    Create a descriptor for a card dict.
    Uses local image file if available (checks image_uris->normal or file://).
    This function is defensive about numpy boolean checks.
    """
    # derive card id safe for filenames
    raw_id = card.get("id") or card.get("oracle_id") or card.get("name", "unknown")
    # normalize to filesystem-friendly name
    card_id = raw_id.replace(" ", "_").replace("/", "_").replace(":", "_").replace("'", "")
    try:
        # find a usable local image path if available
        img_path = None
        # prefer local file url if present
        image_url = card.get("image_url") or ""
        if image_url and isinstance(image_url, str) and image_url.startswith("file://"):
            img_path = image_url[7:]
        # check keys Scryfall provides
        u = card.get("image_uris") or {}
        if not img_path and isinstance(u, dict):
            # prefer normal or small if present, though these will be remote URLs
            candidate = u.get("normal") or u.get("small") or u.get("png")
            if candidate and candidate.startswith("file://"):
                img_path = candidate[7:]
        if not img_path:
            # If no local image, write an empty descriptor placeholder and skip image ops
            save_descriptor(card_id, kps=[], des=[])
            return {"id": card_id, "desc_file": f"{card_id}.npz", "phash": None}

        # compute phash
        ph = phash_for_image_path(img_path)

        # placeholder numeric descriptor from phash
        hexchars = ph.replace(" ", "")
        nums = np.array([int(c, 16) for c in hexchars[:32]], dtype=np.uint8)

        save_descriptor(card_id, kps=np.empty((0,)), des=nums)
        return {"id": card_id, "desc_file": f"{card_id}.npz", "phash": ph}

    except Exception:
        logger.exception("Full traceback when processing card %s", card.get("name") or card_id)
        # attempt to save a placeholder descriptor
        try:
            save_descriptor(card_id, kps=[], des=[])
            return {"id": card_id, "desc_file": f"{card_id}.npz", "phash": None}
        except Exception:
            logger.exception("Failed to save placeholder descriptor for %s", card_id)
            raise


def build_db(cards):
    """Write or update the SQLite cards.db with entries returned from process_card."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS cards (
            id TEXT PRIMARY KEY,
            name TEXT,
            set_code TEXT,
            collector_number TEXT,
            image_url TEXT,
            phash TEXT,
            desc_file TEXT
        )
        """
    )
    inserted = 0
    for c in cards:
        try:
            result = process_card(c)
            cur.execute(
                "INSERT OR REPLACE INTO cards (id,name,set_code,collector_number,image_url,phash,desc_file) VALUES (?,?,?,?,?,?,?)",
                (
                    result.get("id"),
                    c.get("name"),
                    c.get("set_code") or c.get("set"),
                    c.get("collector_number"),
                    c.get("image_url") or c.get("image_path") or None,
                    result.get("phash"),
                    result.get("desc_file"),
                ),
            )
            inserted += 1
        except Exception:
            logger.exception("Skipping card on exception: %s", c.get("id") or c.get("name"))
            continue
    conn.commit()
    conn.close()
    return inserted


def load_card_list():
    """
    Load card metadata for the M20 set.
    Priority:
     1) SCRYFALL_M20_JSON merged file if present
     2) raw_scryfall pages if present (merge them)
     3) attempt to fetch from Scryfall API
     4) fallback: data/ref_images or data/ref.png single stub
    """
    # 1) merged file
    if SCRYFALL_M20_JSON.exists():
        try:
            with SCRYFALL_M20_JSON.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            logger.info("Loaded %d cards from %s", len(data), SCRYFALL_M20_JSON)
            return data
        except Exception:
            logger.exception("Failed to load merged JSON")

    # 2) raw pages
    pages = sorted(RAW_DIR.glob("scryfall_page_*.json"))
    if pages:
        all_cards = []
        for p in pages:
            try:
                with p.open("r", encoding="utf-8") as fh:
                    page = json.load(fh)
                # page may be a dictionary with 'data' or a list, handle both
                if isinstance(page, dict) and "data" in page:
                    all_cards.extend(page["data"])
                elif isinstance(page, list):
                    all_cards.extend(page)
            except Exception:
                logger.exception("Failed to read raw page %s", p)
        if all_cards:
            # save merged
            with SCRYFALL_M20_JSON.open("w", encoding="utf-8") as fh:
                json.dump(all_cards, fh, ensure_ascii=False, indent=2)
            logger.info("Merged %d cards from %d raw pages into %s", len(all_cards), len(pages), SCRYFALL_M20_JSON)
            return all_cards

    # 3) fetch from Scryfall now
    try:
        cards = fetch_scryfall_set("m20")
        if cards:
            return cards
    except Exception:
        logger.exception("Scryfall fetch failed; falling back to local images")

    # 4) fallback to local images
    images_dir = ROOT / "data" / "ref_images"
    images = []
    if images_dir.exists():
        for p in sorted(images_dir.glob("*.*")):
            images.append(
                {
                    "id": p.stem,
                    "name": p.stem,
                    "image_path": str(p),
                    "set_code": "m20",
                    "collector_number": "0",
                    "image_url": f"file://{p}",
                }
            )
    if images:
        logger.info("Using %d local reference images from %s", len(images), images_dir)
        return images

    # single ref.png fallback
    ref = ROOT / "data" / "ref.png"
    if ref.exists():
        logger.info("Using single ref image %s as stub", ref)
        return [
            {
                "id": "ref_stub",
                "name": "ref_stub",
                "image_path": str(ref),
                "set_code": "m20",
                "collector_number": "1",
                "image_url": f"file://{ref}",
            }
        ]

    logger.warning("No card metadata found. Place reference images in data/ref_images or add scryfall JSON.")
    return []


def main():
    logger.info("Builder started")
    cards = load_card_list()
    if not cards:
        logger.error("No card metadata found. Exiting.")
        # ensure DB exists empty
        build_db([])
        return

    logger.info("Processing %d cards...", len(cards))
    try:
        inserted = build_db(cards)
        logger.info("Done. DB at %s Descriptors in %s", DB_PATH, DESCRIPTORS_DIR)
        logger.info("Inserted or updated rows: %d", inserted)
    except Exception:
        logger.exception("Fatal error during build")
        sys.exit(2)


if __name__ == "__main__":
    main()
