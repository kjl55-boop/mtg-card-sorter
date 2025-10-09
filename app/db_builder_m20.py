#!/usr/bin/env python3
"""
Robust builder for M20 scryfall DB + descriptors.

Replaces ambiguous NumPy boolean checks with explicit tests,
prints full tracebacks for failures, and continues processing.
Adjust paths or imports if your project layout differs.
"""

import sys
import traceback
from pathlib import Path
import sqlite3
import json
import os

# third-party libs used by original builder (ensure installed in venv)
import numpy as np
from PIL import Image
import imagehash

# --- Configuration (adjust these if your repo uses different paths) ---
ROOT = Path(__file__).resolve().parents[1]  # repo root (one level above app/)
DATA_DIR = ROOT / "data" / "scryfall_db"
DESCRIPTORS_DIR = DATA_DIR / "descriptors"
DB_PATH = DATA_DIR / "cards.db"
SCRYFALL_JSON = ROOT / "data" / "scryfall_db" / "scryfall_m20.json"  # optional source if you have it

# Ensure directories exist
DESCRIPTORS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def safe_array_is_nonempty(a):
    """Return True if a is a numpy array-like and non-empty, or a non-empty sequence."""
    try:
        if a is None:
            return False
        if isinstance(a, np.ndarray):
            return a.size > 0
        # lists/tuples
        if hasattr(a, "__len__"):
            return len(a) > 0
    except Exception:
        # fallback: treat as non-empty only if truthy and not an ndarray ambiguity
        return bool(a)
    return False


def phash_for_image_path(img_path):
    """Compute perceptual hash string for an image file path."""
    im = Image.open(img_path).convert("RGB")
    return str(imagehash.phash(im))


def save_descriptor(card_id, kps, des):
    """Save descriptor .npz for card_id."""
    target = DESCRIPTORS_DIR / f"{card_id}.npz"
    # ensure arrays are numpy arrays (safe conversion)
    try:
        kps_arr = np.array(kps) if not isinstance(kps, np.ndarray) else kps
    except Exception:
        kps_arr = np.array([])
    try:
        des_arr = np.array(des) if not isinstance(des, np.ndarray) else des
    except Exception:
        des_arr = np.array([])
    # atomically write to a temp file then move
    tmp = target.with_suffix(".npz.tmp")
    np.savez_compressed(tmp, kps=kps_arr, des=des_arr)
    os.replace(tmp, target)


def process_card(card):
    """
    card: dict-like with expected keys:
      - id (string unique)
      - image_path or image_url (local paths currently supported here)
      - name, set_code, collector_number (optional metadata)
    The real builder will have different inputs; adapt as needed.
    This function must not use ambiguous boolean checks on arrays.
    """
    card_id = card.get("id") or card.get("card_id") or card.get("name", "unknown").replace(" ", "_")
    try:
        # Simulated descriptor generation:
        # In your original builder this is where keypoints/descriptors are computed (e.g., SIFT, ORB).
        # Replace the lines below with your actual feature extraction calls.
        # We'll create a simple placeholder: descriptor = phash, kps empty for stub.
        img_path = card.get("image_path") or card.get("image_file")
        if not img_path:
            # fallback: if card provides image_url that points to local file "file://..."
            img_url = card.get("image_url", "")
            if img_url.startswith("file://"):
                img_path = img_url[7:]
        if not img_path or not Path(img_path).exists():
            # If no local image, skip computing descriptors; still write an empty descriptor file
            save_descriptor(card_id, kps=[], des=[])
            return {"id": card_id, "desc_file": f"{card_id}.npz", "phash": None}

        # compute a perceptual hash to use as a quick descriptor
        ph = phash_for_image_path(img_path)

        # Placeholder: a tiny numeric descriptor built from phash hex chars
        # Convert hex string to small numeric array as a deterministic stub
        hexchars = ph.replace(" ", "")
        nums = np.array([int(c, 16) for c in hexchars[:32]], dtype=np.uint8)

        # Save descriptor file
        save_descriptor(card_id, kps=np.empty((0,)), des=nums)

        return {"id": card_id, "desc_file": f"{card_id}.npz", "phash": ph}

    except Exception:
        # full traceback to stderr so logs capture file+line number
        print(f"Full traceback when processing card {card_id}:", file=sys.stderr)
        traceback.print_exc()
        # ensure we still create a placeholder descriptor so output stays consistent
        try:
            save_descriptor(card_id, kps=[], des=[])
            return {"id": card_id, "desc_file": f"{card_id}.npz", "phash": None}
        except Exception:
            # if saving failed, re-raise after logging
            print(f"Failed to save placeholder descriptor for {card_id}", file=sys.stderr)
            traceback.print_exc()
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
                    c.get("set_code"),
                    c.get("collector_number"),
                    c.get("image_url") or c.get("image_path"),
                    result.get("phash"),
                    result.get("desc_file"),
                ),
            )
            inserted += 1
        except Exception:
            # continue after logging; process_card already logs full traceback
            print(f"Skipping card on exception: {c.get('id') or c.get('name')}", file=sys.stderr)
            continue
    conn.commit()
    conn.close()
    return inserted


def load_card_list():
    """
    Load card metadata for the M20 set.
    If you have a JSON export from Scryfall used by the original builder, adapt this loader.
    Fallback: create a tiny stub set if no JSON present.
    """
    if SCRYFALL_JSON.exists():
        try:
            with open(SCRYFALL_JSON, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            # Expecting a list of card dicts; adapt keys to your original data shape
            return data
        except Exception:
            print("Failed to load scryfall JSON, falling back to a small stub", file=sys.stderr)
            traceback.print_exc()
    # Fallback stub: try to find any local images in data/ref_images
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
        return images
    # last resort: minimal single synthetic entry pointing to data/ref.png if present
    ref = ROOT / "data" / "ref.png"
    if ref.exists():
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
    # Completely empty fallback
    return []


def main():
    cards = load_card_list()
    if not cards:
        print("No card metadata found. Place reference images in data/ref_images or add scryfall JSON.", file=sys.stderr)
        # still ensure DB exists empty
        build_db([])
        return

    print(f"Processing {len(cards)} cards...")
    try:
        inserted = build_db(cards)
        print(f"Done. DB at {DB_PATH} Descriptors in {DESCRIPTORS_DIR}")
        print(f"Inserted or updated rows: {inserted}")
    except Exception:
        print("Fatal error during build:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
