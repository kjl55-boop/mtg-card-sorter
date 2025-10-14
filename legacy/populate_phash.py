#!/usr/bin/env python3
"""
Populate cards.db phash column from scryfall JSON images.

Usage:
  source venv/bin/activate
  python tools/populate_phash.py

Behavior:
 - Uses data/scryfall_db/scryfall_m20.json if present (merged pages).
 - Looks for image_uris.normal | png | small in each card entry.
 - If URL is file://local_path it uses local file directly.
 - If URL is remote (http/https) it downloads to data/tmp_images/ (cached).
 - Computes imagehash.phash(Image.convert("RGB")) and writes the hex string to cards.db.phash.
 - Skips cards that already have a non-empty phash unless --force is given.
"""
import sqlite3, json, os, argparse
from pathlib import Path
from PIL import Image
import imagehash
import requests
import time

ROOT = Path.cwd()
DATA_DIR = ROOT / "data" / "scryfall_db"
JSON_MERGED = DATA_DIR / "scryfall_m20.json"
TMP_DIR = DATA_DIR / "tmp_images"
DB_PATH = DATA_DIR / "cards.db"

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true", help="Recompute phash for rows that already have a phash")
parser.add_argument("--delay", type=float, default=0.05, help="Delay between downloads to be polite")
args = parser.parse_args()

TMP_DIR.mkdir(parents=True, exist_ok=True)

def compute_phash_from_path(p: Path):
    img = Image.open(str(p)).convert("RGB")
    return str(imagehash.phash(img))

def download_image(url: str, dest: Path):
    try:
        # stream to avoid loading entire file in memory
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
        return True
    except Exception as e:
        print("Download failed for", url, ":", e)
        if dest.exists():
            try: dest.unlink()
            except: pass
        return False

def load_scryfall_map():
    mapping = {}
    if not JSON_MERGED.exists():
        print("Merged JSON not found:", JSON_MERGED)
        return mapping
    with JSON_MERGED.open("r", encoding="utf-8") as fh:
        cards = json.load(fh)
    for c in cards:
        cid = c.get("id") or c.get("oracle_id") or c.get("name")
        # prioritize local file:// URIs
        img = None
        u = c.get("image_uris") or {}
        if isinstance(u, dict):
            img = u.get("normal") or u.get("png") or u.get("small")
        # some prints have 'card_faces' or 'image_uris' in faces; handle card_faces first face
        if not img and c.get("card_faces"):
            f0 = c["card_faces"][0]
            u2 = f0.get("image_uris") or {}
            if isinstance(u2, dict):
                img = u2.get("normal") or u2.get("png") or u2.get("small")
        mapping[c.get("id")] = img
    return mapping

def main():
    mapping = load_scryfall_map()
    if not mapping:
        print("No image mapping found in merged Scryfall JSON.")
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    # ensure phash column exists
    cur.execute("PRAGMA table_info(cards)")
    cols = [r[1] for r in cur.fetchall()]
    if "phash" not in cols:
        print("Adding phash column to cards table")
        cur.execute("ALTER TABLE cards ADD COLUMN phash TEXT")
        conn.commit()

    cur.execute("SELECT id, image_url, phash FROM cards")
    rows = cur.fetchall()
    total = len(rows)
    updated = 0
    for i,(cid, image_url, phash) in enumerate(rows, start=1):
        if phash and not args.force:
            print(f"[{i}/{total}] {cid}: already has phash, skipping")
            continue

        # determine best image path or url
        img_url = None
        if image_url and isinstance(image_url, str) and image_url.startswith("file://"):
            img_url = image_url[7:]
        elif image_url and isinstance(image_url, str) and image_url.startswith(("http://","https://")):
            img_url = image_url
        else:
            # try mapping from merged JSON
            img_url = mapping.get(cid)

        if not img_url:
            print(f"[{i}/{total}] {cid}: no image URL available, leaving phash empty")
            continue

        # handle local file vs remote
        img_path = None
        if isinstance(img_url, str) and img_url.startswith("file://"):
            img_path = Path(img_url[7:])
        elif isinstance(img_url, str) and img_url.startswith(("http://","https://")):
            # download to cache
            dest = TMP_DIR / f"{cid}.img"
            if not dest.exists():
                ok = download_image(img_url, dest)
                time.sleep(args.delay)
                if not ok:
                    print(f"[{i}/{total}] {cid}: failed to download {img_url}")
                    continue
            img_path = dest
        else:
            # sometimes mapping can be a local path without scheme
            cand = Path(img_url)
            if cand.exists():
                img_path = cand

        if not img_path or not img_path.exists():
            print(f"[{i}/{total}] {cid}: image path does not exist ({img_path})")
            continue

        try:
            ph = compute_phash_from_path(img_path)
            cur.execute("UPDATE cards SET phash=? WHERE id=?", (ph, cid))
            conn.commit()
            updated += 1
            print(f"[{i}/{total}] {cid}: phash {ph} written")
        except Exception as e:
            print(f"[{i}/{total}] {cid}: failed to compute phash: {e}")
            continue

    print("Done. Updated phash rows:", updated)
    conn.close()

if __name__ == "__main__":
    main()
