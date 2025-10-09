#!/usr/bin/env python3
"""
db_builder_m20.py
Build a local Scryfall DB limited to the M20 set.

Creates:
  data/scryfall_db/cards.db
  data/scryfall_db/descriptors/<scryfall_id>.npz

Run from the project root:
  python app/db_builder_m20.py

Dependencies:
  pip install pillow imagehash requests tqdm opencv-python-headless numpy
"""
import os
import sqlite3
import requests
from io import BytesIO
from pathlib import Path
from PIL import Image
import imagehash
import cv2
import numpy as np
from tqdm import tqdm
import time

# Config
OUT_DIR = Path("data/scryfall_db")
DESC_DIR = OUT_DIR / "descriptors"
DB_PATH = OUT_DIR / "cards.db"
ORB_FEATURES = 800     # tune: 800 is a good balance; lower for less CPU/disk
SET_CODE = "m20"       # target set

OUT_DIR.mkdir(parents=True, exist_ok=True)
DESC_DIR.mkdir(parents=True, exist_ok=True)

# Initialize ORB
orb = cv2.ORB_create(nfeatures=ORB_FEATURES)

# SQLite init
conn = sqlite3.connect(str(DB_PATH))
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS cards (
    id TEXT PRIMARY KEY,
    name TEXT,
    set_code TEXT,
    collector_number TEXT,
    image_url TEXT,
    phash TEXT,
    desc_file TEXT
)
""")
conn.commit()

def save_descriptor(card_id, keypoints, descriptors):
    out_path = DESC_DIR / f"{card_id}.npz"
    if descriptors is None or len(descriptors) == 0:
        np.savez_compressed(out_path, kps=np.array([]), des=np.array([]))
        return out_path.name
    kps_arr = np.array([[kp.pt[0], kp.pt[1], kp.size, kp.angle] for kp in keypoints], dtype=np.float32)
    np.savez_compressed(out_path, kps=kps_arr, des=descriptors)
    return out_path.name

def fetch_set_cards(set_code):
    """
    Paginate Scryfall set search: https://api.scryfall.com/cards/search?q=set:SETCODE
    Returns a list of card objects.
    """
    cards = []
    url = f"https://api.scryfall.com/cards/search?q=set%3A{set_code}&order=set&unique=prints"
    while url:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        cards.extend(data.get("data", []))
        url = data.get("next_uri") or None
        # be polite
        time.sleep(0.1)
    return cards

def pick_image_url(card):
    """
    Prefer single-face image_uris.normal, otherwise try card_faces first face.
    """
    if card.get("image_uris"):
        return card["image_uris"].get("normal") or card["image_uris"].get("large") or card["image_uris"].get("png")
    faces = card.get("card_faces")
    if faces and isinstance(faces, list) and faces[0].get("image_uris"):
        return faces[0]["image_uris"].get("normal") or faces[0]["image_uris"].get("png")
    return None

def process_card(card):
    img_url = pick_image_url(card)
    if not img_url:
        print("No image for", card.get("name"))
        return

    try:
        r = requests.get(img_url, timeout=30, stream=True)
        r.raise_for_status()
        pil = Image.open(BytesIO(r.content)).convert("RGB")
    except Exception as e:
        print("Failed download:", card.get("name"), e)
        return

    # pHash
    ph = imagehash.phash(pil)
    ph_hex = ph.__str__()

    # convert to OpenCV BGR and grayscale
    npimg = np.array(pil)[:, :, ::-1]
    gray = cv2.cvtColor(npimg, cv2.COLOR_BGR2GRAY)

    # normalize height for descriptor extraction
    H = 512
    scale = H / float(max(1, gray.shape[0]))
    gray_r = cv2.resize(gray, (int(gray.shape[1]*scale), H), interpolation=cv2.INTER_AREA)

    kps, des = orb.detectAndCompute(gray_r, None)
    desc_file = save_descriptor(card["id"], kps or [], des or [])

    # insert into sqlite
    cur.execute("""INSERT OR REPLACE INTO cards
                   (id, name, set_code, collector_number, image_url, phash, desc_file)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (card.get("id"), card.get("name"), card.get("set"),
                 card.get("collector_number"), img_url, ph_hex, desc_file))
    conn.commit()
    print("Saved:", card.get("name"))

def main():
    print("Fetching set", SET_CODE)
    cards = fetch_set_cards(SET_CODE)
    print("Found", len(cards), "cards in set", SET_CODE)
    for card in tqdm(cards):
        try:
            process_card(card)
        except KeyboardInterrupt:
            print("Interrupted")
            break
        except Exception as e:
            print("Error processing", card.get("name"), e)

    conn.close()
    print("Done. DB at", DB_PATH, "Descriptors in", DESC_DIR)

if __name__ == "__main__":
    main()
