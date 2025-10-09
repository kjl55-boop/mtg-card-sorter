"""
Offline builder to create the sqlite metadata and descriptor files from Scryfall default-cards JSON.
This is the script you run once (or periodically) on a machine with network access and sufficient disk space.
"""

import os
import json
import sqlite3
import requests
from io import BytesIO
from PIL import Image
import imagehash
import cv2
import numpy as np
from tqdm import tqdm
from app import config, utils

ORB_FEATURES = config.ORB_FEATURES
DESC_DIR = config.DESC_DIR
DB_PATH = config.DB_PATH
utils.ensure_dir(DESC_DIR)

def init_db():
    conn = sqlite3.connect(DB_PATH)
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
    )""")
    conn.commit()
    return conn

def save_descriptor(card_id, keypoints, descriptors):
    out_path = DESC_DIR / f"{card_id}.npz"
    if descriptors is None or len(descriptors) == 0:
        np.savez_compressed(out_path, kps=np.array([]), des=np.array([]))
        return out_path.name
    kps_arr = np.array([[kp.pt[0], kp.pt[1], kp.size, kp.angle] for kp in keypoints], dtype=np.float32)
    np.savez_compressed(out_path, kps=kps_arr, des=descriptors)
    return out_path.name

def process_card(card, conn, orb):
    # choose image uri
    img_url = None
    if card.get("image_uris"):
        img_url = card["image_uris"].get("normal") or card["image_uris"].get("large") or card["image_uris"].get("png")
    else:
        faces = card.get("card_faces")
        if faces and isinstance(faces, list) and faces[0].get("image_uris"):
            img_url = faces[0]["image_uris"].get("normal")
    if not img_url:
        return

    try:
        resp = requests.get(img_url, timeout=20)
        resp.raise_for_status()
        pil = Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        print("Download failed:", card.get("name"), e)
        return

    ph = imagehash.phash(pil)
    ph_hex = ph.__str__()

    npimg = np.array(pil)[:, :, ::-1]
    gray = cv2.cvtColor(npimg, cv2.COLOR_BGR2GRAY)
    H = 512
    scale = H / gray.shape[0]
    gray_resized = cv2.resize(gray, (int(gray.shape[1]*scale), H), interpolation=cv2.INTER_LINEAR)
    kps, des = orb.detectAndCompute(gray_resized, None)
    desc_file = save_descriptor(card["id"], kps or [], des or [])

    cur = conn.cursor()
    cur.execute("""INSERT OR REPLACE INTO cards
                   (id, name, set_code, collector_number, image_url, phash, desc_file)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (card.get("id"), card.get("name"), card.get("set"),
                 card.get("collector_number"), img_url, ph_hex, desc_file))
    conn.commit()

def build_from_json(json_path):
    conn = init_db()
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    with open(json_path, "r", encoding="utf-8") as f:
        cards = json.load(f)
    for card in tqdm(cards):
        try:
            process_card(card, conn, orb)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Error:", e)
    conn.close()

if __name__ == "__main__":
    # run as script: set path to default-cards.json
    DEFAULT_JSON = utils.ensure_dir(config.SCRYFALL_RAW) / "default-cards.json"
    if not DEFAULT_JSON.exists():
        print("Place Scryfall default-cards.json into:", DEFAULT_JSON)
    else:
        build_from_json(str(DEFAULT_JSON))
