"""
Runtime matcher: pHash prefilter -> ORB verification.
Lightweight caching of metadata; descriptor files loaded on demand.
"""

import sqlite3
import os
from pathlib import Path
from PIL import Image
import imagehash
import numpy as np
import cv2
from app import config, utils

DB_PATH = config.DB_PATH
DESC_DIR = Path(config.DESC_DIR)
ORB_FEATURES = config.ORB_FEATURES
PH_MAX = config.PHASH_HAMMING_THRESHOLD
TOP_K = config.MATCH_TOP_K

orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Cache tiny table in memory
_CARD_TABLE = None

def _load_table():
    global _CARD_TABLE
    if _CARD_TABLE is not None:
        return _CARD_TABLE
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"DB not found: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, name, set_code, collector_number, image_url, phash, desc_file FROM cards")
    rows = cur.fetchall()
    conn.close()
    _CARD_TABLE = [dict(id=r[0], name=r[1], set_code=r[2], collector_number=r[3],
                        image_url=r[4], phash=r[5], desc_file=r[6]) for r in rows]
    return _CARD_TABLE

def _phash_from_array(np_img):
    pil = Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
    return imagehash.phash(pil)

def _hamming(ph, hex_str):
    other = imagehash.hex_to_hash(hex_str)
    return ph - other

def get_candidates_by_phash(np_img, max_hamming=PH_MAX, top_k=TOP_K):
    table = _load_table()
    ph = _phash_from_array(np_img)
    candidates = []
    for rec in table:
        try:
            d = _hamming(ph, rec["phash"])
            if d <= max_hamming:
                candidates.append((rec, d))
        except Exception:
            continue
    candidates.sort(key=lambda x: x[1])
    return [c[0] for c in candidates[:top_k]]

def _load_descriptor(rec):
    path = DESC_DIR / (rec["id"] + ".npz")
    if not path.exists():
        return None, None
    data = np.load(path)
    des = data["des"] if "des" in data else np.array([])
    kps_arr = data["kps"] if "kps" in data else np.array([])
    # convert kps_arr -> cv2.KeyPoint list
    kps = []
    if kps_arr.size != 0:
        for row in kps_arr:
            x, y, size, angle = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            kps.append(cv2.KeyPoint(x, y, _size=size, _angle=angle))
    return kps, des

def verify_orb(np_img, rec, min_good=12):
    kps2, des2 = _load_descriptor(rec)
    if des2 is None or des2.size == 0:
        return 0, 0
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    H = 512
    scale = H / gray.shape[0]
    gray_r = cv2.resize(gray, (int(gray.shape[1]*scale), H), interpolation=cv2.INTER_LINEAR)
    kp1, des1 = orb.detectAndCompute(gray_r, None)
    if des1 is None or des1.size == 0:
        return 0, 0
    # knn match
    try:
        matches = bf.knnMatch(des1, des2, k=2)
    except Exception:
        return 0, 0
    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good.append(m)
    good_count = len(good)
    inliers = 0
    if good_count >= min_good and len(kps2) > 3:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kps2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        try:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is not None:
                inliers = int(mask.sum())
        except Exception:
            inliers = 0
    return good_count, inliers

def match_card(np_img, ph_max=PH_MAX, top_k=TOP_K):
    candidates = get_candidates_by_phash(np_img, max_hamming=ph_max, top_k=top_k)
    best = None
    best_score = -1
    for rec in candidates:
        good_count, inliers = verify_orb(np_img, rec)
        score = inliers * 2 + good_count
        if score > best_score:
            best_score = score
            best = (rec, score, good_count, inliers)
    return best
