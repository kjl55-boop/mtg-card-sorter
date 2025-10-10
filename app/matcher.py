"""
Matching utilities: phash prefilter and descriptor (ORB) verification.

Functions:
 - load_db_phashes(db_path) -> list of (id,name,phash,desc_file)
 - top_phash_candidates(query_ph, rows, top_n) -> list of candidates sorted by hamming
 - verify_candidate_orb(query_bgr, candidate_npz_path, min_good, ratio) -> (bool, good_matches)
"""

from pathlib import Path
import sqlite3
from typing import Tuple
import numpy as np
import cv2
from .config import DESCRIPTOR_MIN_GOOD_MATCHES, DESCRIPTOR_RATIO_TEST, PHASH_TOP_N_CANDIDATES
import imagehash

def load_db_phashes(db_path: str = str(Path("data/scryfall_db/cards.db"))):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id,name,phash,desc_file FROM cards WHERE phash IS NOT NULL AND phash != ''")
    rows = cur.fetchall()
    conn.close()
    return rows

def _hamming(a_hex: str, b_hex: str) -> int:
    return bin(int(a_hex, 16) ^ int(b_hex, 16)).count("1")

def top_phash_candidates(query_ph: str, rows, top_n: int = PHASH_TOP_N_CANDIDATES):
    scores = []
    for cid, name, ph, desc_file in rows:
        try:
            d = _hamming(query_ph, ph)
            scores.append((d, cid, name, ph, desc_file))
        except Exception:
            continue
    scores.sort(key=lambda x: x[0])
    return scores[:top_n]

def verify_candidate_orb(query_bgr: np.ndarray, candidate_npz_path: Path, min_good: int = DESCRIPTOR_MIN_GOOD_MATCHES, ratio: float = DESCRIPTOR_RATIO_TEST) -> Tuple[bool,int]:
    # compute ORB descriptors for query
    orb = cv2.ORB_create(1000)
    qgray = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2GRAY)
    qk, qd = orb.detectAndCompute(qgray, None)
    if qd is None or len(qd) == 0:
        return False, 0
    # load stored descriptors
    try:
        data = np.load(str(candidate_npz_path), allow_pickle=True)
        des_c = data.get("des")
        if des_c is None or len(des_c) == 0:
            return False, 0
        des_c = des_c.astype(np.uint8)
    except Exception:
        return False, 0
    # match with BFMatcher Hamming
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(qd.astype(np.uint8), des_c, k=2)
    good = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < ratio * n.distance:
                good.append(m)
    return (len(good) >= min_good), len(good)
