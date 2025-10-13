# app/matcher.py
import cv2
import numpy as np
import pickle
from pathlib import Path
from . import config, init as app_init, utils

# DB file naming
DB_INDEX_FILE = app_init.DESCRIPTORS_DIR / "phash_index.pkl"

# Tunable parameters
PHASH_SIZE = 32            # phash block size (higher => more discriminative, slower)
PHASH_DIST_THRESHOLD = 10  # hamming threshold for an initial match
TOP_K = 5                  # number of candidate matches to return to verifier
ORB_MIN_MATCHES = 8        # minimum good ORB matches to consider a verification success

# Ensure descriptors dir exists
app_init.DESCRIPTORS_DIR.mkdir(parents=True, exist_ok=True)

# --- Preprocessing and hashing ------------------------------------------------
def preprocess_for_phash(img, size=256, clahe=True, blur_ksize=(3, 3), crop_margin_pct=0.02):
    """Return a single-channel uint8 image prepared for phash."""
    if img is None:
        raise ValueError("input image is None")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mpx = int(min(h, w) * crop_margin_pct)
    if mpx > 0 and h - 2 * mpx > 0 and w - 2 * mpx > 0:
        gray = gray[mpx:h - mpx, mpx:w - mpx]
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    if blur_ksize:
        gray = cv2.GaussianBlur(gray, blur_ksize, 0)
    if clahe:
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe_obj.apply(gray)
    return gray

def compute_phash_opencv(img_gray, hash_size=PHASH_SIZE):
    """Return a 1D uint8 array representing the phash."""
    # OpenCV's img_hash.PHash_create expects CV_8U single channel images
    phash = cv2.img_hash.PHash_create(hash_size=hash_size)
    h = phash.compute(img_gray)
    if h is None:
        raise RuntimeError("phash computation failed")
    return np.asarray(h).flatten().astype(np.uint8)

def hamming_distance_bytes(a, b):
    """Count differing bits between two uint8 arrays."""
    a = np.asarray(a, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8)
    if a.shape != b.shape:
        raise ValueError("hash shapes differ")
    xor = np.bitwise_xor(a, b)
    # count set bits
    return int(np.unpackbits(xor).sum())

# --- DB load/save -------------------------------------------------------------
def save_index(index, path=DB_INDEX_FILE):
    with open(path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_index(path=DB_INDEX_FILE):
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)

def build_index_from_folder(images_folder, out_path=DB_INDEX_FILE, preprocess_size=256):
    """
    Build a phash index from a folder of images.
    images_folder should contain image files named with an identifier that maps back to Scryfall metadata.
    Stored index format: {id: {"phash": bytes, "meta": {...}}}
    """
    images_folder = Path(images_folder)
    index = {}
    for img_path in images_folder.glob("*.png"):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            gray = preprocess_for_phash(img, size=preprocess_size)
            ph = compute_phash_opencv(gray)
            # meta extraction: use file stem as id; you can extend to load companion metadata
            idx = img_path.stem
            index[idx] = {"phash": ph, "meta": {"path": str(img_path)}}
        except Exception as e:
            # keep building; log via utils if available
            try:
                utils.log_exception(e)
            except Exception:
                pass
    save_index(index, out_path)
    return index

# --- ORB verifier -------------------------------------------------------------
def orb_verify(img1, img2, min_matches=ORB_MIN_MATCHES):
    """Return (good_matches, inlier_count). Both imgs are BGR; we compute ORB keypoints and BF matches."""
    orb = cv2.ORB_create(2000)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    if des1 is None or des2 is None:
        return 0, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    # ratio test
    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < min_matches:
        return len(good), 0
    # estimate homography inliers if possible
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    try:
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is None:
            return len(good), 0
        inliers = int(mask.sum())
        return len(good), inliers
    except Exception:
        return len(good), 0

# --- Matching API -------------------------------------------------------------
def match_card(card_bgr):
    """
    Attempt to match a card image (BGR) to the DB.
    Returns either None or (meta_record, score, good_matches, inliers)
    Score is the Hamming distance; lower is better.
    """
    index = load_index()
    if not index:
        raise FileNotFoundError("phash index not found; build it via build_index_from_folder")

    query_gray = preprocess_for_phash(card_bgr, size=256)
    qph = compute_phash_opencv(query_gray)

    # Compute distances
    candidates = []
    for key, rec in index.items():
        dbph = rec["phash"]
        dist = hamming_distance_bytes(qph, dbph)
        candidates.append((key, rec, dist))
    candidates.sort(key=lambda x: x[2])  # sort by distance asc

    # quick accept if top candidate is below primary threshold
    top_candidates = candidates[:TOP_K]
    best_key, best_rec, best_dist = top_candidates[0]
    if best_dist <= PHASH_DIST_THRESHOLD:
        # do optional ORB verification to reduce false positives
        try:
            db_img_path = Path(best_rec["meta"]["path"])
            db_img = cv2.imread(str(db_img_path)) if db_img_path.exists() else None
            if db_img is not None:
                good, inliers = orb_verify(card_bgr, db_img)
            else:
                good, inliers = 0, 0
        except Exception:
            good, inliers = 0, 0
        return best_rec["meta"], int(best_dist), int(good), int(inliers)

    # no match found within threshold
    return None

# Expose utilities for CLI or programmatic use
__all__ = [
    "compute_phash_opencv",
    "preprocess_for_phash",
    "build_index_from_folder",
    "load_index",
    "save_index",
    "match_card",
]
