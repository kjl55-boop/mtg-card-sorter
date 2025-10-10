"""
Simplified inspector entrypoint that:
 - captures a frame
 - normalizes/crops the card (no padding)
 - computes phash and finds top phash candidates
 - verifies candidates via ORB descriptors
 - prints result and saves normalized debug image
"""

import sys
from pathlib import Path
from .capture import capture_frame, normalize_and_save, compute_phash_bgr
from .matcher import load_db_phashes, top_phash_candidates, verify_candidate_orb
from .config import PHASH_STRICT_THRESHOLD, PHASH_RELAXED_THRESHOLD
from .config import DEBUG_DIR
import cv2

DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def run(camera_index: int = 0):
    try:
        frame = capture_frame(camera_index)
    except Exception as e:
        print("capture error:", e, file=sys.stderr)
        raise

    # normalize and save debug image
    debug_name = "live_capture.png"
    saved = normalize_and_save(frame, debug_name)
    print("Saved normalized crop to", saved)

    # read the normalized image back as BGR (consistent with downstream code)
    img = cv2.imread(str(saved))
    if img is None:
        print("Failed to read saved normalized image", saved, file=sys.stderr)
        return

    # compute phash from saved normalized image
    query_ph = compute_phash_bgr(img)
    print("Query phash:", query_ph)

    # load DB phashes
    rows = load_db_phashes()
    if not rows:
        print("No phashes in DB; falling back to OCR (not implemented here)")
        return

    # find top phash candidates
    candidates = top_phash_candidates(query_ph, rows, top_n=10)
    if not candidates:
        print("No phash candidates found, falling back to OCR")
        return

    # quick accept if strict threshold hit
    d0, cid0, name0, ph0, desc0 = candidates[0]
    print(f"Top phash candidate hamming={d0} id={cid0} name={name0}")
    if d0 <= PHASH_STRICT_THRESHOLD:
        print("Strict phash accept:", cid0, name0)
        return

    # otherwise verify top candidates with descriptor matching
    for d, cid, name, ph, desc_file in candidates:
        print(f"Verifying candidate {cid} {name} (hamming={d})")
        desc_path = Path("data/scryfall_db/descriptors") / (desc_file if desc_file else f"{cid}.npz")
        ok, good = verify_candidate_orb(img, desc_path)
        print("  ORB verify:", ok, "good_matches:", good)
        if ok:
            print("Verified match:", cid, name)
            return
        if d <= PHASH_RELAXED_THRESHOLD:
            print("Candidate within relaxed phash band but not verified; continue checking")

    print("No verified match; running OCR fallback (not implemented here)")


if __name__ == "__main__":
    run()
