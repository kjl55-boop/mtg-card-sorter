#!/usr/bin/env python3
"""
Batch phash inspection using images in data/debug.
Logs match results, query phash, top distances across index, and saves visual diffs for top candidates.
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from imagehash import hex_to_hash

from config.config import CONFIG
from recognizer.phash import Matcher

# ─────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────

CONFIG.logs_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = CONFIG.logs_dir / "diagnostics"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f"inspect_phash_{timestamp}.log"

logging.basicConfig(
    level=CONFIG.log_level,
    format=CONFIG.log_format,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path, mode="w")
    ]
)

log = logging.getLogger("diagnostics.inspect_phash")
log.info("Logging initialized at %s", log_path)
log.info("Using game profile: %s", CONFIG.game_profile.name)
log.info("Phash index path: %s", CONFIG.game_profile.index_path)

# ─────────────────────────────────────────────────────────────
# Matcher Setup
# ─────────────────────────────────────────────────────────────

# CONFIG is a SimpleNamespace — convert to dict for Matcher
matcher = Matcher(config=vars(CONFIG))
log.info("Matcher config → phash_size=%d, top_k=%d, threshold=%d, verify_title=%s",
         matcher.phash_size, matcher.top_k, matcher.threshold, matcher.verify_title)

# Basic index sanity
total_index = len(matcher.index)
valid_phash_count = sum(1 for r in matcher.index.values() if r.get("phash"))
log.info("Index entries total=%d valid_phash=%d", total_index, valid_phash_count)

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def save_side_by_side(query_img: np.ndarray, candidate_path: Path, out_path: Path):
    cand = cv2.imread(str(candidate_path))
    if cand is None:
        return None
    # Resize query to match candidate height while preserving aspect ratio
    h = cand.shape[0]
    q_h, q_w = query_img.shape[:2]
    new_w = int(q_w * (h / q_h))
    q_resized = cv2.resize(query_img, (new_w, h))
    # pad widths to match
    if q_resized.shape[1] < cand.shape[1]:
        pad = cand.shape[1] - q_resized.shape[1]
        q_resized = cv2.copyMakeBorder(q_resized, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=[0,0,0])
    elif q_resized.shape[1] > cand.shape[1]:
        cand = cv2.copyMakeBorder(cand, 0, 0, 0, q_resized.shape[1] - cand.shape[1], cv2.BORDER_CONSTANT, value=[0,0,0])
    side = np.hstack([q_resized, cand])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), side)
    return out_path

# ─────────────────────────────────────────────────────────────
# Files to process
# ─────────────────────────────────────────────────────────────

debug_dir = Path("data/debug")
image_files = sorted(debug_dir.glob("*.png")) + sorted(debug_dir.glob("*.jpg"))

if not image_files:
    log.warning("No images found in %s", debug_dir)
    exit()

match_count = 0
total_dist = 0

# ─────────────────────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────────────────────

for img_path in image_files:
    log.info("Testing image: %s", img_path.name)
    image = cv2.imread(str(img_path))
    if image is None:
        log.warning("Failed to load image: %s", img_path.name)
        continue

    # Compute and log query phash explicitly for parity checks
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    qph = matcher.compute_phash_from_gray(gray)
    log.info("  Query phash: %s", qph)

    # Compute distance distribution across index (top N)
    try:
        qh = hex_to_hash(qph)
    except Exception as e:
        log.warning("  Failed to parse query phash: %s", e)
        qh = None

    dist_list = []
    if qh is not None:
        for cid, rec in matcher.index.items():
            dbp = rec.get("phash")
            if not dbp:
                continue
            try:
                dbh = hex_to_hash(dbp)
            except Exception:
                continue
            if qh.hash.shape != dbh.hash.shape:
                # skip shape mismatches (likely phash_size mismatch)
                continue
            dist_list.append((cid, int(qh - dbh)))
        dist_list.sort(key=lambda x: x[1])

    # Log top distances (if any)
    if dist_list:
        top_lines = ", ".join(f"{c}:{d}" for c, d in dist_list[:10])
        log.info("  Top distances (top 10): %s", top_lines)
    else:
        log.info("  No comparable phash entries (shape mismatch or empty phash fields)")

    # Run the matcher policy (preprocessing, optional title verify)
    result = matcher.match_with_policy(image)
    if result is None:
        log.warning("No candidates found for %s", img_path.name)
        # save debug crop of the query itself for later inspection
        if matcher.config.get("save_debug_on_failure", True):
            dbg_path = matcher._save_debug(image, f"no_candidates_{img_path.stem}")
            log.info("  Saved debug image: %s", dbg_path)
        continue

    match_id = result.get("id")
    match_dist = result.get("dist")
    match_name = result.get("meta", {}).get("name", "unknown")

    # Accept match if within threshold
    if match_id and match_dist is not None and match_dist <= matcher.threshold:
        match_count += 1
        total_dist += match_dist
        log.info("  Match ID: %s", match_id)
        log.info("  Name: %s", match_name)
        log.info("  Distance: %d", match_dist)
    else:
        log.info("  No match within threshold (best dist: %s)", match_dist)
        if matcher.config.get("save_debug_on_failure", True):
            dbg_path = matcher._save_debug(image, f"failed_{img_path.stem}")
            log.info("  Saved debug crop: %s", dbg_path)

    # If there are top candidates from dist_list, save side-by-side comparisons for the top 3
    for i, (cid, dist) in enumerate(dist_list[:3]):
        meta_path_str = matcher.index.get(cid, {}).get("meta", {}).get("path", "")
        candidate_path = Path(meta_path_str)
        if candidate_path.exists():
            out = CONFIG.logs_dir / "diagnostics" / f"{img_path.stem}_vs_{cid}.png"
            saved = save_side_by_side(image, candidate_path, out)
            if saved:
                log.info("  Saved visual compare #%d: %s (dist=%d)", i+1, saved.name, dist)

    # Log title band dist if available
    if matcher.verify_title and result.get("title_dist") is not None:
        log.info("  Title band distance: %s", result.get("title_dist"))

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────

log.info("Phash inspection complete. Results saved to %s", log_path)
log.info("Total images tested: %d", len(image_files))
log.info("Images with matches: %d", match_count)
if match_count > 0:
    avg_dist = total_dist / match_count
    log.info("Average top match distance: %.2f", avg_dist)
