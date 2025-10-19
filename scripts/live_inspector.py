#!/usr/bin/env python3
"""
Run a phash inspection over images in data/debug using the canonical Matcher.
Saves logs in CONFIG.logs_dir/diagnostics and visual comparisons to debug_images.
"""

import logging
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

from config.config import CONFIG
from pipeline import utils
from recognizer.phash.phash import Matcher
from recognizer.preprocess import preprocess_for_phash

# Logging setup
LOG_DIR = Path(CONFIG.logs_dir) / "diagnostics"
LOG_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = LOG_DIR / f"run_inspector_{timestamp}.log"

logging.basicConfig(
    level=getattr(logging, CONFIG.log_level, logging.INFO),
    format=CONFIG.log_format,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(log_path), mode="w")
    ]
)
log = logging.getLogger("diagnostics.run_inspector")
log.info("Logging initialized at %s", log_path)
log.info("Using game profile: %s", CONFIG.game_profile.name)
log.info("Phash index path: %s", CONFIG.game_profile.index_path)

# Build preprocess kwargs from game profile + CONFIG defaults
gp = getattr(CONFIG, "game_profile", None)
gp_pp = gp.preprocess_config if gp and hasattr(gp, "preprocess_config") else {}
pp_kwargs = {
    "out_size": int(getattr(CONFIG, "phash_out_size", 256)),
    "clahe": bool(getattr(CONFIG, "phash_clahe", True)),
    "blur_ksize": tuple(getattr(CONFIG, "phash_blur_ksize", (3, 3))),
    "crop_margin_pct": float(getattr(CONFIG, "phash_crop_margin_pct", 0.02)),
    "highpass": bool(getattr(CONFIG, "phash_highpass", False)),
    "debug": False
}
# override with game-profile values if present
pp_kwargs.update(gp_pp)

# Create matchers
base_cfg = vars(CONFIG)
matcher = Matcher(config=base_cfg)
# tmp matcher for relaxed acceptance during inspection
tmp_cfg = base_cfg.copy()
tmp_cfg["phash_threshold"] = max(64, tmp_cfg.get("phash_threshold", 64))
tmp_matcher = Matcher(config=tmp_cfg)

log.info("Matcher config -> phash_size=%d, top_k=%d, threshold=%d, verify_title=%s",
         matcher.phash_size, matcher.top_k, matcher.threshold, matcher.verify_title)
log.info("Temporary matcher threshold=%d", tmp_matcher.threshold)

# Helpers
DEBUG_IMG_DIR = LOG_DIR / "debug_images"
DEBUG_IMG_DIR.mkdir(parents=True, exist_ok=True)

def save_side_by_side(query_img: np.ndarray, candidate_path: Path, out_path: Path):
    cand = cv2.imread(str(candidate_path))
    if cand is None:
        return None
    h = cand.shape[0]
    q_h, q_w = query_img.shape[:2]
    new_w = max(1, int(q_w * (h / q_h)))
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

def save_debug_img(img: np.ndarray, tag: str, src_name: str) -> Path:
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{src_name}_{tag}.png"
    out_path = DEBUG_IMG_DIR / filename
    cv2.imwrite(str(out_path), img)
    return out_path

# Collect image files
debug_dir = Path("data/debug")
image_files = sorted(debug_dir.glob("*.png")) + sorted(debug_dir.glob("*.jpg")) + sorted(debug_dir.glob("*.jpeg"))

if not image_files:
    log.warning("No debug images found in %s", debug_dir)
    raise SystemExit(1)

stats = {
    "total": 0,
    "matched": 0,
    "accepted": 0,
    "dists": []
}

for img_path in image_files:
    stats["total"] += 1
    log.info("Testing image: %s", img_path.name)
    img = cv2.imread(str(img_path))
    if img is None:
        log.warning("Failed to load image: %s", img_path.name)
        continue

    # Compute query phash explicitly for logging parity-check
    try:
        pre = preprocess_for_phash(img, **pp_kwargs)
        pre_img = pre.final if hasattr(pre, "final") else pre
    except Exception as e:
        log.warning("Preprocessing failed for %s: %s", img_path.name, e)
        pre_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    try:
        qph = matcher.compute_phash_from_gray(pre_img)
        log.info("  Query phash: %s", qph)
    except Exception as e:
        log.warning("  Failed to compute query phash: %s", e)
        continue

    # Distance distribution across index (top 20)
    from imagehash import hex_to_hash
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
                continue
            dist_list.append((cid, int(qh - dbh)))
        dist_list.sort(key=lambda x: x[1])

    if dist_list:
        top20 = ", ".join(f"{c[:8]}:{d}" for c, d in dist_list[:20])
        log.info("  Top distances (top 20): %s", top20)
    else:
        log.info("  No comparable phash entries (shape mismatch or empty phash)")

    # Run relaxed tmp matcher for inspection
    tmp_res = tmp_matcher.match_with_policy(img, preprocess_fn=None, preprocess_kwargs=pp_kwargs)
    if tmp_res:
        log.info("  [tmp_matcher] Best candidate id=%s name=%s dist=%s (tmp_threshold=%d)",
                 tmp_res.get("id"), tmp_res.get("meta", {}).get("name", "unknown"), tmp_res.get("dist"), tmp_matcher.threshold)
    else:
        log.info("  [tmp_matcher] No candidate within tmp threshold=%d", tmp_matcher.threshold)

    # Run canonical matcher (uses pp_defaults merged with pp_kwargs)
    res = matcher.match_with_policy(img, preprocess_fn=None, preprocess_kwargs=pp_kwargs)
    if res is None:
        log.warning("No candidates found for %s (canonical matcher)", img_path.name)
        dbg = save_debug_img(img, "no_candidates", img_path.stem)
        log.info("  Saved debug image: %s", dbg)
        continue

    stats["matched"] += 1
    best_dist = res.get("dist")
    accepted = res.get("dist") <= matcher.threshold
    if accepted:
        stats["accepted"] += 1
    stats["dists"].append(best_dist)

    log.info("  Match ID: %s", res.get("id"))
    log.info("  Name: %s", res.get("meta", {}).get("name", "unknown"))
    log.info("  Distance: %d", best_dist)
    log.info("  Accepted: %s (threshold=%d)", accepted, matcher.threshold)

    # Save debug image on failure to accept if configured
    if not accepted:
        dbg = save_debug_img(img, "failed_accept", img_path.stem)
        log.info("  Saved debug image: %s", dbg)

    # Save top-5 side-by-side comparisons
    for i, (cid, dist) in enumerate(dist_list[:5]):
        meta_path = matcher.index.get(cid, {}).get("meta", {}).get("path", "")
        candidate_path = Path(meta_path)
        if candidate_path.exists():
            out = DEBUG_IMG_DIR / f"{img_path.stem}_vs_{cid[:8]}.png"
            saved = save_side_by_side(img, candidate_path, out)
            if saved:
                log.info("  Saved visual compare #%d: %s (dist=%d)", i+1, saved.name, dist)

# Summary
log.info("Run complete.")
log.info("Total images tested: %d", stats["total"])
log.info("Images with any candidate: %d", stats["matched"])
log.info("Images accepted (<= threshold): %d", stats["accepted"])
if stats["dists"]:
    import numpy as _np
    d = _np.array(stats["dists"])
    log.info("Distance percentiles: 50=%.1f 75=%.1f 90=%.1f 95=%.1f",
             float(_np.percentile(d, 50)), float(_np.percentile(d, 75)),
             float(_np.percentile(d, 90)), float(_np.percentile(d, 95)))
log.info("Detailed logs and debug images saved to %s", LOG_DIR)
