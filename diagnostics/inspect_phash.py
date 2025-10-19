#!/usr/bin/env python3
"""
Batch phash inspection using images in data/debug.
Logs match results, distances, and optionally saves failed crops.
"""

import cv2
from pathlib import Path
from datetime import datetime
import logging

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

matcher = Matcher(config=vars(CONFIG))
log.info("Matcher config → phash_size=%d, top_k=%d, threshold=%d, verify_title=%s",
         matcher.phash_size, matcher.top_k, matcher.threshold, matcher.verify_title)

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

    result = matcher.match_with_policy(image)
    if result is None:
        log.warning("No candidates found for %s", img_path.name)
        continue

    match_id = result.get("id")
    match_dist = result.get("dist")
    match_name = result.get("meta", {}).get("name", "unknown")

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

    if matcher.verify_title and result.get("title_dist") is not None:
        log.info("  Title band distance: %d", result["title_dist"])

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────

log.info("Phash inspection complete. Results saved to %s", log_path)
log.info("Total images tested: %d", len(image_files))
log.info("Images with matches: %d", match_count)
if match_count > 0:
    avg_dist = total_dist / match_count
    log.info("Average top match distance: %.2f", avg_dist)
