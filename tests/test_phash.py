import os
import cv2
import sys
from pathlib import Path
import numpy as np

# Add app/ to module path
sys.path.append(str(Path(__file__).resolve().parent.parent / "app"))

from recognizer.phash import Matcher
from recognizer.phash.phash import compute_phash_from_gray
from config.config import CONFIG
import logging

# Setup logging
log_file = str(Path(CONFIG.logs_dir) / "phash_test_results.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("phash_test")

# Load matcher with tuned config
matcher = Matcher(config={**CONFIG.__dict__, "phash_threshold": 10})

# Folder to scan
debug_dir = Path("data/debug")
image_files = sorted(debug_dir.glob("*.png")) + sorted(debug_dir.glob("*.jpg"))

if not image_files:
    log.warning("No images found in %s", debug_dir)
    exit()

match_count = 0
total_dist = 0

for img_path in image_files:
    log.info("Testing image: %s", img_path.name)
    image = cv2.imread(str(img_path))
    if image is None:
        log.warning("Failed to load image: %s", img_path.name)
        continue

    # Compute phash
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    phash = compute_phash_from_gray(gray)
    log.info("  Computed phash: %s", phash)

    # Match full image
    result = matcher.match_once(image)
    if result is None:
        log.warning("  Matcher returned None — index may be empty or invalid")
        continue

    # Log match result
    if result.success:
        match_count += 1
        total_dist += result.dist or 0
        log.info("  Match ID: %s", result.id)
        log.info("  Name: %s", result.meta.get("name", "unknown"))
        log.info("  Distance: %d", result.dist)
    else:
        log.info("  No match found (best dist: %s)", result.dist)

    # Summary
    log.info("  Summary → pHash: %s | Match: %s (dist: %s)",
             phash, result.id if result.success else "None", result.dist)

# Final summary
log.info("Phash test complete. Results saved to %s", log_file)
log.info("Total images tested: %d", len(image_files))
log.info("Images with matches: %d", match_count)
if match_count > 0:
    avg_dist = total_dist / match_count
    log.info("Average top match distance: %.2f", avg_dist)
