import os
import cv2
import sys
from pathlib import Path

# Add the app/ directory to Python's module search path
sys.path.append(str(Path(__file__).resolve().parent.parent / "app"))

from recognizer.matcher import Matcher
from recognizer.phash import compute_phash_from_gray
from config.config import CONFIG
from recognizer.ocr import ocr_image, crop_title_band
import logging

# Setup logging to file
log_file = "phash_test_results.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("phash_test")

# Load matcher
matcher = Matcher(CONFIG)

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

    # Log phash
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    phash = compute_phash_from_gray(gray)
    log.info("  Computed phash: %s", phash)

    # Optional OCR
    try:
        title_band = crop_title_band(image)
        text, conf = ocr_image(title_band)
        log.info("  OCR result: '%s' (confidence: %d)", text.strip(), conf)
    except Exception as e:
        log.warning("  OCR failed: %s", str(e))

    # Match
    top_matches = matcher.debug_match(image, top_k=5)
    if not top_matches:
        log.info("  No phash candidates found for %s", img_path.name)
        continue

    match_count += 1
    total_dist += top_matches[0][2]  # distance of top match

    for i, (match_id, match_meta, dist) in enumerate(top_matches):
        name = match_meta.get("name", "unknown")
        log.info("  Rank %d: id=%s name=%s dist=%d", i + 1, match_id, name, dist)

# Summary
log.info("Phash test complete. Results saved to %s", log_file)
log.info("Total images tested: %d", len(image_files))
log.info("Images with matches: %d", match_count)
if match_count > 0:
    avg_dist = total_dist / match_count
    log.info("Average top match distance: %.2f", avg_dist)
