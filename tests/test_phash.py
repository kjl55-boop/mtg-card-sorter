import os
import cv2
import sys
from pathlib import Path

# Add the app/ directory to Python's module search path
sys.path.append(str(Path(__file__).resolve().parent.parent / "app"))

from recognizer.matcher import Matcher
from recognizer.phash import compute_phash_from_gray, match_phash
from config.config import CONFIG
from recognizer.ocr import ocr_image, crop_title_band
from recognizer.preprocess import preprocess_for_ocr
import logging

# Setup logging to file
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
matcher = Matcher(config={**CONFIG.__dict__, "phash_threshold": 10, "orb_min_matches": 0})

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

    # Compute and log phash
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    phash = compute_phash_from_gray(gray)
    log.info("  Computed phash: %s", phash)

    # OCR with tuned preprocessing
    try:
        title_band = crop_title_band(image)
        proc = preprocess_for_ocr(cv2.cvtColor(title_band, cv2.COLOR_BGR2GRAY), scale=2.0, debug=False)
        text, conf = ocr_image(proc, tesseract_config="--oem 1 --psm 6")
        log.info("  OCR result: '%s' (confidence: %d)", text.strip(), conf)
    except Exception as e:
        log.warning("  OCR failed: %s", str(e))
        text, conf = "", 0

    # Match
    result = matcher.match_once(image)
    if result is None:
        log.warning("  Matcher returned None — index may be empty or invalid")
        continue

    # Optional: log top-k candidates for tuning
    candidates = match_phash(phash, matcher._index, top_k=5, threshold=20)
    for i, (cid, meta, dist) in enumerate(candidates):
        log.info("  Candidate %d: id=%s name=%s dist=%d", i+1, cid, meta.get("name", "unknown"), dist)

    # Log match result
    if result.success:
        match_count += 1
        total_dist += result.dist or 0
        log.info("  Match ID: %s", result.id)
        log.info("  Name: %s", result.meta.get("name", "unknown"))
        log.info("  Distance: %d", result.dist)
    else:
        log.info("  No match found (best dist: %s)", result.dist)

    # Combined summary per image
    log.info("  Summary → OCR: '%s' (conf: %d) | pHash: %s | Match: %s (dist: %s)",
             text.strip(), conf, phash, result.id if result.success else "None", result.dist)

# Final summary
log.info("Phash test complete. Results saved to %s", log_file)
log.info("Total images tested: %d", len(image_files))
log.info("Images with matches: %d", match_count)
if match_count > 0:
    avg_dist = total_dist / match_count
    log.info("Average top match distance: %.2f", avg_dist)
