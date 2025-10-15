import os
import cv2
from pathlib import Path
from card_inspector.matcher import Matcher
from card_inspector.phash import compute_phash
from card_inspector.config import load_config
import logging

# Setup logging to file
log_file = "phash_test_results.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()  # Optional: keep this if you want to see progress in terminal
    ]
)
log = logging.getLogger("phash_test")

# Load matcher
config = load_config()
matcher = Matcher(config)

# Folder to scan
debug_dir = Path("data/debug")
image_files = sorted(debug_dir.glob("*.png")) + sorted(debug_dir.glob("*.jpg"))

if not image_files:
    log.warning("No images found in %s", debug_dir)
    exit()

for img_path in image_files:
    log.info("Testing image: %s", img_path.name)
    image = cv2.imread(str(img_path))
    if image is None:
        log.warning("Failed to load image: %s", img_path.name)
        continue

    top_matches = matcher.debug_match(image, top_k=5)
    if not top_matches:
        log.info("No phash candidates found for %s", img_path.name)
        continue

    for i, (match_id, match_meta, dist) in enumerate(top_matches):
        name = match_meta.get("name", "unknown")
        log.info("  Rank %d: id=%s name=%s dist=%d", i + 1, match_id, name, dist)

log.info("Phash test complete. Results saved to %s", log_file)
