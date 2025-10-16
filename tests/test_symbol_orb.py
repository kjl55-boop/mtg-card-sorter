import os
import cv2
import numpy as np
from pathlib import Path
import logging
import sys

# Add app/ to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "app"))

from recognizer.crop import crop_mana_cost
from recognizer.symbol import isolate_mana_symbols
from recognizer.orb import load_symbol_db, match_mana_symbols

# Setup logging
log_file = "symbol_orb_test.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("symbol_orb_test")

# Paths
image_dir = Path("data/symbol_test")

# Load reference symbol DB
symbol_db = load_symbol_db(Path("data/mana_symbols_png"))

# Scan images
image_files = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
if not image_files:
    log.warning("No images found in %s", image_dir)
    exit()

for img_path in image_files:
    log.info("Testing image: %s", img_path.name)
    image = cv2.imread(str(img_path))
    if image is None:
        log.warning("Failed to load image: %s", img_path.name)
        continue

    try:
        mana_crop = crop_mana_cost(image)
        symbol_crops = isolate_mana_symbols(mana_crop, debug=False)

        if not symbol_crops:
            log.info("  No symbols isolated from mana band")
            continue

        for i, symbol_img in enumerate(symbol_crops):
            matches = match_mana_symbols(symbol_img, symbol_db)
            log.info("  Symbol %d → %d candidates", i+1, len(matches))
            if matches:
                top = matches[0]
                log.info("    Best match: %s (%d inliers)", top[0], top[1])
            else:
                log.info("    No match found")

    except Exception as e:
        log.warning("  Symbol isolation/matching failed: %s", str(e))

log.info("Symbol ORB test complete. Results saved to %s", log_file)
