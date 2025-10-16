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
Path("logs").mkdir(parents=True, exist_ok=True)
log_file = str(Path("logs") / "symbol_orb_test.log")
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
source_dir = Path("data/debug")  # Full card images
crop_dir = Path("data/symbol_test/crops")
crop_dir.mkdir(parents=True, exist_ok=True)

# Load reference symbol DB
symbol_db = load_symbol_db(Path("data/mana_symbols_png"))

def circular_crop(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = min(center[0], center[1])
    cv2.circle(mask, center, radius, 255, -1)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

# Step 1: Crop and save symbols
image_files = sorted(source_dir.glob("*.png")) + sorted(source_dir.glob("*.jpg"))
if not image_files:
    log.warning("No source images found in %s", source_dir)
    exit()

for img_path in image_files:
    log.info("Cropping symbols from: %s", img_path.name)
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
            # Resize with anti-aliasing
            resized = cv2.resize(symbol_img, (64, 64), interpolation=cv2.INTER_CUBIC)

            # Apply circular mask
            circled = circular_crop(resized)

            # Save with transparency
            alpha = cv2.cvtColor(circled, cv2.COLOR_GRAY2BGR)
            mask = np.zeros((64, 64), dtype=np.uint8)
            cv2.circle(mask, (32, 32), 32, 255, -1)
            rgba = cv2.merge([circled, circled, circled, mask])

            crop_path = crop_dir / f"{img_path.stem}_symbol_{i}.png"
            cv2.imwrite(str(crop_path), rgba)
            log.info("  Saved symbol %d -> %s", i+1, crop_path.name)

    except Exception as e:
        log.warning("  Cropping failed: %s", str(e))

# Step 2: Run ORB matching on saved crops
crop_files = sorted(crop_dir.glob("*.png")) + sorted(crop_dir.glob("*.jpg"))
log.info("Starting ORB matching on %d symbol crops", len(crop_files))

for crop_path in crop_files:
    log.info("Testing image: %s", crop_path.name)
    image = cv2.imread(str(crop_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        log.warning("Failed to load crop: %s", crop_path.name)
        continue

    try:
        # Use grayscale and enhance contrast
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY) if image.shape[2] == 4 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        orb = cv2.ORB_create(nfeatures=1000)
        kp, des = orb.detectAndCompute(blurred, None)
        log.info("  Keypoints detected: %d", len(kp) if kp else 0)

        matches = match_mana_symbols(blurred, symbol_db)
        log.info("  %d candidates", len(matches))
        if matches:
            top = matches[0]
            log.info("    Best match: %s (%d inliers)", top[0], top[1])
        else:
            log.info("    No match found")

    except Exception as e:
        log.warning("  ORB matching failed: %s", str(e))

log.info("Symbol ORB test complete. Results saved to %s", log_file)
