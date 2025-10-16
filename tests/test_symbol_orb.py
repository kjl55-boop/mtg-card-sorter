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
source_dir = Path("data/debug")
crop_dir = Path("data/symbol_test/crops")
crop_dir.mkdir(parents=True, exist_ok=True)

# Load reference symbol DB
symbol_db = load_symbol_db(Path("data/mana_symbols_png"))

def preprocess_symbol(symbol_img: np.ndarray, size: int = 96, pad_ratio: float = 0.25, use_mask: bool = False) -> np.ndarray:
    gray = cv2.cvtColor(symbol_img, cv2.COLOR_BGR2GRAY) if symbol_img.ndim == 3 else symbol_img
    h, w = gray.shape
    target_inner = int(size * (1 - pad_ratio))
    scale = target_inner / max(h, w)
    resized = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    pad_h = size - resized.shape[0]
    pad_w = size - resized.shape[1]
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    enhanced = cv2.equalizeHist(padded)

    if use_mask:
        mask = np.zeros_like(enhanced)
        center = (size // 2, size // 2)
        radius = size // 2
        cv2.circle(mask, center, radius, 255, -1)
        enhanced = cv2.bitwise_and(enhanced, enhanced, mask=mask)

    return enhanced

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
            processed = preprocess_symbol(symbol_img, size=96, pad_ratio=0.25, use_mask=False)
            crop_path = crop_dir / f"{img_path.stem}_symbol_{i}.png"
            cv2.imwrite(str(crop_path), processed)
            log.info("  Saved symbol %d → %s (%dx%d)", i+1, crop_path.name, processed.shape[1], processed.shape[0])

    except Exception as e:
        log.warning("  Cropping failed: %s", str(e))

# Step 2: Run ORB matching on saved crops
crop_files = sorted(crop_dir.glob("*.png")) + sorted(crop_dir.glob("*.jpg"))
log.info("Starting ORB matching on %d symbol crops", len(crop_files))

for crop_path in crop_files:
    log.info("Testing image: %s", crop_path.name)
    image = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        log.warning("Failed to load crop: %s", crop_path.name)
        continue

    try:
        orb = cv2.ORB_create(nfeatures=1000)
        kp, des = orb.detectAndCompute(image, None)
        log.info("  Keypoints detected: %d", len(kp) if kp else 0)

        matches = match_mana_symbols(image, symbol_db)
        log.info("  %d candidates", len(matches))
        if matches:
            top = matches[0]
            log.info("    Best match: %s (%d inliers)", top[0], top[1])
        else:
            log.info("    No match found")

        # Optional: visualize keypoints
        # debug_img = cv2.drawKeypoints(image, kp, None, color=(0,255,0), flags=0)
        # cv2.imshow("Keypoints", debug_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    except Exception as e:
        log.warning("  ORB matching failed: %s", str(e))

log.info("Symbol ORB test complete. Results saved to %s", log_file)
