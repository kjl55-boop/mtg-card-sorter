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
from recognizer.orb import load_symbol_db, match_mana_symbols
from recognizer.crop import crop_mana_cost
from recognizer.symbol import isolate_mana_symbols
import logging
import recognizer.symbol
print("Using symbol.py from:", recognizer.symbol.__file__)


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

# Load mana symbol reference set
symbol_db = load_symbol_db(Path("data/mana_symbols_png"))

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

    # Isolate and match individual mana symbols
    try:
        mana_crop = crop_mana_cost(image)
        symbol_crops = isolate_mana_symbols(mana_crop, debug=False)

        if not symbol_crops:
            log.info("  No symbols isolated from mana band")
        else:
            # Combine all crops into one horizontal strip
            strip = cv2.hconcat(symbol_crops)

            # Ensure mana_crop is grayscale and matches strip width and type
            mana_gray = cv2.cvtColor(mana_crop, cv2.COLOR_BGR2GRAY) if mana_crop.ndim == 3 else mana_crop
            mana_resized = cv2.resize(mana_gray, (strip.shape[1], strip.shape[0]), interpolation=cv2.INTER_AREA)

            # Ensure both are same type and shape
            mana_resized = mana_resized.astype(np.uint8)
            strip = strip.astype(np.uint8)

            combined = cv2.vconcat([mana_resized, strip])


            cv2.imshow("Mana Band + Symbols", combined)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            for i, symbol_img in enumerate(symbol_crops):
                matches = match_mana_symbols(symbol_img, symbol_db)
                if matches:
                    top = matches[0]
                    log.info("  Symbol %d → Best match: %s (%d inliers)", i+1, top[0], top[1])
                else:
                    log.info("  Symbol %d → No match found", i+1)
    except Exception as e:
        log.warning("  Symbol isolation/matching failed: %s", str(e))

    # Match
    result = matcher.match_once(image)
    if result is None:
        log.warning("  Matcher returned None — index may be empty or invalid")
        continue

    # Optional: log top-k candidates for tuning
    candidates = match_phash(phash, matcher._index, top_k=5, threshold=20)
    #for i, (cid, meta, dist) in enumerate(candidates):
    #    log.info("  Candidate %d: id=%s name=%s dist=%d", i+1, cid, meta.get("name", "unknown"), dist)

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
