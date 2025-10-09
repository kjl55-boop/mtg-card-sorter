"""
Configuration constants for the project.
Edit values here to tune behavior across modules.
"""

from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SCRYFALL_RAW = DATA_DIR / "raw_scryfall"
SCRYFALL_DB = DATA_DIR / "scryfall_db"
SCRYFALL_DB.mkdir(parents=True, exist_ok=True)

# DB files (builder will create these)
DB_PATH = SCRYFALL_DB / "cards.db"
DESC_DIR = SCRYFALL_DB / "descriptors"

# Camera / image sizes
CAMERA_PREVIEW_SIZE = (2304, 1296)  # (width, height)
DISPLAY_SCALE = 0.5                 # scale for GUI preview

# ORB / hashing
ORB_FEATURES = 1200
PHASH_HAMMING_THRESHOLD = 6
MATCH_TOP_K = 12

# Crop/snippet defaults (fractions of card height)
DEFAULT_TOP_PCT = 0.12
DEFAULT_MID_START_PCT = 0.48
DEFAULT_MID_END_PCT = 0.58
DEFAULT_BOTTOM_PCT = 0.78

# OCR defaults
TESSERACT_CONFIG_TITLE = "--oem 1 --psm 7"
TESSERACT_CONFIG_SNIPPET = "--oem 1 --psm 6"

# Misc
DEBUG = True
