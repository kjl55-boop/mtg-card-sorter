"""
Configuration constants for the project.
Edit values here to tune behavior across modules.
"""

from pathlib import Path
import app as app_pkg

# Camera / image sizes
CAMERA_PREVIEW_SIZE = (2304, 1296)  # (width, height)
DISPLAY_SCALE = 0.5                 # scale for GUI preview

# ORB / hashing
ORB_FEATURES = 1200
PHASH_HAMMING_THRESHOLD = 6
MATCH_TOP_K = 12

# Crop/snippet defaults (fractions of card height)
DEFAULT_TOP_PCT = 0.12
DEFAULT_MID_START_PCT = 0.55
DEFAULT_MID_END_PCT = 0.63
DEFAULT_BOTTOM_PCT = 0.78

# Derived paths if needed
SCRYFALL_DB_DIR = app_pkg.SCRYFALL_DIR
DESCRIPTORS_DIR = app_pkg.DESCRIPTORS_DIR
DEBUG_DIR = app_pkg.DEBUG_DIR
