"""
Configuration constants for the project.
Edit values here to tune behavior across modules.
"""

from . import paths

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"  # ✅ could be a runtime flag
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# ─────────────────────────────────────────────────────────────
# Camera Settings
# ─────────────────────────────────────────────────────────────
CAMERA_PREVIEW_SIZE = (2304, 1296)  # ✅ could be a runtime flag
CAMERA_TIMEOUT = 1.0
AUTOFOCUS_ENABLED = True
CAMERA_WARMUP_SEC = 0.5
DISPLAY_SCALE = 0.5  # ✅ could be a runtime flag

# ─────────────────────────────────────────────────────────────
# Matching / Hashing
# ─────────────────────────────────────────────────────────────
PHASH_SIZE = 16
PHASH_THRESHOLD = 16
PHASH_TOP_K = 12
PHASH_OUT_SIZE = 256
PHASH_CLAHE = True
PHASH_BLUR_KSIZE = (3, 3)
PHASH_CROP_MARGIN_PCT = 0.02
PHASH_HIGHPASS = False


# ─────────────────────────────────────────────────────────────
# Cropping Defaults
# ─────────────────────────────────────────────────────────────
DEFAULT_MIN_AREA = 5000
DEFAULT_TOP_PCT = 0.12
DEFAULT_MID_START_PCT = 0.55
DEFAULT_MID_END_PCT = 0.63
DEFAULT_BOTTOM_PCT = 0.78
DEFAULT_PAD_X_PCT = 0.0
DEFAULT_PAD_Y_PCT = 0.0

# ─────────────────────────────────────────────────────────────
# Output / Normalization
# ─────────────────────────────────────────────────────────────
NORMALIZED_SIZE = (256, 356)  # ✅ could be a runtime flag
SAVE_CROPS_ENABLED = True

# ─────────────────────────────────────────────────────────────
# Derived Paths
# ─────────────────────────────────────────────────────────────
SCRYFALL_DB_DIR = paths.SCRYFALL_DIR
DESCRIPTORS_DIR = paths.DESCRIPTORS_DIR
DEBUG_DIR = paths.DEBUG_DIR
RESULTS_DIR = paths.RESULTS_DIR
LOGS_DIR = paths.LOGS_DIR
