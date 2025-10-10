"""
Central configuration used across capture, crop, matcher, and run_inspector.
"""
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SCRYFALL_DIR = DATA_DIR / "scryfall_db"
DESCRIPTORS_DIR = SCRYFALL_DIR / "descriptors"
DEBUG_DIR = DATA_DIR / "debug"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"

# Normalized crop size (width, height)
NORMALIZED_SIZE: Tuple[int, int] = (400, 560)

# PHASH thresholds
PHASH_STRICT_THRESHOLD = 6
PHASH_RELAXED_THRESHOLD = 8
PHASH_CANDIDATE_THRESHOLD = 12

# Descriptor verification parameters (ORB)
DESCRIPTOR_MIN_GOOD_MATCHES = 10
DESCRIPTOR_RATIO_TEST = 0.75

# I/O limits
PHASH_TOP_N_CANDIDATES = 10

# Camera defaults (last-good settings)
CAMERA_PREVIEW_SIZE = (1280, 720)
CAMERA_WARMUP_SEC = 0.25

# Ensure existence
for p in (DATA_DIR, SCRYFALL_DIR, DESCRIPTORS_DIR, DEBUG_DIR, RESULTS_DIR, LOGS_DIR):
    p.mkdir(parents=True, exist_ok=True)
