#!/usr/bin/env python3
"""
Batch test for phash matching using images in data/debug.
"""

import cv2
from pathlib import Path
from datetime import datetime
import logging

from config.config import CONFIG
from recognizer.phash import Matcher

# ─────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────

CONFIG.logs_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = CONFIG.logs_dir / "phash_test"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f"phash_test_{timestamp}.log"

logging.basicConfig(
    level=CONFIG.log_level,
    format=CONFIG.log_format,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path, mode="w")
    ]
)

