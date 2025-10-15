"""
config.paths

Expose common paths and ensure data folders exist.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SCRYFALL_DIR = DATA_DIR / "scryfall_db"
DESCRIPTORS_DIR = SCRYFALL_DIR / "descriptors"
DEBUG_DIR = DATA_DIR / "debug"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"

def ensure_directories():
    for p in (DATA_DIR, SCRYFALL_DIR, DESCRIPTORS_DIR, DEBUG_DIR, RESULTS_DIR, LOGS_DIR):
        p.mkdir(parents=True, exist_ok=True)

__all__ = [
    "ROOT",
    "DATA_DIR",
    "SCRYFALL_DIR",
    "DESCRIPTORS_DIR",
    "DEBUG_DIR",
    "RESULTS_DIR",
    "LOGS_DIR",
]
