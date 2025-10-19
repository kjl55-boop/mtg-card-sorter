"""
Runtime flags for overriding config defaults via CLI.
Use this in scripts like run_inspector.py to allow dynamic tuning.
"""

import argparse

def parse_flags():
    parser = argparse.ArgumentParser(description="Runtime flags for MTG card inspector")

    # Logging
    parser.add_argument("--log-level", type=str, default=None, help="Logging level (e.g., DEBUG, INFO)")

    # Camera
    parser.add_argument("--preview-width", type=int, default=None, help="Camera preview width")
    parser.add_argument("--preview-height", type=int, default=None, help="Camera preview height")
    parser.add_argument("--display-scale", type=float, default=None, help="Scale factor for display window")

    # Matching
    parser.add_argument("--phash-threshold", type=int, default=None, help="Max Hamming distance for phash match")
    parser.add_argument("--match-attempts", type=int, default=None, help="Number of match retries")
    parser.add_argument("--phash_size", type=int, default=None, help="phash size must be [8, 16, 32]")
    parser.add_argument("--match_top_k", type=int, default=None, help="must be a positive integer")

    # Output
    parser.add_argument("--debug-dir", type=str, default=None, help="Directory to save debug crops")
    parser.add_argument("--save-crops", action="store_true", help="Enable saving cropped card images")

    return parser.parse_args()
