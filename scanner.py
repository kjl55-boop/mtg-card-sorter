"""
import cv2
from camera_module import PiCameraCapture
import os
import time
from math import ceil
from pathlib import Path
import pytesseract
import imagehash
from PIL import Image
import numpy as np
import subprocess
from typing import List, Sequence, Optional, Tuple







def _make_label_overlay(tile: np.ndarray, label: str, band_h: int = 28, font_scale: float = 0.6) -> np.ndarray:
    Return tile with a small dark label bar on top (BGR).
    if tile.ndim == 2:
        tile_bgr = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
    else:
        tile_bgr = tile.copy()
    h, w = tile_bgr.shape[:2]
    # draw label background
    cv2.rectangle(tile_bgr, (0, 0), (w, band_h), (20, 20, 20), -1)
    cv2.putText(tile_bgr, label, (8, band_h - 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (220, 220, 220), 1, cv2.LINE_AA)
    return tile_bgr

def _overlay_text_block(tile: np.ndarray, text: str, max_lines: int = 6) -> np.ndarray:
    ""Append a dark text band under the tile and render OCR text lines.""
    if tile.ndim == 2:
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
    h, w = tile.shape[:2]
    band_h = int(h * 0.28)
    img_top = tile.copy()
    band = np.full((band_h, w, 3), 18, dtype=np.uint8)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:max_lines]
    y = 20
    for ln in lines:
        cv2.putText(band, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA)
        y += 18
    combined = np.vstack([img_top, band])
    return combined

def stack_images_grid(scale: float, img_matrix: Sequence[Sequence[np.ndarray]],
                      labels: Optional[Sequence[Sequence[str]]] = None,
                      ocr_text_for_tile: Optional[Tuple[int, int, str]] = None) -> np.ndarray:
    ""
    Scale and stack a 2D array of images into a single tiled image.

    - scale: uniform scale factor for tile resizing (e.g., 0.5)
    - img_matrix: 2D list/tuple of images (rows x cols) where each item is a numpy image (BGR or gray)
    - labels: optional 2D list/tuple of same shape containing labels for each tile
    - ocr_text_for_tile: optional triple (row_idx, col_idx, text) to overlay OCR text under that tile

    Returns a single BGR image suitable for cv2.imshow.
    ""
    if len(img_matrix) == 0 or len(img_matrix[0]) == 0:
        return np.zeros((10,10,3), dtype=np.uint8)

    rows = len(img_matrix)
    cols = len(img_matrix[0])
    # determine reference shape from top-left tile
    ref = img_matrix[0][0]
    if ref is None:
        raise ValueError("Top-left tile must be a valid image")
    ref_h, ref_w = ref.shape[:2]

    # build a list of rows, each row is list of processed tiles
    processed_rows = []
    for r in range(rows):
        processed_row = []
        for c in range(cols):
            tile = img_matrix[r][c]
            if tile is None:
                tile = np.zeros((ref_h, ref_w), dtype=np.uint8)
            # convert to BGR if needed
            if tile.ndim == 2:
                tile_bgr = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
            else:
                tile_bgr = tile.copy()
            # resize preserving aspect to fit reference dimensions scaled by scale
            target_w = max(1, int(ref_w * scale))
            target_h = max(1, int(ref_h * scale))
            ih, iw = tile_bgr.shape[:2]
            scale_local = min(target_w / iw, target_h / ih)
            new_w = max(1, int(iw * scale_local))
            new_h = max(1, int(ih * scale_local))
            tile_resized = cv2.resize(tile_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # pad to (target_h, target_w)
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            top = (target_h - new_h) // 2
            left = (target_w - new_w) // 2
            canvas[top:top+new_h, left:left+new_w] = tile_resized

            # label overlay if provided
            if labels is not None:
                lbl = labels[r][c] if r < len(labels) and c < len(labels[r]) else ""
                if lbl:
                    canvas = _make_label_overlay(canvas, lbl)

            # OCR overlay if this is the designated tile
            if ocr_text_for_tile is not None:
                orow, ocol, otext = ocr_text_for_tile
                if r == orow and c == ocol:
                    canvas = _overlay_text_block(canvas, otext)

            processed_row.append(canvas)
        processed_rows.append(processed_row)

    # horizontally stack each row and then vertically stack rows
    hor_rows = [np.hstack(row) for row in processed_rows]
    grid = np.vstack(hor_rows)
    return grid



def match_set_symbol(image, templates):
    # Compare cropped symbol region to known templates
    # Return best match or confidence scores
    pass


def match_art_phash(image, known_hashes):
    # Compute pHash of cropped art region
    # Compare to known hashes
    hash = imagehash.phash(image.fromarray(image))
    return find_closest_match(hash, known_hashes)

def scan_card():
    image = capture_image()
    text = extract_text(image)
    symbol = match_set_symbol(image, templates)
    art_match = match_art_phash(image, known_hashes)
    
    return {
        "text": text,
        "set": symbol,
        "art": art_match
    }
"""
