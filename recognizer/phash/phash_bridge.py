from pathlib import Path
import cv2
import numpy as np
from config.config import CONFIG
from pipeline import utils
from recognizer.crop import crop_card_from_box, find_card_contour
from recognizer.phash.phash_tools import PhashComparator

log = utils.get_logger("phash")
OUT_W, OUT_H = CONFIG.normalized_size

def normalize_and_save(frame_bgr: np.ndarray,
                       filename: str,
                       pad_x_pct: float = 0.02,
                       pad_y_pct: float = 0.02,
                       min_area: int = 2000) -> Path:
    """
    Crop and normalize a card from the frame, then save to debug directory.
    Returns saved Path or raises RuntimeError on failure.
    """
    if frame_bgr is None:
        raise ValueError("normalize_and_save: input frame is None")

    box, _ = find_card_contour(frame_bgr, min_area=min_area)
    if box is None:
        raise RuntimeError("normalize_and_save: no card contour found")

    card = crop_card_from_box(frame_bgr, box, pad_x_pct=pad_x_pct, pad_y_pct=pad_y_pct)
    if card is None or card.size == 0:
        raise RuntimeError("normalize_and_save: crop failed")

    norm = cv2.resize(card, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    p = Path(CONFIG.debug_dir) / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    written = utils.safe_imwrite(str(p), norm)
    if not written:
        raise RuntimeError(f"Failed to write normalized image to {p}")
    log.info("Saved normalized image to %s", p)
    return p

def compute_phash_bgr(frame_bgr: np.ndarray) -> str:
    """
    Compute perceptual hash from a BGR image using PhashComparator.
    """
    return PhashComparator().compute(frame_bgr)
