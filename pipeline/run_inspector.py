#!/usr/bin/env python3
"""
Live inspector with toggleable controls window.

Keys:
  c  - capture preview (crop, snippets, try match via Matcher)
  s  - save current crop to debug dir
  m  - toggle controls window (open/close)
  q  - quit
  f  - trigger autofocus
  h  - show help menu
"""

import time
import cv2
from pathlib import Path
from collections import Counter

from config import loader

from . import utils

from . import capture
from recognizer import crop, ocr
from recognizer.matcher import Matcher

# ─────────────────────────────────────────────────────────────
# Logging and Defaults
# ─────────────────────────────────────────────────────────────

utils.configure_logging(level=getattr(loader, "LOG_LEVEL", None))
log = utils.get_logger("run_inspector")

DEFAULTS = {
    "pad_x_pct": 0,
    "pad_y_pct": 0,
    "min_area": getattr(loader, "DEFAULT_MIN_AREA", 5000),
    "top_pct": getattr(loader, "DEFAULT_TOP_PCT", 0.12),
    "mid_start_pct": getattr(loader, "DEFAULT_MID_START_PCT", 0.55),
    "mid_end_pct": getattr(loader, "DEFAULT_MID_END_PCT", 0.63),
    "bot_pct": getattr(loader, "DEFAULT_BOTTOM_PCT", 0.78),
    "display_scale_pct": getattr(loader, "DISPLAY_SCALE", 0.7)
}

CONTROLS_WIN = "Controls"

# ─────────────────────────────────────────────────────────────
# UI Controls
# ─────────────────────────────────────────────────────────────

def create_controls():
    cv2.namedWindow(CONTROLS_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROLS_WIN, 700, 240)
    for name, default, max_val in [
        ("Pad X %", int(DEFAULTS["pad_x_pct"] * 100), 50),
        ("Pad Y %", int(DEFAULTS["pad_y_pct"] * 100), 50),
        ("Min Area", DEFAULTS["min_area"], 50000),
        ("Top %", int(DEFAULTS["top_pct"] * 100), 40),
        ("Mid Start %", int(DEFAULTS["mid_start_pct"] * 100), 70),
        ("Mid End %", int(DEFAULTS["mid_end_pct"] * 100), 90),
        ("Bot %", int(DEFAULTS["bot_pct"] * 100), 95),
        ("Display %", int(DEFAULTS["display_scale_pct"] * 100), 200),
    ]:
        cv2.createTrackbar(name, CONTROLS_WIN, default, max_val, lambda x: None)

def destroy_controls():
    try:
        cv2.destroyWindow(CONTROLS_WIN)
    except Exception:
        pass

def controls_open():
    try:
        return cv2.getWindowProperty(CONTROLS_WIN, 0) >= 0
    except Exception:
        return False

def read_controls():
    def safe_pct(name, default):
        try:
            return cv2.getTrackbarPos(name, CONTROLS_WIN) / 100.0
        except Exception:
            return default

    def safe_val(name, default):
        try:
            return cv2.getTrackbarPos(name, CONTROLS_WIN)
        except Exception:
            return default

    return {
        "pad_x_pct": safe_pct("Pad X %", DEFAULTS["pad_x_pct"]),
        "pad_y_pct": safe_pct("Pad Y %", DEFAULTS["pad_y_pct"]),
        "min_area": max(100, safe_val("Min Area", DEFAULTS["min_area"])),
        "top_pct": safe_pct("Top %", DEFAULTS["top_pct"]),
        "mid_start_pct": safe_pct("Mid Start %", DEFAULTS["mid_start_pct"]),
        "mid_end_pct": safe_pct("Mid End %", DEFAULTS["mid_end_pct"]),
        "bot_pct": safe_pct("Bot %", DEFAULTS["bot_pct"]),
        "display_scale_pct": max(0.1, safe_pct("Display %", DEFAULTS["display_scale_pct"]))
    }

# ─────────────────────────────────────────────────────────────
# Matching Logic
# ─────────────────────────────────────────────────────────────

def overlay_text(img, text, org=(10, 30), color=(0, 255, 0)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def confirm_match_with_retries(card_image, matcher, attempts=3, dist_threshold=8):
    results = []
    for i in range(attempts):
        result = matcher.match_with_policy(card_image)
        if result:
            card_name = result.meta.get("name", "unknown")
            log.info(
                "Attempt %d: success=%s id=%s name=%s dist=%s",
                i + 1,
                result.success,
                result.id,
                card_name,
                result.dist,
            )
            if result.success and result.dist is not None and result.dist <= dist_threshold:
                results.append((result.id, result.dist, card_name))
        else:
            log.info("Attempt %d: result=None", i + 1)


    if not results:
        log.info("No valid phash matches across attempts")
        return None

    counts = Counter([r[0] for r in results])
    most_common_id, freq = counts.most_common(1)[0]
    log.info("Most common ID: %s (freq=%d)", most_common_id, freq)

    if freq >= 2:
        for r in results:
            if r[0] == most_common_id:
                return r  # (id, dist, name)

    log.info("No consensus match found")
    return None


def match_with_shudder_capture(camera, matcher, box, attempts=3, dist_threshold=None):
    results = []
    dist_threshold = dist_threshold or matcher.config["phash_threshold"]

    for i in range(attempts):
        frame = camera.read(timeout=1.0)
        if frame is None:
            continue
        card = crop.crop_card_from_box(frame, box, pad_x_pct=0.02, pad_y_pct=0.02)
        result = matcher.match_with_policy(card)
        if result and result.success and result.dist is not None and result.dist <= dist_threshold:
            card_name = result.meta.get("name", "unknown")
            log.info(
                "Shudder attempt %d: success=%s id=%s name=%s dist=%s",
                i + 1,
                result.success,
                result.id,
                card_name,
                result.dist,
            )
            results.append((result.id, result.dist, card_name))
        else:
            log.info("Shudder attempt %d: no valid match", i + 1)

    if not results:
        log.info("No valid matches across shudder attempts")
        return None

    counts = Counter([r[0] for r in results])
    most_common_id, freq = counts.most_common(1)[0]
    if freq >= 2:
        for r in results:
            if r[0] == most_common_id:
                return r  # (id, dist, name)

    log.info("No consensus match found across shudder attempts")
    return None


def fallback_ocr(card, top_pct, tesseract_config):
    title_crop = ocr.crop_title_band(card, init_top_pct=top_pct)
    proc = ocr.preprocess_for_ocr(title_crop)
    text, conf = ocr.ocr_image(proc, tesseract_config)
    overlay_text(card, f"OCR: {text[:30]} [{conf}]", org=(10, 40), color=(0, 200, 255))
    log.info("OCR title: %s conf=%s", text, conf)

# ─────────────────────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────────────────────

def run(debug_dir: str = None, tesseract_config: str = None):
    debug_dir = debug_dir or getattr(loader, "DEBUG_DIR", "data/debug")
    utils.ensure_dir(debug_dir)
    log.info("Starting run_inspector; debug_dir=%s", debug_dir)

    cam = capture.init_camera(preview_size=getattr(loader, "CAMERA_PREVIEW_SIZE", None))
    cam.autofocus()

    matcher = Matcher()
    tesseract_config = tesseract_config or getattr(loader, "TESSERACT_CONFIG_TITLE", None)

    controls_visible = False

    try:
        while True:
            frame = capture.grab_frame(cam, timeout=1.0)
            if frame is None:
                log.warning("No frame read from camera; retrying")
                time.sleep(0.1)
                continue

            ctrl = read_controls() if controls_visible and controls_open() else DEFAULTS
            box, contour = crop.find_card_contour(frame, min_area=ctrl["min_area"])
            vis = frame.copy()
            if box is not None:
                cv2.drawContours(vis, [box], -1, (0, 0, 255), 3)

            preview = cv2.resize(vis, (0, 0), fx=ctrl["display_scale_pct"], fy=ctrl["display_scale_pct"])
            cv2.imshow("Live", preview)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("m"):
                controls_visible = not controls_visible
                create_controls() if controls_visible else destroy_controls()

            elif key == ord("q"):
                log.info("Quit requested")
                break

            elif key == ord("f"):
                cam.autofocus()

            elif key == ord("h"):
                print("\nControls:")
                print("  c  - capture preview (crop, snippets, try match via Matcher)")
                print("  s  - save current crop to debug dir")
                print("  m  - toggle controls window (open/close)")
                print("  q  - quit")
                print("  f  - trigger autofocus")
                print("  h  - show this help menu\n")

            elif key == ord("c") and box is not None:
                log.info("Capture triggered")
                card = crop.crop_card_from_box(frame, box, pad_x_pct=ctrl["pad_x_pct"], pad_y_pct=ctrl["pad_y_pct"])
                if card is None or card.size == 0:
                    log.warning("Crop failed")
                    continue
                log.info("Card cropped successfully: shape=%s", card.shape)

                snippets = crop.extract_snippets(card, ctrl["top_pct"], ctrl["mid_start_pct"], ctrl["mid_end_pct"], ctrl["bot_pct"])
                cv2.imshow("Card", cv2.resize(card, (0, 0), fx=0.6, fy=0.6))
                for label, snip in snippets:
                    cv2.imshow(f"Snippet - {label}", cv2.resize(snip, (0, 0), fx=0.6, fy=0.6))

                match = match_with_shudder_capture(cam, matcher, box)
                if match:
                    match_id, dist, card_name = match
                    log.info("MATCH id=%s name=%s dist=%s", match_id, card_name, dist)
                    overlay_text(card, f"{card_name} [{dist}]", org=(10, 40))
                else:
                    log.info("No confident match; running OCR fallback")
                    fallback_ocr(card, ctrl["top_pct"], tesseract_config)


                cv2.imshow("Card", cv2.resize(card, (0, 0), fx=0.6, fy=0.6))


            elif key == ord("s") and box is not None:
                ts = int(time.time())
                card = crop.crop_card_from_box(frame, box, pad_x_pct=ctrl["pad_x_pct"], pad_y_pct=ctrl["pad_y_pct"])
                if card is not None and card.size:
                    p = Path(debug_dir) / f"{ts}_card.png"
                    ok = utils.safe_imwrite(str(p), card)
                    if ok:
                        log.info("Saved card to %s", p)
                    else:
                        log.warning("Failed to save card to %s", p)

    finally:
        try:
            capture.close_camera(cam)
        except Exception:
            log.debug("Error closing camera on exit", exc_info=True)
        try:
            destroy_controls()
        except Exception:
            pass
        cv2.destroyAllWindows()
        log.info("Inspector stopped")

if __name__ == "__main__":
    run()
