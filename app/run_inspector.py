#!/usr/bin/env python3
"""
Live inspector with toggleable controls window.

Keys:
  c  - capture preview (crop, snippets, try match via Matcher)
  s  - save current crop to debug dir
  m  - toggle controls window (open/close)
  q  - quit
  f  - 
"""
from pathlib import Path
import time
import cv2

from . import capture, crop, ocr, config, utils
from .matcher import Matcher, MatchResult

# Configure logging once at startup
utils.configure_logging(level=config.LOG_LEVEL if hasattr(config, "LOG_LEVEL") else None)
log = utils.get_logger("run_inspector")

# Defaults (sourced from config where available)
DEFAULTS = {
    "pad_x_pct": 0,
    "pad_y_pct": 0,
    "min_area": getattr(config, "DEFAULT_MIN_AREA", 5000),
    "top_pct": getattr(config, "DEFAULT_TOP_PCT", 0.12),
    "mid_start_pct": getattr(config, "DEFAULT_MID_START_PCT", 0.55),
    "mid_end_pct": getattr(config, "DEFAULT_MID_END_PCT", 0.63),
    "bot_pct": getattr(config, "DEFAULT_BOTTOM_PCT", 0.78),
    "display_scale": getattr(config, "DISPLAY_SCALE", 0.7)
}

CONTROLS_WIN = "Controls"

def create_controls():
    cv2.namedWindow(CONTROLS_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROLS_WIN, 700, 240)
    cv2.createTrackbar("Pad X %", CONTROLS_WIN, int(DEFAULTS["pad_x_pct"]*100), 50, lambda x: None)
    cv2.createTrackbar("Pad Y %", CONTROLS_WIN, int(DEFAULTS["pad_y_pct"]*100), 50, lambda x: None)
    cv2.createTrackbar("Min Area", CONTROLS_WIN, DEFAULTS["min_area"], 50000, lambda x: None)
    cv2.createTrackbar("Top %", CONTROLS_WIN, int(DEFAULTS["top_pct"]*100), 40, lambda x: None)
    cv2.createTrackbar("Mid Start %", CONTROLS_WIN, int(DEFAULTS["mid_start_pct"]*100), 70, lambda x: None)
    cv2.createTrackbar("Mid End %", CONTROLS_WIN, int(DEFAULTS["mid_end_pct"]*100), 90, lambda x: None)
    cv2.createTrackbar("Bot %", CONTROLS_WIN, int(DEFAULTS["bot_pct"]*100), 95, lambda x: None)
    cv2.createTrackbar("Display %", CONTROLS_WIN, int(DEFAULTS["display_scale"]*100), 200, lambda x: None)

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
    pad_x = cv2.getTrackbarPos("Pad X %", CONTROLS_WIN) / 100.0
    pad_y = cv2.getTrackbarPos("Pad Y %", CONTROLS_WIN) / 100.0
    min_area = max(100, cv2.getTrackbarPos("Min Area", CONTROLS_WIN))
    top_pct = cv2.getTrackbarPos("Top %", CONTROLS_WIN) / 100.0
    mid_start = cv2.getTrackbarPos("Mid Start %", CONTROLS_WIN) / 100.0
    mid_end = cv2.getTrackbarPos("Mid End %", CONTROLS_WIN) / 100.0
    bot_pct = cv2.getTrackbarPos("Bot %", CONTROLS_WIN) / 100.0
    display_scale = max(10, cv2.getTrackbarPos("Display %", CONTROLS_WIN)) / 100.0
    return {
        "pad_x": pad_x,
        "pad_y": pad_y,
        "min_area": min_area,
        "top_pct": top_pct,
        "mid_start": mid_start,
        "mid_end": mid_end,
        "bot_pct": bot_pct,
        "display_scale": display_scale
    }

def overlay_text(img, text, org=(10,30), color=(0,255,0)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def run(debug_dir: str = None, tesseract_config: str = None):
    debug_dir = debug_dir or getattr(config, "DEBUG_DIR", "data/debug")
    utils.ensure_dir(debug_dir)
    log.info("Starting run_inspector; debug_dir=%s", debug_dir)

    cam = capture.init_camera(preview_size=getattr(config, "CAMERA_PREVIEW_SIZE", None))
    # Run autofocus once at startup
    cam.autofocus()

    matcher = Matcher()  # uses defaults and loads index if available
    tesseract_config = tesseract_config or getattr(config, "TESSERACT_CONFIG_TITLE", None)

    controls_visible = False

    try:
        while True:
            frame = capture.grab_frame(cam, timeout=1.0)
            if frame is None:
                log.warning("No frame read from camera; retrying")
                time.sleep(0.1)
                continue

            if controls_visible and controls_open():
                ctrl = read_controls()
                min_area = ctrl["min_area"]
                display_scale = ctrl["display_scale"]
            else:
                min_area = DEFAULTS["min_area"]
                display_scale = DEFAULTS["display_scale"]

            box, contour = crop.find_card_contour(frame, min_area=min_area)
            vis = frame.copy()
            if box is not None:
                cv2.drawContours(vis, [box], -1, (0,0,255), 3)

            preview = cv2.resize(vis, (0,0), fx=display_scale, fy=display_scale)
            cv2.imshow("Live", preview)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("m"):
                controls_visible = not controls_visible
                if controls_visible:
                    create_controls()
                else:
                    destroy_controls()

            if key == ord("q"):
                log.info("Quit requested")
                break

            if key == ord("c") and box is not None:
                # read control params at capture time
                if controls_visible and controls_open():
                    c = read_controls()
                    pad_x = c["pad_x"]; pad_y = c["pad_y"]
                    top_pct = c["top_pct"]; mid_start = c["mid_start"]
                    mid_end = c["mid_end"]; bot_pct = c["bot_pct"]
                else:
                    pad_x = DEFAULTS["pad_x"]; pad_y = DEFAULTS["pad_y"]
                    top_pct = DEFAULTS["top_pct"]; mid_start = DEFAULTS["mid_start_pct"]
                    mid_end = DEFAULTS["mid_end_pct"]; bot_pct = DEFAULTS["bot_pct"]

                card = crop.crop_card_from_box(frame, box, pad_x_pct=pad_x, pad_y_pct=pad_y)
                if card is None or card.size == 0:
                    log.warning("Crop failed")
                    continue

                # show card and snippets
                snippets = crop.extract_snippets(card, top_pct, mid_start, mid_end, bot_pct)
                cv2.imshow("Card", cv2.resize(card, (0,0), fx=0.6, fy=0.6))
                for (label, snip) in snippets:
                    cv2.imshow(f"Snippet - {label}", cv2.resize(snip, (0,0), fx=0.6, fy=0.6))

                # Run matcher policy (phash attempts, ORB verify, OCR fallback)
                try:
                    result = matcher.match_with_policy(card)
                except Exception as e:
                    utils.log_exception(log, e, "Matcher raised unexpected exception")
                    result = MatchResult(success=False)

                # Present result
                if result and result.success:
                    log.info("MATCH id=%s dist=%s attempts=%s elapsed=%.3fs", result.id, result.dist, result.attempts, result.elapsed)
                    # overlay a short label on the card display
                    label = result.meta.get("name", result.id or "MATCH")
                    overlay_text(card, f"{label} [{result.dist}]", org=(10,40))
                    cv2.imshow("Card", cv2.resize(card, (0,0), fx=0.6, fy=0.6))
                else:
                    log.info("No match; running OCR fallback for diagnostics")
                    title_crop = ocr.crop_title_band(card, init_top_pct=top_pct)
                    proc = ocr.preprocess_for_ocr(title_crop)
                    text, conf = ocr.ocr_image(proc, config=tesseract_config)
                    log.info("OCR title: %s conf=%s", text, conf)
                    overlay_text(card, f"OCR: {text[:30]} [{conf}]", org=(10,40), color=(0,200,255))
                    cv2.imshow("Card", cv2.resize(card, (0,0), fx=0.6, fy=0.6))

            if key == ord("s") and box is not None:
                ts = int(time.time())
                if controls_visible and controls_open():
                    pad_x = read_controls()["pad_x"]
                    pad_y = read_controls()["pad_y"]
                else:
                    pad_x = 0.08
                    pad_y = 0.08
                card = crop.crop_card_from_box(frame, box, pad_x_pct=pad_x, pad_y_pct=pad_y)
                if card is not None and card.size:
                    p = Path(debug_dir) / f"{ts}_card.png"
                    ok = utils.safe_imwrite(str(p), card)
                    if ok:
                        log.info("Saved card to %s", p)
                    else:
                        log.warning("Failed to save card to %s", p)
            if key == ord("f"):
                cam.autofocus()
            if key == ord("h"):
                print("\nControls:")
                print("  c  - capture preview (crop, snippets, try match via Matcher)")
                print("  s  - save current crop to debug dir")
                print("  m  - toggle controls window (open/close)")
                print("  q  - quit")
                print("  f  - trigger autofocus")
                print("  h  - show this help menu\n")

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
