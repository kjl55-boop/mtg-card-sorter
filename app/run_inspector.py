#!/usr/bin/env python3
"""
Live inspector with toggleable controls window.

Keys:
  c  - capture preview (crop, snippets, try match)
  s  - save current crop to debug dir
  m  - toggle controls window (open/close)
  q  - quit
"""
import cv2
import time
from pathlib import Path
from app import capture, crop, matcher, config, utils, ocr

# --- Controls defaults (sync with config but adjustable at runtime) ----------
DEFAULTS = {
    "pad_x_pct": 0,        # repurposed default; will be overwritten
    "pad_y_pct": 0,
    "min_area": 5000,
    "top_pct": int(config.DEFAULT_TOP_PCT * 100),
    "mid_start_pct": int(config.DEFAULT_MID_START_PCT * 100),
    "mid_end_pct": int(config.DEFAULT_MID_END_PCT * 100),
    "bot_pct": int(config.DEFAULT_BOTTOM_PCT * 100),
    "display_scale": int(config.DISPLAY_SCALE * 100)
}

CONTROLS_WIN = "Controls"

def create_controls():
    cv2.namedWindow(CONTROLS_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROLS_WIN, 700, 220)
    cv2.createTrackbar("Pad X %", CONTROLS_WIN, DEFAULTS["pad_x_pct"], 50, lambda x: None)
    cv2.createTrackbar("Pad Y %", CONTROLS_WIN, DEFAULTS["pad_y_pct"], 50, lambda x: None)
    cv2.createTrackbar("Min Area", CONTROLS_WIN, DEFAULTS["min_area"], 20000, lambda x: None)
    cv2.createTrackbar("Top %", CONTROLS_WIN, DEFAULTS["top_pct"], 40, lambda x: None)
    cv2.createTrackbar("Mid Start %", CONTROLS_WIN, DEFAULTS["mid_start_pct"], 70, lambda x: None)
    cv2.createTrackbar("Mid End %", CONTROLS_WIN, DEFAULTS["mid_end_pct"], 90, lambda x: None)
    cv2.createTrackbar("Bot %", CONTROLS_WIN, DEFAULTS["bot_pct"], 95, lambda x: None)
    cv2.createTrackbar("Display %", CONTROLS_WIN, DEFAULTS["display_scale"], 100, lambda x: None)

def destroy_controls():
    try:
        cv2.destroyWindow(CONTROLS_WIN)
    except Exception:
        pass

def controls_open():
    return CONTROLS_WIN in cv2.getWindowProperty.__self__.__dict__ if False else (cv2.getWindowProperty(CONTROLS_WIN, 0) >= 0 if cv2.getWindowProperty(CONTROLS_WIN, 0) is not None else False)

def read_controls():
    """Read trackbar values and return a dict with normalized floats where appropriate."""
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

def run(debug_dir="data/debug", tesseract_config=None):
    utils.ensure_dir(debug_dir)
    cam = capture.init_camera(config.CAMERA_PREVIEW_SIZE)
    tesseract_config = tesseract_config or config.TESSERACT_CONFIG_SNIPPET

    controls_visible = False

    try:
        while True:
            frame = capture.grab_frame(cam)
            # dynamic min_area: if controls open, use that; otherwise use default
            if controls_visible:
                ctrl = read_controls()
                min_area = ctrl["min_area"]
                display_scale = ctrl["display_scale"]
            else:
                min_area = 5000
                display_scale = config.DISPLAY_SCALE

            box, contour = crop.find_card_contour(frame, min_area=min_area)
            vis = frame.copy()
            if box is not None:
                cv2.drawContours(vis, [box], -1, (0,0,255), 3)

            preview = cv2.resize(vis, (0,0), fx=display_scale, fy=display_scale)
            cv2.imshow("Live", preview)
            key = cv2.waitKey(1) & 0xFF

            # toggle controls window
            if key == ord("m"):
                controls_visible = not controls_visible
                if controls_visible:
                    create_controls()
                else:
                    destroy_controls()

            if key == ord("q"):
                break

            if key == ord("c") and box is not None:
                # read current control params for cropping
                if controls_visible:
                    c = read_controls()
                    pad_x = c["pad_x"]
                    pad_y = c["pad_y"]
                    top_pct = c["top_pct"]
                    mid_start = c["mid_start"]
                    mid_end = c["mid_end"]
                    bot_pct = c["bot_pct"]
                else:
                    pad_x = 0.0
                    pad_y = 0.0
                    top_pct = config.DEFAULT_TOP_PCT
                    mid_start = config.DEFAULT_MID_START_PCT
                    mid_end = config.DEFAULT_MID_END_PCT
                    bot_pct = config.DEFAULT_BOTTOM_PCT

                card = crop.crop_card_from_box(frame, box, pad_x_pct=pad_x, pad_y_pct=pad_y)
                if card is None or card.size == 0:
                    print("Crop failed")
                    continue

                # show card and snippets
                snippets = crop.extract_snippets(card, top_pct, mid_start, mid_end, bot_pct)
                cv2.imshow("Card", cv2.resize(card, (0,0), fx=0.5, fy=0.5))
                for (label, snip) in snippets:
                    cv2.imshow(f"Snippet - {label}", cv2.resize(snip, (0,0), fx=0.6, fy=0.6))

                # try matcher (DB may not exist; function may return None)
                res = None
                try:
                    res = matcher.match_card(card)
                except FileNotFoundError as e:
                    # DB not found; will run OCR fallback
                    print("Matcher DB missing:", e)

                if res:
                    rec, score, good, inl = res
                    print("MATCH:", rec["name"], rec["set_code"], rec["collector_number"], "score", score)
                    overlay_text(card, f"{rec['name']} [{score}]", org=(10,40))
                    cv2.imshow("Card", cv2.resize(card, (0,0), fx=0.5, fy=0.5))
                else:
                    print("No match; running OCR fallback")
                    title_crop = ocr.crop_title_band(card, init_top_pct=top_pct)
                    p = ocr.preprocess_for_ocr(title_crop)
                    t, conf = ocr.ocr_image(p, config=config.TESSERACT_CONFIG_TITLE)
                    print("OCR title:", t, "conf", conf)

            if key == ord("s") and box is not None:
                ts = int(time.time())
                if controls_visible:
                    pad_x = read_controls()["pad_x"]
                    pad_y = read_controls()["pad_y"]
                else:
                    pad_x = 0.08
                    pad_y = 0.08
                card = crop.crop_card_from_box(frame, box, pad_x_pct=pad_x, pad_y_pct=pad_y)
                Path(debug_dir).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(Path(debug_dir) / f"{ts}_card.png"), card)
                print("Saved", ts)

    finally:
        capture.close_camera(cam)
        # Ensure controls window destroyed cleanly
        try:
            destroy_controls()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
