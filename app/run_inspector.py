"""
Glue code to run the live inspector.
Wires capture -> crop -> matcher, displays overlays and simple key commands.
"""

import cv2
from app import capture, crop, matcher, config, utils, ocr
import time
from pathlib import Path

def overlay_text(img, text, org=(10,30), color=(0,255,0)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

def run(debug_dir="data/debug", tesseract_config=None):
    utils.ensure_dir(debug_dir)
    cam = capture.init_camera(config.CAMERA_PREVIEW_SIZE)
    tesseract_config = tesseract_config or config.TESSERACT_CONFIG_SNIPPET

    try:
        while True:
            frame = capture.grab_frame(cam)
            box, contour = crop.find_card_contour(frame)
            vis = frame.copy()
            if box is not None:
                cv2.drawContours(vis, [box], -1, (0,0,255), 3)

            preview = cv2.resize(vis, (0,0), fx=config.DISPLAY_SCALE, fy=config.DISPLAY_SCALE)
            cv2.imshow("Live", preview)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("c") and box is not None:
                pad_x = 0.08
                pad_y = 0.08
                card = crop.crop_card_from_box(frame, box, pad_x, pad_y)
                if card is None or card.size == 0:
                    print("Crop failed")
                    continue

                # show card and snippets
                top_pct = config.DEFAULT_TOP_PCT
                mid_start = config.DEFAULT_MID_START_PCT
                mid_end = config.DEFAULT_MID_END_PCT
                bot_pct = config.DEFAULT_BOTTOM_PCT

                snippets = crop.extract_snippets(card, top_pct, mid_start, mid_end, bot_pct)
                cv2.imshow("Card", cv2.resize(card, (0,0), fx=0.5, fy=0.5))
                for (label, snip) in snippets:
                    cv2.imshow(f"Snippet - {label}", cv2.resize(snip, (0,0), fx=0.6, fy=0.6))

                # try matcher
                res = matcher.match_card(card)
                if res:
                    rec, score, good, inl = res
                    print("MATCH:", rec["name"], rec["set_code"], rec["collector_number"], "score", score)
                    overlay_text(card, f"{rec['name']} [{score}]", org=(10,40))
                    cv2.imshow("Card", cv2.resize(card, (0,0), fx=0.5, fy=0.5))
                else:
                    print("No match; running OCR fallback")
                    title_crop = ocr.crop_title_band(card, init_top_pct=config.DEFAULT_TOP_PCT)
                    p = ocr.preprocess_for_ocr(title_crop)
                    t, conf = ocr.ocr_image(p, config=config.TESSERACT_CONFIG_TITLE)
                    print("OCR title:", t, "conf", conf)

            if key == ord("s") and box is not None:
                ts = int(time.time())
                card = crop.crop_card_from_box(frame, box, 0.08, 0.08)
                Path(debug_dir).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(Path(debug_dir) / f"{ts}_card.png"), card)
                print("Saved", ts)
    finally:
        capture.close_camera(cam)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
