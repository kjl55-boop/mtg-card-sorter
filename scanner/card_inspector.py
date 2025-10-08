import cv2
import pytesseract
import time
from pathlib import Path
from picamera2 import Picamera2
import numpy as np
from libcamera import controls

def auto_detect_card(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = [c for c in contours if cv2.contourArea(c) > 5000]
    if not contours:
        return None

    card_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(card_contour)

    pad = 20
    x = max(x - pad, 0)
    y = max(y - pad, 0)
    w = min(w + 2 * pad, frame.shape[1] - x)
    h = min(h + 2 * pad, frame.shape[0] - y)

    return frame[y:y+h, x:x+w]

def rotate_ccw_90(image):
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

def extract_snippets(card_img, top_pct, mid_start_pct, mid_end_pct, bot_pct):
    h, w = card_img.shape[:2]
    snippets = []

    top = card_img[0:int(top_pct * h), :]
    middle = card_img[int(mid_start_pct * h):int(mid_end_pct * h), :]
    bottom = card_img[int(bot_pct * h):, :]

    snippets.append(("Top", top))
    snippets.append(("Middle", middle))
    snippets.append(("Bottom", bottom))

    return snippets

def run_card_inspector(debug_dir="debug_card", tesseract_config="--oem 1 --psm 7"):
    Path(debug_dir).mkdir(parents=True, exist_ok=True)

    picam = Picamera2()
    config = picam.create_preview_configuration(main={"size": (2304, 1296)})
    picam.configure(config)
    picam.set_controls({
        "AfMode": controls.AfModeEnum.Continuous,
        "AwbEnable": True,
        "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.HighQuality,
        "Sharpness": 2.0,
        "Contrast": 1.5,
        "Saturation": 1.5
    })
    picam.start()

    cv2.namedWindow("Live Feed")
    cv2.createTrackbar("Top %", "Live Feed", 20, 100, lambda x: None)
    cv2.createTrackbar("Mid Start %", "Live Feed", 35, 100, lambda x: None)
    cv2.createTrackbar("Mid End %", "Live Feed", 65, 100, lambda x: None)
    cv2.createTrackbar("Bottom %", "Live Feed", 80, 100, lambda x: None)

    while True:
        frame = picam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        card = auto_detect_card(frame)

        if card is not None:
            rotated = rotate_ccw_90(card)
            cv2.imshow("Card", rotated)

            # Get slider values
            top_pct = cv2.getTrackbarPos("Top %", "Live Feed") / 100.0
            mid_start_pct = cv2.getTrackbarPos("Mid Start %", "Live Feed") / 100.0
            mid_end_pct = cv2.getTrackbarPos("Mid End %", "Live Feed") / 100.0
            bot_pct = cv2.getTrackbarPos("Bottom %", "Live Feed") / 100.0

            snippets = extract_snippets(rotated, top_pct, mid_start_pct, mid_end_pct, bot_pct)

            for label, snippet in snippets:
                cv2.imshow(label, snippet)
                gray = cv2.cvtColor(snippet, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray, config=tesseract_config).strip()
                cv2.putText(frame, f"{label}: {text[:30]}", (10, 30 + 25 * ["Top", "Middle", "Bottom"].index(label)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        scaled = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
        cv2.imshow("Live Feed", scaled)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and card is not None:
            ts = int(time.time())
            cv2.imwrite(str(Path(debug_dir) / f"{ts}_card.png"), rotated)
            with open(Path(debug_dir) / f"{ts}_ocr.txt", "w") as f:
                for label, snippet in snippets:
                    gray = cv2.cvtColor(snippet, cv2.COLOR_BGR2GRAY)
                    text = pytesseract.image_to_string(gray, config=tesseract_config).strip()
                    f.write(f"{label}:\n{text}\n\n")

    cv2.destroyAllWindows()