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

    # Filter contours by area
    contours = [c for c in contours if cv2.contourArea(c) > 10000]
    if not contours:
        return None

    card_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(card_contour)
    return frame[y:y+h, x:x+w]

def rotate_ccw_90(image):
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

def extract_snippets(card_img, regions=None):
    h, w = card_img.shape[:2]
    snippets = []

    # Default regions if none provided
    if regions is None:
        regions = {
            "Top": (0.00, 0.25),
            "Middle": (0.35, 0.65),
            "Bottom": (0.75, 1.00)
        }

    for label, (start, end) in regions.items():
        y1 = int(start * h)
        y2 = int(end * h)
        snippet = card_img[y1:y2, :]
        snippets.append((label, snippet))

    return snippets

def run_card_inspector(debug_dir="debug_card", tesseract_config="--oem 1 --psm 7"):
    Path(debug_dir).mkdir(parents=True, exist_ok=True)

    picam = Picamera2()
    config = picam.create_preview_configuration(main={"size": (1280, 720)})
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

    while True:
        frame = picam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        card = auto_detect_card(frame)

        if card is not None:
            rotated = rotate_ccw_90(card)
            cv2.imshow("Card", rotated)

            snippets = extract_snippets(rotated)
            for label, snippet in snippets:
                cv2.imshow(label, snippet)
                gray = cv2.cvtColor(snippet, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray, config=tesseract_config).strip()
                print(f"{label} OCR:\n{text}\n")

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