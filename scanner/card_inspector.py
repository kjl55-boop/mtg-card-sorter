import cv2
import pytesseract
import time
from pathlib import Path
from picamera2 import Picamera2
import numpy as np

drawing = False
selected_roi = None
start_point = (0, 0)

def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, selected_roi
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        selected_roi = (start_point[0], start_point[1], x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        selected_roi = (start_point[0], start_point[1], x, y)

def rotate_ccw_90(image):
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

def extract_snippets(card_img):
    h, w = card_img.shape[:2]
    snippets = []

    # Example: top line
    top_line = card_img[0:int(0.12*h), :]
    snippets.append(("Top Line", top_line))

    # Example: bottom right corner
    corner = card_img[int(0.85*h):, int(0.7*w):]
    snippets.append(("Bottom Right", corner))

    # Add more regions as needed
    return snippets

def run_card_inspector(debug_dir="debug_card", tesseract_config="--oem 1 --psm 7"):
    global selected_roi
    Path(debug_dir).mkdir(parents=True, exist_ok=True)

    picam = Picamera2()
    from libcamera import controls
    config = picam.create_preview_configuration(main={"size": (2304, 1296)})
    picam.configure(config)
    picam.start()

    picam.set_controls({
        "AfMode": controls.AfModeEnum.Continuous,
        "AwbEnable": True,
        "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.HighQuality,
        "Sharpness": 2.0,
        "Contrast": 1.5,
        "Saturation": 1.5
    })

    cv2.namedWindow("Live Feed")
    cv2.setMouseCallback("Live Feed", mouse_callback)

    while True:
        frame = picam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        display = frame.copy()

        if selected_roi:
            x1, y1, x2, y2 = selected_roi
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            card = frame[y1:y2, x1:x2]
            rotated = rotate_ccw_90(card)
            snippets = extract_snippets(rotated)

            for i, (label, snippet) in enumerate(snippets):
                gray = cv2.cvtColor(snippet, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray, config=tesseract_config).strip()
                cv2.putText(display, f"{label}: {text[:30]}", (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        scale = 0.6
        resized = cv2.resize(display, (0, 0), fx=scale, fy=scale)
        cv2.imshow("Live Feed", resized)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            selected_roi = None
        elif key == ord('s') and selected_roi:
            ts = int(time.time())
            x1, y1, x2, y2 = selected_roi
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            card = frame[y1:y2, x1:x2]
            rotated = rotate_ccw_90(card)
            cv2.imwrite(str(Path(debug_dir) / f"{ts}_card_rotated.png"), rotated)
            with open(Path(debug_dir) / f"{ts}_ocr.txt", "w") as f:
                for label, snippet in extract_snippets(rotated):
                    gray = cv2.cvtColor(snippet, cv2.COLOR_BGR2GRAY)
                    text = pytesseract.image_to_string(gray, config=tesseract_config).strip()
                    f.write(f"{label}:\n{text}\n\n")

    cv2.destroyAllWindows()