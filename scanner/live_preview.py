import cv2
import pytesseract
import time
from pathlib import Path
from picamera2 import Picamera2

selected_roi = None
drawing = False
start_point = (0, 0)

def mouse_callback(event, x, y, flags, param):
    global selected_roi, drawing, start_point
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        selected_roi = (start_point[0], start_point[1], x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        selected_roi = (start_point[0], start_point[1], x, y)

def extract_top_line(region, height_ratio=0.15):
    h = region.shape[0]
    top = region[:int(h * height_ratio), :]
    gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
    return top, gray

def run_live_preview(debug_dir="debug_live", tesseract_config="--oem 1 --psm 7"):
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
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = picam.capture_array()
        display = frame.copy()
        ocr_text = ""

        if selected_roi:
            x1, y1, x2, y2 = selected_roi
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            card = frame[y1:y2, x1:x2]
            top_line, gray = extract_top_line(card)
            ocr_text = pytesseract.image_to_string(gray, config=tesseract_config).strip()

            for i, line in enumerate(ocr_text.splitlines()[:3]):
                cv2.putText(display, line, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        scale_factor = 0.6  # or 0.5, 0.75 — tweak as needed
        resized = cv2.resize(display, (0, 0), fx=scale_factor, fy=scale_factor)
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
            cv2.imwrite(str(Path(debug_dir) / f"{ts}_card.png"), card)
            with open(Path(debug_dir) / f"{ts}_ocr.txt", "w") as f:
                f.write(ocr_text)

    cv2.destroyAllWindows()