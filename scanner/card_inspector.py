import cv2
import pytesseract
import time
from pathlib import Path
from picamera2 import Picamera2
import numpy as np
from libcamera import controls

def get_rotated_card_bounds(frame, scale_x=1.0, scale_y=1.0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = [c for c in contours if cv2.contourArea(c) > 5000]
    if not contours:
        return None, None

    card_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(card_contour)
    box = cv2.boxPoints(rect)
    box = box.astype(np.intp)

    center = np.mean(box, axis=0)
    scaled_box = np.array([
        [
            center[0] + scale_x * (pt[0] - center[0]),
            center[1] + scale_y * (pt[1] - center[1])
        ]
        for pt in box
    ], dtype=np.int32)

    return scaled_box, card_contour

def crop_rotated_box(frame, box, pad_x_pct=0.1, pad_y_pct=0.1):
    rect = cv2.minAreaRect(box.astype(np.float32))
    center, size, angle = rect

    if angle < -45:
        angle += 90

    if size[0] > size[1]:
        size = (size[1], size[0])
        angle += 90

    # Expand canvas
    h, w = frame.shape[:2]
    canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    canvas[h//2:h//2 + h, w//2:w//2 + w] = frame
    new_center = (w, h)  # center of expanded canvas

    # Rotate on expanded canvas
    M = cv2.getRotationMatrix2D(new_center, angle, 1.0)
    rotated = cv2.warpAffine(canvas, M, (w * 2, h * 2), flags=cv2.INTER_CUBIC)

    # Recalculate bounding box
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 5000]
    if not contours:
        return rotated

    card_contour = max(contours, key=cv2.contourArea)
    x, y, w_box, h_box = cv2.boundingRect(card_contour)

    # Apply padding
    pad_x = int(w_box * pad_x_pct / 2)
    pad_y = int(h_box * pad_y_pct / 2)
    x = max(x - pad_x, 0)
    y = max(y - pad_y, 0)
    w_box = min(w_box + 2 * pad_x, rotated.shape[1] - x)
    h_box = min(h_box + 2 * pad_y, rotated.shape[0] - y)

    cropped = rotated[y:y + h_box, x:x + w_box]
    return cropped





def is_title_upright(snippet, expected_title="Forest"):
    gray = cv2.cvtColor(snippet, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config="--oem 1 --psm 7").strip().lower()
    return expected_title.lower() in text

def extract_snippets(card_img, top_pct, mid_start_pct, mid_end_pct, bot_pct):
    h, w = card_img.shape[:2]
    snippets = [
        ("Top", card_img[0:int(top_pct * h), :]),
        ("Middle", card_img[int(mid_start_pct * h):int(mid_end_pct * h), :]),
        ("Bottom", card_img[int(bot_pct * h):, :])
    ]
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

    cv2.namedWindow("Controls")
    cv2.createTrackbar("Pad X %", "Controls", 10, 50, lambda x: None)
    cv2.createTrackbar("Pad Y %", "Controls", 10, 50, lambda x: None)

    cv2.createTrackbar("Top %", "Controls", 20, 100, lambda x: None)
    cv2.createTrackbar("Mid Start %", "Controls", 35, 100, lambda x: None)
    cv2.createTrackbar("Mid End %", "Controls", 65, 100, lambda x: None)
    cv2.createTrackbar("Bottom %", "Controls", 80, 100, lambda x: None)

    while True:
        frame = picam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        scale_x = 1.0
        scale_y = 1.0
        pad_pct = cv2.getTrackbarPos("Box Pad %", "Controls") / 100.0
        box, contour = get_rotated_card_bounds(frame, scale_x, scale_y)

        if box is not None:
            cv2.drawContours(frame, [box], -1, (0, 0, 255), 4)

        scaled = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Controls", scaled)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and box is not None:
            pad_x = cv2.getTrackbarPos("Pad X %", "Controls") / 100.0
            pad_y = cv2.getTrackbarPos("Pad Y %", "Controls") / 100.0
            card = crop_rotated_box(frame, box, pad_x, pad_y)


            top_pct = cv2.getTrackbarPos("Top %", "Controls") / 100.0
            mid_start_pct = cv2.getTrackbarPos("Mid Start %", "Controls") / 100.0
            mid_end_pct = cv2.getTrackbarPos("Mid End %", "Controls") / 100.0
            bot_pct = cv2.getTrackbarPos("Bottom %", "Controls") / 100.0

            snippets = extract_snippets(card, top_pct, mid_start_pct, mid_end_pct, bot_pct)
            if not is_title_upright(snippets[0][1]):
                card = cv2.rotate(card, cv2.ROTATE_180)
                snippets = extract_snippets(card, top_pct, mid_start_pct, mid_end_pct, bot_pct)

            resized_card = cv2.resize(card, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("Card", resized_card)

            for label, snippet in snippets:
                resized_snippet = cv2.resize(snippet, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow(f"Confirmed {label}", resized_snippet)

                gray = cv2.cvtColor(snippet, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray, config=tesseract_config).strip()
                print(f"[Confirmed] {label} OCR:\n{text}\n")

        elif key == ord('s') and box is not None:
            ts = int(time.time())
            card = crop_rotated_box(frame, box, pad_pct)
            cv2.imwrite(str(Path(debug_dir) / f"{ts}_card.png"), card)

            with open(Path(debug_dir) / f"{ts}_ocr.txt", "w") as f:
                top_pct = cv2.getTrackbarPos("Top %", "Controls") / 100.0
                mid_start_pct = cv2.getTrackbarPos("Mid Start %", "Controls") / 100.0
                mid_end_pct = cv2.getTrackbarPos("Mid End %", "Controls") / 100.0
                bot_pct = cv2.getTrackbarPos("Bottom %", "Controls") / 100.0

                snippets = extract_snippets(card, top_pct, mid_start_pct, mid_end_pct, bot_pct)
                for label, snippet in snippets:
                    gray = cv2.cvtColor(snippet, cv2.COLOR_BGR2GRAY)
                    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, 11, 2)
                    text = pytesseract.image_to_string(gray, config="--oem 1 --psm 6")
                    f.write(f"{label}:\n{text}\n\n")

    cv2.destroyAllWindows()
