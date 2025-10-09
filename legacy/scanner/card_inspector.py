import cv2
import pytesseract
import time
from pathlib import Path
from picamera2 import Picamera2
import numpy as np
from libcamera import controls

# --- Utilities --------------------------------------------------------------

def get_rotated_card_bounds(frame, scale_x=1.0, scale_y=1.0, min_area=5000):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    if not contours:
        return None, None

    card_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(card_contour)
    box = cv2.boxPoints(rect).astype(np.int32)

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
    # Compute rect and normalized angle/portrait orientation
    rect = cv2.minAreaRect(box.astype(np.float32))
    _, size, angle = rect
    if angle < -45:
        angle += 90
    # force portrait dimensions for card crop (w <= h)
    if size[0] > size[1]:
        size = (size[1], size[0])
        angle += 90

    h, w = frame.shape[:2]

    # Expand canvas to avoid clipping when rotating
    canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    canvas[h // 2:h // 2 + h, w // 2:w // 2 + w] = frame
    new_center = (w, h)

    # Rotate full expanded canvas
    M = cv2.getRotationMatrix2D(new_center, angle, 1.0)
    rotated = cv2.warpAffine(canvas, M, (w * 2, h * 2), flags=cv2.INTER_CUBIC)

    # Re-detect largest contour in rotated image (ensures centering)
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 5000]
    if not contours:
        return rotated  # fallback: nothing found

    card_contour = max(contours, key=cv2.contourArea)
    x, y, w_box, h_box = cv2.boundingRect(card_contour)

    # Apply symmetric padding around bounding rect
    pad_x = int(w_box * pad_x_pct / 2)
    pad_y = int(h_box * pad_y_pct / 2)
    x = max(x - pad_x, 0)
    y = max(y - pad_y, 0)
    w_box = min(w_box + 2 * pad_x, rotated.shape[1] - x)
    h_box = min(h_box + 2 * pad_y, rotated.shape[0] - y)

    cropped = rotated[y:y + h_box, x:x + w_box]
    return cropped

def extract_snippets(card_img, top_pct, mid_start_pct, mid_end_pct, bot_pct):
    h, w = card_img.shape[:2]
    top_h = int(top_pct * h)
    mid_s = int(mid_start_pct * h)
    mid_e = int(mid_end_pct * h)
    bot_y = int(bot_pct * h)

    # clamp values
    top_h = max(1, min(top_h, h - 1))
    mid_s = max(0, min(mid_s, h - 1))
    mid_e = max(mid_s + 1, min(mid_e, h))
    bot_y = max(0, min(bot_y, h - 1))

    top = card_img[0:top_h, :]
    middle = card_img[mid_s:mid_e, :]
    bottom = card_img[bot_y:, :]

    return [("Top", top), ("Middle", middle), ("Bottom", bottom)]

def preprocess_for_ocr(img, scale=2.0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # Try adaptive threshold + slight morphological cleanup
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 15, 6)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    return gray

# --- Main pipeline ---------------------------------------------------------

def run_card_inspector(debug_dir="debug_card", tesseract_config="--oem 1 --psm 6"):
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

    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 800, 200)
    cv2.createTrackbar("Pad X %", "Controls", 8, 50, lambda x: None)
    cv2.createTrackbar("Pad Y %", "Controls", 8, 50, lambda x: None)
    cv2.createTrackbar("Top %", "Controls", 12, 40, lambda x: None)
    cv2.createTrackbar("Mid Start %", "Controls", 48, 70, lambda x: None)
    cv2.createTrackbar("Mid End %", "Controls", 58, 85, lambda x: None)
    cv2.createTrackbar("Bottom %", "Controls", 78, 95, lambda x: None)

    # optional: set tesseract executable path if on Windows
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    while True:
        frame = picam.capture_array()
        # Picamera2 returns RGB arrays; convert to BGR for OpenCV display/processing
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        pad_x = cv2.getTrackbarPos("Pad X %", "Controls") / 100.0
        pad_y = cv2.getTrackbarPos("Pad Y %", "Controls") / 100.0

        box, contour = get_rotated_card_bounds(frame, scale_x=1.0, scale_y=1.0)

        # draw detection box on live feed
        vis = frame.copy()
        if box is not None:
            cv2.drawContours(vis, [box], -1, (0, 0, 255), 3)

        # show live preview
        preview = cv2.resize(vis, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Live", preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # capture and show card + snippets
        if key == ord('c') and box is not None:
            top_pct = cv2.getTrackbarPos("Top %", "Controls") / 100.0
            mid_start_pct = cv2.getTrackbarPos("Mid Start %", "Controls") / 100.0
            mid_end_pct = cv2.getTrackbarPos("Mid End %", "Controls") / 100.0
            bot_pct = cv2.getTrackbarPos("Bottom %", "Controls") / 100.0

            card = crop_rotated_box(frame, box, pad_x, pad_y)
            if card is None or card.size == 0:
                print("Crop failed; card empty")
                continue

            # optionally ensure card is upright by checking title via OCR heuristics
            snippets = extract_snippets(card, top_pct, mid_start_pct, mid_end_pct, bot_pct)
            top_proc = preprocess_for_ocr(snippets[0][1])
            top_text = pytesseract.image_to_string(top_proc, config="--oem 1 --psm 7").strip().lower()
            # if title detected at bottom, rotate 180
            if len(top_text) == 0:
                bottom_proc = preprocess_for_ocr(snippets[2][1])
                bottom_text = pytesseract.image_to_string(bottom_proc, config="--oem 1 --psm 7").strip().lower()
                # heuristic: if bottom has letters and top doesn't, flip
                if len(bottom_text) > len(top_text):
                    card = cv2.rotate(card, cv2.ROTATE_180)
                    snippets = extract_snippets(card, top_pct, mid_start_pct, mid_end_pct, bot_pct)

            # show card and snippets
            cv2.imshow("Card", cv2.resize(card, (0,0), fx=0.5, fy=0.5))
            for label, snippet in snippets:
                win = f"Snippet - {label}"
                cv2.imshow(win, cv2.resize(snippet, (0,0), fx=0.6, fy=0.6))

            # OCR each snippet with preprocessing
            for label, snippet in snippets:
                processed = preprocess_for_ocr(snippet, scale=2.0)
                text = pytesseract.image_to_string(processed, config=tesseract_config).strip()
                print(f"[{label}] OCR:\n{text}\n")

        # save card + OCR
        if key == ord('s') and box is not None:
            ts = int(time.time())
            card = crop_rotated_box(frame, box, pad_x, pad_y)
            if card is None or card.size == 0:
                print("Save failed; empty crop")
                continue
            cv2.imwrite(str(Path(debug_dir) / f"{ts}_card.png"), card)

            top_pct = cv2.getTrackbarPos("Top %", "Controls") / 100.0
            mid_start_pct = cv2.getTrackbarPos("Mid Start %", "Controls") / 100.0
            mid_end_pct = cv2.getTrackbarPos("Mid End %", "Controls") / 100.0
            bot_pct = cv2.getTrackbarPos("Bottom %", "Controls") / 100.0

            snippets = extract_snippets(card, top_pct, mid_start_pct, mid_end_pct, bot_pct)
            with open(Path(debug_dir) / f"{ts}_ocr.txt", "w", encoding="utf-8") as f:
                for label, snippet in snippets:
                    processed = preprocess_for_ocr(snippet, scale=2.0)
                    text = pytesseract.image_to_string(processed, config=tesseract_config).strip()
                    # save snippet image too
                    cv2.imwrite(str(Path(debug_dir) / f"{ts}_{label}.png"), snippet)
                    f.write(f"{label}:\n{text}\n\n")
            print(f"Saved {ts}_card.png and OCR to {debug_dir}")

    cv2.destroyAllWindows()
    picam.stop()

# --- Entry point -----------------------------------------------------------

if __name__ == "__main__":
    run_card_inspector()
