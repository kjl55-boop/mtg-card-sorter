import cv2
from camera_module import PiCameraCapture
import os
import time
from math import ceil
from pathlib import Path
import pytesseract
import imagehash
from PIL import Image
import numpy as np
import subprocess



def capture_with_rpicam_still(filename="capture.jpg"):
    subprocess.run(["rpicam-still", "-o", filename], check=True)
    time.sleep(0.05)
    img = cv2.imread(filename)  # BGR, ready for processing
    return img

def capture_image():
    # Save or return captured image for processing
    #cam = PiCameraCapture()
    #cam.start()

    frame = capture_with_rpicam_still() #cam.capture_match()
    while True:
        cv2.namedWindow("Feed", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Feed", 1280, 720)
        cv2.imshow("Feed", frame)
        # Wait for Key to save or close image capture
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            #cv2.imwrite("captured_image.jpg", frame)
            print("Image saved as 'capture.jpg'")
            break
        elif key == ord('q'):
            break
        elif key == ord('r'):
            cv2.destroyAllWindows()
            frame = capture_with_rpicam_still()

    # Release the capture object and destroy all windows
    #cam.stop()
    cv2.destroyAllWindows()
    return frame


# ---------- helper pipeline functions (reuse or import from your module) ----------
def order_pts(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")

def deskew_image(img_gray):
    if img_gray is None or img_gray.size == 0:
        return img_gray
    coords = np.column_stack(np.where(img_gray < 255))
    if coords.shape[0] < 10:
        return img_gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h,w) = img_gray.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(img_gray, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def compute_pipeline_steps(img_bgr, roi=None):
    """Return ordered dict name->image (grayscale where applicable)."""
    steps = {}
    if roi is not None:
        x,y,w,h = roi
        img = img_bgr[y:y+h, x:x+w].copy()
    else:
        img = img_bgr.copy()

    # 01 gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    steps["01_gray"] = gray

    # 02 denoised
    denoised = cv2.GaussianBlur(gray, (3,3), 0)
    steps["02_denoised"] = denoised

    # 03 clahe
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    steps["03_clahe"] = enhanced

    # 04 adaptive threshold (binary inverted for morphology)
    th = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 10)
    steps["04_adaptivethresh"] = th

    # 05 morph close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,3))
    morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    steps["05_morph"] = morph

    # 06 candidate detection and crop/warp
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis = img.copy()
    candidate_crop = enhanced.copy()
    box_points = None
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 1000:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                box_points = approx.reshape(4,2)
                break
        if box_points is None:
            x,y,w,h = cv2.boundingRect(contours[0])
            candidate_crop = enhanced[y:y+h, x:x+w]
            cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
        else:
            pts = order_pts(box_points)
            cv2.polylines(vis, [box_points.astype(int)], True, (0,255,0), 3)
            (tl,tr,br,bl) = pts
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxW = int(max(widthA, widthB))
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxH = int(max(heightA, heightB))
            dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(pts, dst)
            warped = cv2.warpPerspective(enhanced, M, (maxW, maxH))
            candidate_crop = warped

    steps["06_candidate_vis"] = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
    steps["07_candidate_crop"] = candidate_crop

    # 08 deskew
    deskewed = deskew_image(candidate_crop)
    steps["08_deskewed"] = deskewed

    # 09 sharpen
    blur = cv2.GaussianBlur(deskewed, (0,0), sigmaX=1.0)
    sharp = cv2.addWeighted(deskewed, 1.5, blur, -0.5, 0)
    steps["09_sharp"] = sharp

    # 10 final threshold
    final = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 10)
    steps["10_final"] = final

    return steps

# ---------- display helpers ----------
def make_labelled_tile(img, label, tile_size=(480,360), font_scale=0.6):
    w_t, h_t = tile_size
    if img is None:
        tile = np.zeros((h_t, w_t, 3), dtype=np.uint8)
    elif img.ndim == 2:
        tile = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        tile = img.copy()
    ih, iw = tile.shape[:2]
    scale = min(w_t/iw, h_t/ih)
    new_w, new_h = max(1, int(iw*scale)), max(1, int(ih*scale))
    tile = cv2.resize(tile, (new_w, new_h), interpolation=cv2.INTER_AREA)
    top = (h_t - new_h) // 2
    left = (w_t - new_w) // 2
    canvas = np.zeros((h_t, w_t, 3), dtype=np.uint8)
    canvas[top:top+new_h, left:left+new_w] = tile
    cv2.rectangle(canvas, (0,0), (w_t, 36), (20,20,20), -1)
    # label plus small secondary text area
    cv2.putText(canvas, label, (8,24), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (200,200,200), 1, cv2.LINE_AA)
    return canvas

def overlay_text_block(tile, text, max_lines=6):
    """Overlay up to max_lines of OCR text inside a dark band under the tile image."""
    h,w = tile.shape[:2]
    band_h = int(h * 0.28)
    # shrink tile vertically to make room for text band
    img_top = tile[:h-band_h, :, :].copy()
    band = np.zeros((band_h, w, 3), dtype=np.uint8) + 18
    # render text lines
    lines = text.strip().splitlines()[:max_lines]
    y = 20
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        cv2.putText(band, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA)
        y += 18
    # combine
    out = np.vstack([img_top, band])
    return out

# ---------- main function ----------
def show_pipeline_grid_with_ocr(path_or_img, roi=None, debug_dir="debug_grid", tile_size=(480,360), cols=3, live=False, tesseract_config=r'--oem 1 --psm 6'):
    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    if isinstance(path_or_img, str):
        base_img = cv2.imread(path_or_img)
    else:
        base_img = path_or_img.copy()
    if base_img is None:
        raise ValueError("Could not load image")

    window_name = "Pipeline Grid with OCR"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    paused = not live
    while True:
        if live or not paused:
            steps = compute_pipeline_steps(base_img, roi=roi)

            # run Tesseract on the final processed binary image (ensure correct format)
            final_img = steps.get("10_final")
            ocr_text = ""
            if final_img is not None:
                # prefer passing an 8-bit single-channel image
                ocr_text = pytesseract.image_to_string(final_img, config=tesseract_config)

            # create tiles and if the final tile, overlay OCR text under it
            names = list(steps.keys())
            tiles = []
            for n in names:
                label = n
                tile = make_labelled_tile(steps[n], label, tile_size=tile_size)
                if n == "10_final":
                    tile = overlay_text_block(tile, ocr_text)
                tiles.append(tile)

        # assemble grid
        rows = ceil(len(tiles) / cols)
        w_t, h_t = tile_size
        # account for final tile possibly taller due to text band
        band_h = int(h_t * 0.28)
        canvas = np.zeros((rows * h_t, cols * w_t, 3), dtype=np.uint8)
        for i, t in enumerate(tiles):
            r = i // cols
            c = i % cols
            y0 = r * h_t; x0 = c * w_t
            # if a tile is taller (overlay_text_block made it larger), center it vertically in cell
            th, tw = t.shape[:2]
            if th <= h_t:
                canvas[y0:y0+th, x0:x0+tw] = t
            else:
                # crop/pad top if needed
                canvas[y0:y0+h_t, x0:x0+w_t] = cv2.resize(t, (w_t, h_t), interpolation=cv2.INTER_AREA)

        # display scaled canvas
        screen_h = 900
        scale = min(1.0, screen_h / canvas.shape[0])
        disp = cv2.resize(canvas, (int(canvas.shape[1]*scale), int(canvas.shape[0]*scale)), interpolation=cv2.INTER_AREA)
        cv2.imshow(window_name, disp)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('s'):
            ts = int(time.time())
            for name, img in steps.items():
                fname = os.path.join(debug_dir, f"{ts}_{name}.png")
                cv2.imwrite(fname, img)
            # save OCR text too
            with open(os.path.join(debug_dir, f"{ts}_ocr.txt"), "w") as f:
                f.write(ocr_text)
            print("Saved pipeline images and OCR text to", debug_dir)
        if key == ord(' '):
            live = not live
            paused = not live

    cv2.destroyAllWindows()



def match_set_symbol(image, templates):
    # Compare cropped symbol region to known templates
    # Return best match or confidence scores
    pass

"""
def match_art_phash(image, known_hashes):
    # Compute pHash of cropped art region
    # Compare to known hashes
    hash = imagehash.phash(image.fromarray(image))
    return find_closest_match(hash, known_hashes)

def scan_card():
    image = capture_image()
    text = extract_text(image)
    symbol = match_set_symbol(image, templates)
    art_match = match_art_phash(image, known_hashes)
    
    return {
        "text": text,
        "set": symbol,
        "art": art_match
    }
"""
