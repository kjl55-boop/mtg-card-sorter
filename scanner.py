import cv2
from camera_module import PiCameraCapture
import pytesseract
import imagehash
from PIL import Image
import numpy as np
import subprocess
import time


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
            cv2.imwrite("captured_image.jpg", frame)
            print("Image saved as 'captured_image.jpg'")
        elif key == ord('q'):
            break
        elif key == ord('r'):
            cv2.destroyAllWindows()
            frame = capture_with_rpicam_still()

    # Release the capture object and destroy all windows
    #cam.stop()
    cv2.destroyAllWindows()
    return frame

def preprocess_for_ocr(path_or_img, roi=None, debug_save_prefix=None):
    """
    Returns a preprocessed image ready for pytesseract.
    - path_or_img: filename or BGR numpy array (cv2.imread output).
    - roi: optional tuple (x,y,w,h) to crop first on the full-res image.
    - debug_save_prefix: if provided, saves intermediate images with this prefix for inspection.
    """
    # --- load ---
    if isinstance(path_or_img, str):
        img = cv2.imread(path_or_img)  # BGR
    else:
        img = path_or_img.copy()
    orig = img.copy()

    # optional crop to ROI from full-res capture
    if roi is not None:
        x,y,w,h = roi
        img = img[y:y+h, x:x+w]

    # --- grayscale + denoise ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (3,3), 0)  # or cv2.medianBlur(gray, 3)

    # --- contrast enhance (CLAHE) ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)

    if debug_save_prefix:
        cv2.imwrite(f"{debug_save_prefix}_01_gray.jpg", gray)
        cv2.imwrite(f"{debug_save_prefix}_02_enhanced.jpg", enhanced)

    # --- detect candidate text area (morphology + Canny or adaptive threshold) ---
    th = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,3))
    morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    if debug_save_prefix:
        cv2.imwrite(f"{debug_save_prefix}_03_thresh.jpg", th)
        cv2.imwrite(f"{debug_save_prefix}_04_morph.jpg", morph)

    # --- find largest rectangular contour (likely the text box) ---
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        candidate = enhanced  # fallback: whole enhanced crop
    else:
        # sort by area and try to approximate rectangle
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        box = None
        for c in contours:
            area = cv2.contourArea(c)
            if area < 1000:  # ignore tiny noise
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                box = approx.reshape(4,2)
                break
        if box is None:
            # fallback: bounding rect of largest contour
            x,y,w,h = cv2.boundingRect(contours[0])
            candidate = enhanced[y:y+h, x:x+w]
        else:
            # order points and perspective transform to get a flat ROI
            def order_pts(pts):
                s = pts.sum(axis=1)
                diff = np.diff(pts, axis=1).flatten()
                tl = pts[np.argmin(s)]
                br = pts[np.argmax(s)]
                tr = pts[np.argmin(diff)]
                bl = pts[np.argmax(diff)]
                return np.array([tl, tr, br, bl], dtype="float32")
            pts = order_pts(box)
            (tl, tr, br, bl) = pts
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxW = int(max(widthA, widthB))
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxH = int(max(heightA, heightB))
            dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(pts, dst)
            warped = cv2.warpPerspective(enhanced, M, (maxW, maxH))
            candidate = warped

    # --- deskew (if needed) ---
    def deskew(img_gray):
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

    deskewed = deskew(candidate)

    # --- final cleanup for OCR: optional unsharp + thresholding ---
    # unsharp mask
    blur = cv2.GaussianBlur(deskewed, (0,0), sigmaX=1.0)
    sharp = cv2.addWeighted(deskewed, 1.5, blur, -0.5, 0)
    # final adaptive threshold or Otsu depending on contrast
    final = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 10)

    if debug_save_prefix:
        cv2.imwrite(f"{debug_save_prefix}_05_candidate.jpg", candidate)
        cv2.imwrite(f"{debug_save_prefix}_06_deskewed.jpg", deskewed)
        cv2.imwrite(f"{debug_save_prefix}_07_sharp.jpg", sharp)
        cv2.imwrite(f"{debug_save_prefix}_08_final.jpg", final)

    return final  # single-channel binary image ready for tesseract



def extract_text(image):
    # Preprocess image (grayscale, threshold, etc.)
    # Run Tesseract OCR
    text = pytesseract.image_to_string(image)
    return text

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
