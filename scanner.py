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
    cv2.namedWindow("Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Feed", 1280, 720)
    cv2.imshow("Feed", frame)
    while True:
        # Wait for Key to save or close image capture
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cv2.imwrite("captured_image.jpg", frame)
            print("Image saved as 'captured_image.jpg'")
        elif key == ord('q'):
            break

    # Release the capture object and destroy all windows
    #cam.stop()
    return frame
    cv2.destroyAllWindows()

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
