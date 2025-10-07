import cv2
from camera_module import PiCameraCapture
import pytesseract
import imagehash
from PIL import Image
import numpy as np

def capture_image():
    # Save or return captured image for processing
    cam = PiCameraCapture()
    cam.start()

    while True:
        frame = cam.capture_frame()
        frame = self.picam2.capture_array()
        resized = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
        cv2.imshow("Pi Camera Feed", resized)

        # Wait for Key to save or close image capture
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cv2.imwrite("captured_image.jpg", resized)
            print("Image saved as 'captured_image.jpg'")
        elif key == ord('q'):
            break

    # Release the capture object and destroy all windows
    cam.stop()
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
