import cv2
import pytesseract
import imagehash
from PIL import Image
import numpy as np

def capture_image():
    # Save or return captured image for processing
    # Create a VideoCapture object
    cap = cv2.VideoCapture(1)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Display the resulting frame
        cv2.imshow('Camera Feed', frame)

        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF

        # Save image on 's' key press
        if key == ord('s'):
            cv2.imwrite('captured_image.jpg', frame)
            print("Image saved as 'captured_image.jpg'")

        # Exit on 'q' key press
        if key == ord('q'):
            break

    # Release the capture object and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
    pass

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
