import cv2
import pytesseact
import imagehash
from PIL import image
import numpy as np

def capture_image():
    # Use PiCamera or OpenCV to capture image
    # Save or return image for processing
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

def match_art_phash(image, known_hashes):
    # Compute pHash of cropped art region
    # Compare to known hashes
    hash = imagehash.phash(Image.fromarray(image))
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
