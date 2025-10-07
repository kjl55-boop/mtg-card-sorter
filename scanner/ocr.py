# scanner/ocr.py
import pytesseract
import re

def run_tesseract(img, config: str = r'--oem 1 --psm 6') -> str:
    """
    Run tesseract on the provided image (single-channel or BGR numpy array).
    Returns raw OCR string.
    """
    return pytesseract.image_to_string(img, config=config)

def clean_ocr_text(raw: str) -> str:
    """Basic cleanup for OCR output."""
    # collapse multi-space, remove control chars, trim
    s = re.sub(r'\s+', ' ', raw).strip()
    return s