# scanner/utils.py
from PIL import Image
from PIL.ExifTags import TAGS
import numpy as np

def read_exif(path: str) -> dict:
    img = Image.open(path)
    exif = img._getexif() or {}
    return {TAGS.get(k,k): v for k,v in exif.items()}

import cv2
def laplacian_var(gray_img) -> float:
    return float(cv2.Laplacian(gray_img, cv2.CV_64F).var())