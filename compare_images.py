# compare_images.py
import os, cv2, numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

def read_exif(path):
    img = Image.open(path)
    exif = img._getexif() or {}
    return {TAGS.get(k,k): v for k,v in exif.items()}

def stats(path):
    img = cv2.imread(path)
    if img is None:
        return {"size": os.path.getsize(path), "img": None}
    lap = cv2.Laplacian(img, cv2.CV_64F).var()
    return {
        "size_bytes": os.path.getsize(path),
        "shape": img.shape,
        "min": int(img.min()),
        "max": int(img.max()),
        "var": float(np.var(img)),
        "lap_var": float(lap)
    }

for p in ("rpicam.jpg","code.jpg"):
    print("\n---", p)
    print("EXIF:", read_exif(p))
    print("Stats:", stats(p))