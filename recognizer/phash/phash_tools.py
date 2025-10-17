import imagehash
from PIL import Image
import cv2
import numpy as np

class PhashComparator:
    def __init__(self, hash_size: int = 16):
        self.hash_size = hash_size

    def compute(self, bgr_img: np.ndarray) -> str:
        """Compute pHash from OpenCV BGR image."""
        pil_img = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
        return str(imagehash.phash(pil_img, hash_size=self.hash_size))

    def compare(self, bgr_img1: np.ndarray, bgr_img2: np.ndarray) -> int:
        """Return Hamming distance between two BGR images."""
        h1 = self.compute(bgr_img1)
        h2 = self.compute(bgr_img2)
        return imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2)

    def compare_hashes(self, h1: str, h2: str) -> int:
        """Compare two hex pHash strings directly."""
        return imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2)
