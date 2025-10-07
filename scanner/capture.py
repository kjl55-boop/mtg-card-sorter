#  Camera wrappers (rpicam or Picamera2) and start/stop API
# currently rpicam until I can get PiCamera2 to look better

# scanner/capture.py
import subprocess
import time
import cv2
from typing import Tuple, Dict

def capture_with_rpicam_still(filename: str = "capture.jpg") -> 'np.ndarray':
    """
    Quick parity capture using the rpicam-still binary.
    Returns BGR image loaded with OpenCV.
    """
    subprocess.run(["rpicam-still", "-o", filename], check=True)
    time.sleep(0.05)
    img = cv2.imread(filename)
    return img

# Optional Picamera2 wrapper (if Picamera2 is required)
try:
    from picamera2 import Picamera2
except Exception:
    Picamera2 = None

class Picamera2Capture:
    def __init__(self, warmup_time: float = 3.0):
        if Picamera2 is None:
            raise RuntimeError("Picamera2 not available")
        self.picam2 = Picamera2()
        self.warmup_time = warmup_time
        cfg = self.picam2.create_still_configuration(main={"size": (4608,2592), "format":"RGB888"})
        self.picam2.configure(cfg)

    def start(self) -> None:
        self.picam2.start()
        time.sleep(self.warmup_time)

    def capture_match(self, filename: str = "capture.jpg", extra_wait: float = 0.8, return_exif: bool = False):
        self.picam2.set_controls({"AeEnable": True})
        time.sleep(extra_wait)
        self.picam2.capture_file(filename)
        if return_exif:
            from .utils import read_exif
            exif = read_exif(filename)
            img = cv2.imread(filename)
            return img, exif
        return cv2.imread(filename)

    def stop(self) -> None:
        self.picam2.stop()