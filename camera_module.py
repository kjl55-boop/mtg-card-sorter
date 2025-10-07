# OpenCV’s cv2.VideoCapture() expects a standard V4L2 device like a USB webcam. 
# The Raspberry Pi Camera Module (especially on Pi OS Bullseye and later) uses 
# libcamera, which doesn’t expose the camera as /dev/video0 by default. 
# So OpenCV can’t “see” it unless you manually enable V4L2 support or use a workaround.

# camera_module_picamera_highq.py
# camera_module_picamera_preserve.py
from picamera2 import Picamera2
import time
import cv2
from PIL import Image
from PIL.ExifTags import TAGS

def read_exif(path):
    img = Image.open(path)
    exif = img._getexif() or {}
    return {TAGS.get(k,k): v for k,v in exif.items()}

class PiCameraCapture:
    def __init__(self, warmup_time=3.0):
        self.picam2 = Picamera2()
        self.warmup_time = warmup_time
        cfg = self.picam2.create_still_configuration(main={"size": (4608,2592), "format":"RGB888"})
        self.picam2.configure(cfg)

    def start(self):
        self.picam2.start()
        time.sleep(self.warmup_time)

    def capture_match(self, filename="capture.jpg", extra_wait=0.8, return_exif=False):
        # Ensure AE/AWB engaged and allowed to settle
        self.picam2.set_controls({"AeEnable": True})
        time.sleep(extra_wait)

        # Let Picamera2 write its JPEG with its encoder and EXIF
        self.picam2.capture_file(filename)

        # Read EXIF from the saved JPEG (should be present)
        try:
            exif = read_exif(filename)
        except Exception:
            exif = {}

        # Read JPEG with OpenCV for processing (BGR)
        img = cv2.imread(filename)
        return (img, exif) if return_exif else img

    def stop(self):
        self.picam2.stop()