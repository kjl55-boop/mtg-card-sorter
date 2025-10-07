# OpenCV’s cv2.VideoCapture() expects a standard V4L2 device like a USB webcam. 
# The Raspberry Pi Camera Module (especially on Pi OS Bullseye and later) uses 
# libcamera, which doesn’t expose the camera as /dev/video0 by default. 
# So OpenCV can’t “see” it unless you manually enable V4L2 support or use a workaround.

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
    def __init__(self, warmup_time=2):
        self.picam2 = Picamera2()
        self.warmup_time = warmup_time
        
        # Use default still configuration - matches rpicam-still defaults
        config = self.picam2.create_still_configuration()
        self.picam2.configure(config)
        
        # Don't set any controls - let the camera use its defaults
        # rpicam-still doesn't set these, so neither should we

    def start(self):
        self.picam2.start()
        time.sleep(self.warmup_time)  # Standard warmup for AE/AWB

    def capture_match(self, filename="capture.jpg", extra_wait=0.3, return_exif=False):
        self.picam2.set_controls({"AeEnable": True})
        time.sleep(extra_wait)
        self.picam2.capture_file(filename)
        exif = read_exif(filename)
        img = cv2.imread(filename)
        return (img, exif) if return_exif else img


    def stop(self):
        self.picam2.stop()
