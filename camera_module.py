# OpenCV’s cv2.VideoCapture() expects a standard V4L2 device like a USB webcam. 
# The Raspberry Pi Camera Module (especially on Pi OS Bullseye and later) uses 
# libcamera, which doesn’t expose the camera as /dev/video0 by default. 
# So OpenCV can’t “see” it unless you manually enable V4L2 support or use a workaround.

# camera_module_picamera_highq.py
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
        # explicit still pipeline at full sensor res and RGB output
        cfg = self.picam2.create_still_configuration(main={"size": (4608,2592), "format":"RGB888"})
        self.picam2.configure(cfg)

    def start(self):
        self.picam2.start()
        time.sleep(self.warmup_time)  # give AE/AWB/ISP time to settle

    def capture_match(self, filename="capture.jpg", extra_wait=0.6, jpeg_quality=95, return_exif=False):
        # Ensure AE/AWB are active and settled
        self.picam2.set_controls({"AeEnable": True})
        time.sleep(extra_wait)

        # 1) capture file (Picamera2 encoder)
        self.picam2.capture_file(filename)

        # 2) re-open and re-encode with high JPEG quality to avoid aggressive encoder defaults
        img = cv2.imread(filename)  # BGR
        if img is not None:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            pil.save(filename, "JPEG", quality=jpeg_quality, optimize=True)

        exif = read_exif(filename)
        img = cv2.imread(filename)
        return (img, exif) if return_exif else img

    def stop(self):
        self.picam2.stop()
