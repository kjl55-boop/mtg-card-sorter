# OpenCV’s cv2.VideoCapture() expects a standard V4L2 device like a USB webcam. 
# The Raspberry Pi Camera Module (especially on Pi OS Bullseye and later) uses 
# libcamera, which doesn’t expose the camera as /dev/video0 by default. 
# So OpenCV can’t “see” it unless you manually enable V4L2 support or use a workaround.

from picamera2 import Picamera2
import cv2
import time

class PiCameraCapture:
    def __init__(self, resolution=(4608, 2592), warmup_time=2):
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_still_configuration(
            main={"size": resolution, "format": "RGB888"}
        ))
        self.warmup_time = warmup_time

    def start(self):
        self.picam2.start()
        time.sleep(self.warmup_time)

    def capture_frame(self):
        frame = self.picam2.capture_array()
        return frame

    def stop(self):
        self.picam2.stop()
