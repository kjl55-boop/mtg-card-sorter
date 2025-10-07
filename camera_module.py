# OpenCV’s cv2.VideoCapture() expects a standard V4L2 device like a USB webcam. 
# The Raspberry Pi Camera Module (especially on Pi OS Bullseye and later) uses 
# libcamera, which doesn’t expose the camera as /dev/video0 by default. 
# So OpenCV can’t “see” it unless you manually enable V4L2 support or use a workaround.

from picamera2 import Picamera2
import cv2
import time

class PiCameraCapture:
    def __init__(self, mode="scan", warmup_time=2):
        self.picam2 = Picamera2()
        self.warmup_time = warmup_time

        if mode == "preview":
            self.picam2.configure(self.picam2.create_preview_configuration(
                main={"size": (1280, 720), "format": "RGB888"}
            ))
        elif mode == "scan":
            self.picam2.configure(self.picam2.create_still_configuration(
                main={"size": (4608, 2592), "format": "RGB888"},
                display="off"
            ))

        # 🔧 Set image quality controls here
        self.picam2.set_controls({
            "Sharpness": 1.0,
            "Contrast": 1.0,
            "Saturation": 1.0,
            "NoiseReductionMode": 2,  # High quality
            "AwbMode": 1              # Auto white balance
        })

    def start(self):
        self.picam2.start()
        time.sleep(self.warmup_time)

    def capture_frame(self):
        return self.picam2.capture_array()

    def stop(self):
        self.picam2.stop()

'''
    def capture_frame(self):
        frame = self.picam2.capture_array()
        resized = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
        return resized
'''
