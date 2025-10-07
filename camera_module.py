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
        self.picam2.configure(self.picam2.create_still_configuration(
            main={"size": (4608, 2592), "format": "RGB888"},
            raw={"size": (4608, 2592)}
        ))
        self.picam2.set_controls({
            "NoiseReductionMode": 2,
            "Sharpness": 1.0,
            "Contrast": 1.0,
            "Saturation": 1.0,
            "AwbMode": 1
        })

    def start(self):
        self.picam2.start()
        time.sleep(self.warmup_time + 1.0)  # give AE/AWB extra time

    def capture_match(self, filename="capture.jpg", exposure_us=None, analogue_gain=None, extra_wait=0.3):
        if exposure_us is not None and analogue_gain is not None:
            # switch to manual exposure/gain
            self.picam2.set_controls({
                "AeEnable": False,
                "ExposureTime": int(exposure_us),
                "AnalogueGain": float(analogue_gain)
            })
            time.sleep(0.1)
        else:
            # ensure AE is enabled for libcamera auto selection
            self.picam2.set_controls({"AeEnable": True})

        time.sleep(extra_wait)
        # capture_file ensures libcamera ISP + JPEG + EXIF
        self.picam2.capture_file(filename)
        exif = read_exif(filename)
        img = cv2.imread(filename)  # BGR
        return img#, exif

    def stop(self):
        self.picam2.stop()

'''
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
                raw={"size": (4608, 2592)}  # Enables full-res raw stream

            ))

        # 🔧 Set image quality controls here
        self.picam2.set_controls({
            "Sharpness": 1.0,
            "Contrast": 1.0,
            "Saturation": 1.0,
            "NoiseReductionMode": 2,  # High quality
            #"AwbMode": 1              # Auto white balance
        })

    def start(self):
        self.picam2.start()
        time.sleep(self.warmup_time)

    def capture_frame(self, scale=None):
        #Modular scaling of frame
        self.picam2.capture_file("temp.jpg")
        frame = cv2.imread("temp.jpg")
        if scale:
            w = int(frame.shape[1] * scale)
            h = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        return frame


    def stop(self):
        self.picam2.stop()
'''