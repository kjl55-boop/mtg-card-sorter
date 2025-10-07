# OpenCV’s cv2.VideoCapture() expects a standard V4L2 device like a USB webcam. 
# The Raspberry Pi Camera Module (especially on Pi OS Bullseye and later) uses 
# libcamera, which doesn’t expose the camera as /dev/video0 by default. 
# So OpenCV can’t “see” it unless you manually enable V4L2 support or use a workaround.

# live_camera_tuner.py
import cv2
import time
from picamera2 import Picamera2
import numpy as np

WINDOW_NAME = "Camera Tuner"

def create_trackbars():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    # Exposure in milliseconds (1 .. 200 ms)
    cv2.createTrackbar("Exposure ms", WINDOW_NAME, 50, 200, lambda x: None)
    # Analogue gain as (100 .. 800) -> gain = pos / 100
    cv2.createTrackbar("Analogue gain x100", WINDOW_NAME, 400, 800, lambda x: None)
    # Sharpness (-200 .. 200) -> map to -4.0 .. 4.0
    cv2.createTrackbar("Sharpness x100", WINDOW_NAME, 100, 300, lambda x: None)
    # Contrast (-200 .. 200)
    cv2.createTrackbar("Contrast x100", WINDOW_NAME, 100, 300, lambda x: None)
    # Saturation (-200 .. 200)
    cv2.createTrackbar("Saturation x100", WINDOW_NAME, 100, 300, lambda x: None)
    # Noise reduction mode (0,1,2)
    cv2.createTrackbar("NoiseReduction", WINDOW_NAME, 2, 2, lambda x: None)
    # AWB mode toggle (0=off,1=auto)
    cv2.createTrackbar("AwbMode", WINDOW_NAME, 1, 1, lambda x: None)

def read_trackbar_values():
    exp_ms = cv2.getTrackbarPos("Exposure ms", WINDOW_NAME)
    ag_x100 = cv2.getTrackbarPos("Analogue gain x100", WINDOW_NAME)
    sharp_x100 = cv2.getTrackbarPos("Sharpness x100", WINDOW_NAME)
    contrast_x100 = cv2.getTrackbarPos("Contrast x100", WINDOW_NAME)
    sat_x100 = cv2.getTrackbarPos("Saturation x100", WINDOW_NAME)
    nr_mode = cv2.getTrackbarPos("NoiseReduction", WINDOW_NAME)
    awb_mode = cv2.getTrackbarPos("AwbMode", WINDOW_NAME)

    exposure_us = max(1, exp_ms) * 1000
    analogue_gain = max(1, ag_x100) / 100.0
    sharpness = (sharp_x100 - 100) / 50.0
    contrast = (contrast_x100 - 100) / 50.0
    saturation = (sat_x100 - 100) / 50.0

    return {
        "ExposureTime": int(exposure_us),
        "AnalogueGain": float(analogue_gain),
        "Sharpness": float(sharpness),
        "Contrast": float(contrast),
        "Saturation": float(saturation),
        "NoiseReductionMode": int(nr_mode),
        "AwbMode": int(awb_mode),
    }

def main():
    picam2 = Picamera2()
    # preview configuration for interactive tuning (lower size for speed)
    picam2.configure(picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"}
    ))

    create_trackbars()
    picam2.start()
    time.sleep(1.0)  # let AE/AGC settle a bit

    manual_ae = False
    last_controls = {}

    try:
        while True:
            controls = read_trackbar_values()

            # If manual AE is enabled, we turn AeEnable off and apply ExposureTime & AnalogueGain
            if manual_ae:
                set_controls = {
                    "AeEnable": False,
                    "ExposureTime": controls["ExposureTime"],
                    "AnalogueGain": controls["AnalogueGain"],
                    "Sharpness": controls["Sharpness"],
                    "Contrast": controls["Contrast"],
                    "Saturation": controls["Saturation"],
                    "NoiseReductionMode": controls["NoiseReductionMode"],
                    "AwbMode": controls["AwbMode"],
                }
            else:
                set_controls = {
                    "AeEnable": True,
                    # still apply ISP tuning controls while AE is on
                    "Sharpness": controls["Sharpness"],
                    "Contrast": controls["Contrast"],
                    "Saturation": controls["Saturation"],
                    "NoiseReductionMode": controls["NoiseReductionMode"],
                    "AwbMode": controls["AwbMode"],
                }

            # Only call set_controls when values changed to reduce overhead
            if set_controls != last_controls:
                picam2.set_controls(set_controls)
                last_controls = set_controls.copy()

            # Capture a frame for live preview (fast path)
            frame = picam2.capture_array()
            # convert RGB to BGR for OpenCV display
            frame_bgr = frame[:, :, ::-1]

            # Downscale for a reasonable window size if needed
            h, w = frame_bgr.shape[:2]
            target_w, target_h = 1280, 720
            if w > target_w:
                scale = target_w / w
                frame_disp = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                frame_disp = frame_bgr

            # Overlay current mode and brief instructions
            mode_text = "MANUAL AE" if manual_ae else "AUTO AE"
            cv2.putText(frame_disp, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            cv2.putText(frame_disp, "q=quit  s=save libcamera JPEG  m=toggle AE", (10, frame_disp.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            cv2.imshow(WINDOW_NAME, frame_disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                manual_ae = not manual_ae
                # small delay to let setting take effect
                time.sleep(0.05)
            elif key == ord('s'):
                # Capture a processed JPEG with current controls applied by libcamera
                fname = f"capture_{int(time.time())}.jpg"
                # if manual AE, make sure AeEnable already False and exposure/gain set above
                picam2.capture_file(fname)
                print("Saved", fname)

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

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