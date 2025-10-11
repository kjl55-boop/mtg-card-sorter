# high_quality_capture.py
from picamera2 import Picamera2, Preview
from libcamera import controls
import time, cv2, numpy as np, sys, os
OUT = "/tmp/picam_high_quality.png"

def high_quality_capture(target_size=(1920,1080), warmup=0.5, settle=0.5):
    pc = Picamera2()
    # Use still configuration for highest quality processing pipeline
    still_cfg = pc.create_still_configuration(main={"size": target_size, "format": "RGB888"})
    pc.configure(still_cfg)

    # Helpful control choices; tune these for your lighting and sensor
    default_controls = {
        # let AE/AWB run initially; we'll optionally lock them below
        "AfMode": controls.AfModeEnum.Continuous,
        "AwbEnable": True,
        # Noise reduction - use HighQuality if available
        "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.HighQuality,
        # Sharpening/contrast are sensor/IPA dependent
        "Sharpness": 2.0,
        "Contrast": 1.0,
        "Saturation": 1.0,
    }
    try:
        pc.set_controls(default_controls)
    except Exception:
        pass

    pc.start()
    try:
        # warm up auto-exposure/auto-whitebalance
        time.sleep(warmup)
        # let AE/AWB settle a bit more for still exposure
        time.sleep(settle)

        # Optionally read current AE/AWB and lock them for repeatable captures:
        # info = pc.capture_metadata()
        # print("metadata sample:", info)
        # To lock AE/AWB (uncomment if you want fixed settings):
        # current_controls = {"AeEnable": False, "AwbEnable": False}
        # pc.set_controls(current_controls)

        arr = pc.capture_array()
        # Picamera2 still config yields RGB array; convert to BGR for OpenCV
        if arr is None:
            print("capture_array returned None", file=sys.stderr)
            return 2
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        # Optional: mild denoise + sharpening using OpenCV (tune or remove if unwanted)
        denoised = cv2.fastNlMeansDenoisingColored(bgr, None, 6, 6, 7, 21)
        # sharpen kernel
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
        sharp = cv2.filter2D(denoised, -1, kernel)

        cv2.imwrite(OUT, sharp)
        print("Wrote:", OUT)
    finally:
        try:
            pc.stop()
        except Exception:
            pass

if __name__ == "__main__":
    sys.exit(high_quality_capture())
