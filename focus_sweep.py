import time
import os
from picamera2 import Picamera2
import cv2
import numpy as np

# --- CONFIG ---
OUT_DIR = "focus_sweep_out"
BEST_NAME = "best_focus.jpg"
ROI_SIZE = 600             # pixels: square ROI used to score sharpness (center crop)
STEPS = 11                 # number of focus positions to test
START_POS = 0.0            # start of control range (0.0 .. 1.0 typical)
END_POS = 1.0              # end of control range (0.0 .. 1.0 typical)
WARMUP_SECS = 2.5          # initial warmup for AE/AWB
PER_POS_WAIT = 0.12        # wait after setting control before capture
# ------------------

def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)

def laplacian_var(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

def center_crop(img, size):
    h, w = img.shape[:2]
    cx, cy = w//2, h//2
    half = size//2
    x0 = max(0, cx-half); y0 = max(0, cy-half)
    x1 = min(w, cx+half); y1 = min(h, cy+half)
    return img[y0:y1, x0:x1]

def find_focus_control(pic):
    # common control names tried in order
    candidates = ["LensPosition", "Focus", "lens_position", "focus"]
    controls = pic.camera_controls or {}
    for name in candidates:
        if name in controls:
            return name
    # if camera_controls is not descriptive, try a pragmatic approach:
    for name in candidates:
        try:
            pic.set_controls({name: 0.0})
            pic.set_controls({"AeEnable": True})  # restore
            return name
        except Exception:
            pass
    return None

def run_sweep():
    ensure_out_dir()
    pic = Picamera2()
    cfg = pic.create_still_configuration()
    pic.configure(cfg)
    pic.start()
    time.sleep(WARMUP_SECS)

    ctrl_name = find_focus_control(pic)
    if ctrl_name is None:
        print("No focus control exposed by Picamera2 on this device.")
        print("You can still run manual mechanical adjustments or try different lens hardware.")
        pic.stop()
        return

    print("Using focus control:", ctrl_name)
    positions = np.linspace(START_POS, END_POS, STEPS)
    best_score = -1.0
    best_pos = None
    best_file = None

    for i, pos in enumerate(positions):
        try:
            # clamp and set control
            value = float(pos)
            pic.set_controls({ctrl_name: value})
        except Exception as e:
            print(f"Failed to set {ctrl_name} to {pos}: {e}")
            continue

        time.sleep(PER_POS_WAIT)
        fname = os.path.join(OUT_DIR, f"pos_{i:02d}.jpg")
        pic.capture_file(fname)

        img = cv2.imread(fname)
        if img is None:
            print("Failed reading captured image", fname)
            continue

        crop = center_crop(img, ROI_SIZE)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        score = laplacian_var(gray)

        print(f"Pos {i+1}/{len(positions)} value={pos:.4f} score={score:.1f} file={fname}")

        if score > best_score:
            best_score = score
            best_pos = pos
            best_file = fname

    pic.stop()

    if best_file:
        out_best = os.path.join(OUT_DIR, BEST_NAME)
        # copy best file to a known name
        img_best = cv2.imread(best_file)
        cv2.imwrite(out_best, img_best)
        print("Best focus saved:", out_best, "position:", best_pos, "score:", best_score)
    else:
        print("No valid captures produced.")

if __name__ == "__main__":
    run_sweep()
