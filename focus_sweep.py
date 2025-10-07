# focus_tuner.py
import time
import cv2
import numpy as np
from picamera2 import Picamera2

WINDOW_PREVIEW = "Preview"
WINDOW_CROP = "Center Crop"
CROP_SIZE = 600  # pixels for 1:1 crop scoring

def lap_var(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def center_crop(img, size):
    h,w = img.shape[:2]
    cx,cy = w//2, h//2
    half = size//2
    return img[cy-half:cy+half, cx-half:cx+half].copy()

def find_focus_control(pic):
    candidates = ["LensPosition","Focus","lens_position","focus"]
    controls = pic.camera_controls or {}
    for c in candidates:
        if c in controls:
            return c
    return None

def nothing(x): pass

def main():
    pic = Picamera2()
    cfg = pic.create_still_configuration(main={"size": (3280,2464), "format":"RGB888"})
    pic.configure(cfg)
    pic.start()
    time.sleep(2.0)

    ctrl = find_focus_control(pic)
    if ctrl is None:
        print("No electronic focus control exposed; use mechanical rotation workflow.")
        pic.stop()
        return

    cv2.namedWindow(WINDOW_PREVIEW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_PREVIEW, 1280, 720)
    cv2.namedWindow(WINDOW_CROP, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_CROP, 600, 600)
    cv2.createTrackbar("Focus", WINDOW_PREVIEW, 0, 1000, nothing)

    last_pos = None
    try:
        while True:
            # read trackbar and map 0..1000 to 0.0..1.0
            pos = cv2.getTrackbarPos("Focus", WINDOW_PREVIEW)
            value = pos / 1000.0

            if last_pos is None or pos != last_pos:
                try:
                    pic.set_controls({ctrl: float(value)})
                except Exception as e:
                    print("Failed to set focus control:", e)
                last_pos = pos

            frame = pic.capture_array()
            if frame is None:
                continue
            frame_bgr = frame[:, :, ::-1]

            # show preview (downscaled)
            h,w = frame_bgr.shape[:2]
            preview = cv2.resize(frame_bgr, (1280, int(1280 * h / w)), interpolation=cv2.INTER_AREA) if w>1280 else frame_bgr
            cv2.imshow(WINDOW_PREVIEW, preview)

            # show 1:1 center crop and sharpness
            crop = center_crop(frame_bgr, CROP_SIZE)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            score = lap_var(gray)
            cv2.putText(crop, f"LapVar: {score:.1f}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imshow(WINDOW_CROP, crop)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                fname = f"focus_{pos}.jpg"
                pic.capture_file(fname)
                print("Saved", fname)

    finally:
        pic.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()