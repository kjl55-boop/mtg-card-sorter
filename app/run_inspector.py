"""
GUI inspector with high-resolution capture (1080p) and scaled window display.

- Captures at HIGH_RES = (1920, 1080) when possible (Picamera2 preview configured to this size).
- Shows a scaled preview window (preview_scale default 0.6), adjustable at runtime:
    - 'S' (uppercase) increases display scale (zoom in)
    - 'D' decreases display scale (zoom out)
- Draggable corner handles control the quadrilateral used to compute the normalized warp.
- Keys: Space/S = save normalized crop; M = match; R = reset handles; Q/ESC = quit
"""
from pathlib import Path
from datetime import datetime
import time
import cv2
import numpy as np

# Try to import app utilities; fall back gracefully
try:
    from .capture import capture_frame, normalize_and_save, compute_phash_bgr
    from .matcher import load_db_phashes, top_phash_candidates, verify_candidate_orb
    from .config import CAMERA_WARMUP_SEC, NORMALIZED_SIZE, PHASH_STRICT_THRESHOLD
except Exception:
    capture_frame = None
    normalize_and_save = None
    compute_phash_bgr = None
    load_db_phashes = None
    top_phash_candidates = None
    verify_candidate_orb = None
    CAMERA_WARMUP_SEC = 0.25
    NORMALIZED_SIZE = (400, 560)
    PHASH_STRICT_THRESHOLD = 6

# Prefer Picamera2 for high-res capture when available
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except Exception:
    PICAMERA2_AVAILABLE = False

# Desired high resolution for capture (attempt 1080p)
HIGH_RES = (1920, 1080)

DEBUG_DIR = Path("data/debug")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

WIN = "Inspector"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

# Trackbars for detection params
tk_defaults = {"canny_low": 50, "canny_high": 150, "eps_pct": 2, "min_area": 2000, "show_norm": 1}
cv2.createTrackbar("canny_low", WIN, tk_defaults["canny_low"], 300, lambda v: None)
cv2.createTrackbar("canny_high", WIN, tk_defaults["canny_high"], 400, lambda v: None)
cv2.createTrackbar("eps_pct", WIN, tk_defaults["eps_pct"], 50, lambda v: None)
cv2.createTrackbar("min_area", WIN, tk_defaults["min_area"], 30000, lambda v: None)
cv2.createTrackbar("show_norm", WIN, tk_defaults["show_norm"], 1, lambda v: None)

def read_trackbar_vals():
    cl = cv2.getTrackbarPos("canny_low", WIN)
    ch = cv2.getTrackbarPos("canny_high", WIN)
    eps_pct = max(1, cv2.getTrackbarPos("eps_pct", WIN))
    min_area = max(100, cv2.getTrackbarPos("min_area", WIN))
    show_norm = bool(cv2.getTrackbarPos("show_norm", WIN))
    return cl, ch, eps_pct / 100.0, min_area, show_norm

# Draggable quad implementation (same as before)
HANDLE_RADIUS = 10
HANDLE_COLOR = (0, 255, 0)
ACTIVE_HANDLE_COLOR = (0, 128, 255)
LINE_COLOR = (0, 200, 200)
LINE_THICK = 2

class DraggableQuad:
    def __init__(self, img_w, img_h):
        self.img_w = img_w
        self.img_h = img_h
        self.reset_to_corners()
        self.active = -1

    def reset_to_corners(self):
        w, h = self.img_w, self.img_h
        self.pts = np.array([[int(0.12*w), int(0.12*h)],
                             [int(0.88*w), int(0.12*h)],
                             [int(0.88*w), int(0.88*h)],
                             [int(0.12*w), int(0.88*h)]], dtype=np.int32)

    def draw(self, img, scale=1.0):
        cv2.polylines(img, [self.pts.reshape((-1,1,2))], True, LINE_COLOR, LINE_THICK, cv2.LINE_AA)
        for i, (x,y) in enumerate(self.pts):
            color = ACTIVE_HANDLE_COLOR if i == self.active else HANDLE_COLOR
            cv2.circle(img, (int(x*scale), int(y*scale)), int(HANDLE_RADIUS*scale), color, -1, cv2.LINE_AA)

    def find_handle(self, x, y):
        dists = np.hypot(self.pts[:,0]-x, self.pts[:,1]-y)
        idx = int(np.argmin(dists))
        if dists[idx] <= HANDLE_RADIUS*2.5:
            return idx
        return -1

    def start_drag(self, x, y):
        self.active = self.find_handle(x, y)

    def drag(self, x, y):
        if self.active >= 0:
            nx = max(0, min(self.img_w-1, int(x)))
            ny = max(0, min(self.img_h-1, int(y)))
            self.pts[self.active] = [nx, ny]

    def end_drag(self):
        self.active = -1

    def as_float32(self):
        return self.pts.astype("float32")

_state = {"quad": None, "mouse_down": False}
def _mouse_cb(event, x, y, flags, param):
    q = _state.get("quad")
    if q is None:
        return
    # display is scaled; param contains current scale
    scale = param.get("scale", 1.0)
    # convert window coords back to full-resolution coords
    full_x = int(x / scale)
    full_y = int(y / scale)
    if event == cv2.EVENT_LBUTTONDOWN:
        q.start_drag(full_x, full_y)
        _state["mouse_down"] = True
    elif event == cv2.EVENT_MOUSEMOVE and _state["mouse_down"]:
        q.drag(full_x, full_y)
    elif event == cv2.EVENT_LBUTTONUP:
        q.end_drag()
        _state["mouse_down"] = False

cv2.setMouseCallback(WIN, _mouse_cb, {"scale": 0.6})

def warp_from_quad(frame_bgr, quad_pts, out_size):
    dst = np.array([[0,0],[out_size[0]-1,0],[out_size[0]-1,out_size[1]-1],[0,out_size[1]-1]], dtype="float32")
    try:
        M = cv2.getPerspectiveTransform(quad_pts, dst)
        warped = cv2.warpPerspective(frame_bgr, M, out_size, flags=cv2.INTER_LINEAR)
        return warped
    except Exception:
        H, W = frame_bgr.shape[:2]
        target_ratio = out_size[0]/out_size[1]
        current_ratio = W/H
        if current_ratio > target_ratio:
            new_w = int(target_ratio * H)
            x0 = max(0, (W - new_w)//2)
            crop = frame_bgr[:, x0:x0+new_w]
        else:
            new_h = int(W / target_ratio)
            y0 = max(0, (H - new_h)//2)
            crop = frame_bgr[y0:y0+new_h, :]
        return cv2.resize(crop, out_size, interpolation=cv2.INTER_AREA)

def main_loop(cam_index=0, preview_scale=0.55):
    picam = None
    if PICAMERA2_AVAILABLE:
        try:
            picam = Picamera2()
            cfg = picam.create_preview_configuration({"size": HIGH_RES})
            picam.configure(cfg)
            picam.start()
            time.sleep(CAMERA_WARMUP_SEC)
            print(f"Picamera2 started at {HIGH_RES}")
        except Exception as e:
            print("Picamera2 start error:", e)
            picam = None

    last_saved_norm = None
    quad = None

    print("Inspector (high-res): Space/S=save M=match R=reset +/- to adjust scale Q to quit")
    while True:
        frame = None
        if picam is not None:
            try:
                arr = picam.capture_array()
                frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print("Picamera2 capture error:", e)
                frame = None

        if frame is None and capture_frame is not None:
            try:
                frame = capture_frame(cam_index)
            except Exception as e:
                print("capture_frame error:", e)
                frame = None

        if frame is None:
            wait_img = np.zeros((360,640,3), dtype=np.uint8)
            cv2.putText(wait_img, "No camera frame", (20,180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
            cv2.imshow(WIN, wait_img)
            k = cv2.waitKey(200) & 0xFF
            if k in (ord('q'), 27):
                break
            continue

        h, w = frame.shape[:2]
        if quad is None or quad.img_w != w or quad.img_h != h:
            quad = DraggableQuad(w, h)
            _state["quad"] = quad

        cl, ch, eps_scale, min_area, show_norm = read_trackbar_vals()

        # compute normalized image by warping quad -> NORMALIZED_SIZE
        norm = warp_from_quad(frame, quad.as_float32(), tuple(NORMALIZED_SIZE))

        # decide what to show: normalized (high quality) or raw frame
        if show_norm:
            display = norm.copy()
            disp_scale = preview_scale * (NORMALIZED_SIZE[0] / w)  # maintain visual size relative to screen
        else:
            display = frame.copy()
            disp_scale = preview_scale

        # draw quad handles only when showing raw frame
        if not show_norm:
            quad.draw(display, scale=1.0)

        # overlay info
        info = f"Res capture {w}x{h} display scale {preview_scale:.2f}  S=save M=match R=reset +/- adjust display"
        cv2.putText(display, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        # scaled preview
        sw = max(160, int(display.shape[1] * disp_scale))
        sh = max(120, int(display.shape[0] * disp_scale))
        show_img = cv2.resize(display, (sw, sh), interpolation=cv2.INTER_AREA)

        # update mouse callback scale param so drag coords map correctly
        cv2.setMouseCallback(WIN, _mouse_cb, {"scale": disp_scale})

        cv2.imshow(WIN, show_img)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            quad.reset_to_corners()
            print("Quad reset")
        elif key in (ord('s'), ord(' ')):
            ts = datetime.now().strftime("%Y%m%dT%H%M%S")
            try:
                out_path = DEBUG_DIR / f"{ts}_norm.png"
                cv2.imwrite(str(out_path), norm)
                last_saved_norm = out_path
                print("Saved normalized crop:", out_path)
            except Exception as e:
                print("Save error:", e)
        elif key == ord('m'):
            if last_saved_norm is None:
                print("No saved normalized crop to match. Save first with S.")
            else:
                try:
                    img = cv2.imread(str(last_saved_norm))
                    if img is None:
                        print("Failed to read saved image:", last_saved_norm)
                        continue
                    if compute_phash_bgr is None or load_db_phashes is None:
                        print("Matcher utilities unavailable")
                        continue
                    ph = compute_phash_bgr(img)
                    print("Query phash:", ph)
                    rows = load_db_phashes()
                    cands = top_phash_candidates(ph, rows, top_n=10)
                    if not cands:
                        print("No phash candidates")
                        continue
                    d0, cid0, name0, ph0, desc0 = cands[0]
                    print(f"Top candidate h={d0} id={cid0} name={name0}")
                    matched = False
                    if d0 <= PHASH_STRICT_THRESHOLD:
                        print("Strict accept:", cid0, name0)
                        matched = True
                    else:
                        for d, cid, name, phx, desc in cands:
                            desc_path = Path("data/scryfall_db/descriptors") / (desc if desc else f"{cid}.npz")
                            ok, good = verify_candidate_orb(img, desc_path)
                            print("Verify", name, "h=", d, "ok=", ok, "good=", good)
                            if ok:
                                print("Verified match:", cid, name)
                                matched = True
                                break
                        if not matched:
                            print("No verified match; top h=", d0)
                except Exception as e:
                    print("Match error:", e)
        elif key == ord('+') or key == ord('='):
            preview_scale = min(1.0, preview_scale + 0.05)
            print("Preview scale ->", preview_scale)
        elif key == ord('-') or key == ord('_'):
            preview_scale = max(0.2, preview_scale - 0.05)
