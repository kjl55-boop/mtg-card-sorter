"""
GUI inspector with live preview, trackbars, and four draggable corner controls.

Controls:
 - Drag the small corner handles to adjust the card quadrilateral
 - Space or S : save normalized crop (uses normalize_and_save when available)
 - M : run matching on the last saved normalized crop
 - R : reset handles to image corners
 - Q or ESC : quit

Requirements:
 - Picamera2 installed (preferred) or app.capture.capture_frame available
 - OpenCV with GUI support (cv2.imshow)
 - app.capture, app.crop, app.matcher present for save/normalize/match
"""
from pathlib import Path
from datetime import datetime
import time
import cv2
import numpy as np

# Try to import app utilities; keep graceful fallbacks
try:
    from .capture import capture_frame, normalize_and_save, compute_phash_bgr
    from .matcher import load_db_phashes, top_phash_candidates, verify_candidate_orb
    from .config import CAMERA_PREVIEW_SIZE, CAMERA_WARMUP_SEC, NORMALIZED_SIZE, PHASH_STRICT_THRESHOLD
except Exception:
    capture_frame = None
    normalize_and_save = None
    compute_phash_bgr = None
    load_db_phashes = None
    top_phash_candidates = None
    verify_candidate_orb = None
    CAMERA_PREVIEW_SIZE = (1280, 720)
    CAMERA_WARMUP_SEC = 0.25
    NORMALIZED_SIZE = (400, 560)
    PHASH_STRICT_THRESHOLD = 6

# Picamera2 live capture preference
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except Exception:
    PICAMERA2_AVAILABLE = False

DEBUG_DIR = Path("data/debug")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

WIN = "Inspector"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

# Trackbar defaults and creation
tk_defaults = {
    "canny_low": 50,
    "canny_high": 150,
    "eps_pct": 2,
    "min_area": 2000,
    "show_norm": 1
}
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

# Draggable corner handles implementation
HANDLE_RADIUS = 8
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

    def draw(self, img):
        # polygon
        cv2.polylines(img, [self.pts.reshape((-1,1,2))], True, LINE_COLOR, LINE_THICK, cv2.LINE_AA)
        # handles
        for i, (x,y) in enumerate(self.pts):
            color = ACTIVE_HANDLE_COLOR if i == self.active else HANDLE_COLOR
            cv2.circle(img, (int(x), int(y)), HANDLE_RADIUS, color, -1, cv2.LINE_AA)

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
            # clamp inside image
            nx = max(0, min(self.img_w-1, int(x)))
            ny = max(0, min(self.img_h-1, int(y)))
            self.pts[self.active] = [nx, ny]

    def end_drag(self):
        self.active = -1

    def as_float32(self):
        return self.pts.astype("float32")

# Mouse callbacks wired to OpenCV window
_state = {"quad": None, "mouse_down": False}
def _mouse_cb(event, x, y, flags, param):
    q = _state.get("quad")
    if q is None:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        q.start_drag(x, y)
        _state["mouse_down"] = True
    elif event == cv2.EVENT_MOUSEMOVE and _state["mouse_down"]:
        q.drag(x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        q.end_drag()
        _state["mouse_down"] = False

cv2.setMouseCallback(WIN, _mouse_cb)

def runtime_normalize_with_quad(frame_bgr, quad_pts, out_size):
    # quad_pts expected as float32 array shape (4,2)
    dst = np.array([[0, 0], [out_size[0]-1, 0], [out_size[0]-1, out_size[1]-1], [0, out_size[1]-1]], dtype="float32")
    try:
        M = cv2.getPerspectiveTransform(quad_pts, dst)
        warped = cv2.warpPerspective(frame_bgr, M, out_size, flags=cv2.INTER_LINEAR)
        return warped
    except Exception:
        # fallback center-crop
        H, W = frame_bgr.shape[:2]
        target_ratio = out_size[0] / out_size[1]
        current_ratio = W / H
        if current_ratio > target_ratio:
            new_w = int(target_ratio * H)
            x0 = max(0, (W - new_w)//2)
            crop = frame_bgr[:, x0:x0+new_w]
        else:
            new_h = int(W / target_ratio)
            y0 = max(0, (H - new_h)//2)
            crop = frame_bgr[y0:y0+new_h, :]
        resized = cv2.resize(crop, out_size, interpolation=cv2.INTER_AREA)
        return resized

def main_loop(cam_index=0, preview_scale=0.6):
    # init capture (Picamera2 preferred)
    picam = None
    if PICAMERA2_AVAILABLE:
        try:
            picam = Picamera2()
            cfg = picam.create_preview_configuration({"size": CAMERA_PREVIEW_SIZE})
            picam.configure(cfg)
            picam.start()
            time.sleep(CAMERA_WARMUP_SEC)
        except Exception:
            picam = None

    last_saved_norm = None

    print("Inspector started. Drag corner handles to adjust quad. S=save M=match R=reset Q=quit")
    quad = None
    while True:
        # grab frame
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
            blank = np.zeros((360,640,3), dtype=np.uint8)
            cv2.putText(blank, "No camera frame", (20,180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
            cv2.imshow(WIN, blank)
            k = cv2.waitKey(200) & 0xFF
            if k in (ord('q'), 27):
                break
            continue

        h, w = frame.shape[:2]
        if quad is None or quad.img_w != w or quad.img_h != h:
            quad = DraggableQuad(w, h)
            _state["quad"] = quad

        cl, ch, eps_scale, min_area, show_norm = read_trackbar_vals()

        # compute normalized crop from quad
        norm = runtime_normalize_with_quad(frame, quad.as_float32(), tuple(NORMALIZED_SIZE))
        display = norm.copy() if show_norm else frame.copy()

        # overlay quad on display (if showing raw frame, draw quad mapped to full frame; if showing norm, draw rectangle)
        if show_norm:
            # show handles mapped onto normalized crop edges by drawing the rectangle border
            cv2.putText(display, "Normalized view (drag handles on raw view)", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
        else:
            quad.draw(display)

        # info overlay
        info = f"canny {cl}/{ch} eps {eps_scale:.3f} min_area {min_area}  S=save M=match R=reset Q=quit"
        cv2.putText(display, info, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

        # scaled preview
        sw = int(w * preview_scale)
        sh = int(h * preview_scale)
        show_img = cv2.resize(display, (sw, sh), interpolation=cv2.INTER_AREA)
        cv2.imshow(WIN, show_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            quad.reset_to_corners()
            print("Quad reset")
        elif key in (ord('s'), ord(' ')):
            ts = datetime.now().strftime("%Y%m%dT%H%M%S")
            # save normalized crop via normalize_and_save if available (it expects full-frame; pass the warped result as full normalized)
            try:
                if normalize_and_save is not None:
                    # normalize_and_save assumes a full-frame and will detect; we already have normalized image, so save directly
                    out_path = DEBUG_DIR / f"{ts}_norm.png"
                    cv2.imwrite(str(out_path), norm)
                else:
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

    try:
        if PICAMERA2_AVAILABLE and 'picam' in locals() and picam is not None:
            picam.stop()
    except Exception:
        pass
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()
