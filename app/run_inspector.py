"""
Interactive inspector UI

- Shows live preview from capture_frame
- Trackbars to tweak detection: canny low/high, poly_eps_scale, min_area
- Press S to save the current normalized crop to data/debug/<timestamp>.png
- Press M to compute phash and run matcher verification for the current frame
- Press Q or ESC to quit
"""
import time
import cv2
from pathlib import Path
from datetime import datetime
from .capture import capture_frame, normalize_and_save, compute_phash_bgr
from .crop import normalize_card_image
from .matcher import load_db_phashes, top_phash_candidates, verify_candidate_orb
from .config import NORMALIZED_SIZE, PHASH_STRICT_THRESHOLD, PHASH_RELAXED_THRESHOLD
import numpy as np

DEBUG_DIR = Path("data/debug")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# default detection params (these mirror values in app/crop but are adjustable here)
DEFAULTS = {
    "canny_low": 50,
    "canny_high": 150,
    "poly_eps_scale": 2,   # trackbar value; actual eps = scale/100 * peri (converted below)
    "min_area": 2000
}

WIN = "Inspector"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

# create trackbars for runtime tuning
cv2.createTrackbar("canny_low", WIN, DEFAULTS["canny_low"], 300, lambda v: None)
cv2.createTrackbar("canny_high", WIN, DEFAULTS["canny_high"], 400, lambda v: None)
cv2.createTrackbar("poly_eps_scale", WIN, DEFAULTS["poly_eps_scale"], 50, lambda v: None)
cv2.createTrackbar("min_area", WIN, DEFAULTS["min_area"], 20000, lambda v: None)
cv2.createTrackbar("show_norm", WIN, 1, 1, lambda v: None)   # 1 = show normalized crop, 0 = show raw frame

def _read_trackbars():
    cl = cv2.getTrackbarPos("canny_low", WIN)
    ch = cv2.getTrackbarPos("canny_high", WIN)
    eps_scale = cv2.getTrackbarPos("poly_eps_scale", WIN)
    min_area = cv2.getTrackbarPos("min_area", WIN)
    show_norm = bool(cv2.getTrackbarPos("show_norm", WIN))
    return cl, ch, max(1, eps_scale), max(100, min_area), show_norm

# small helper to run the same normalization as app.crop but using runtime params
def runtime_normalize(frame_bgr, out_size=tuple(NORMALIZED_SIZE), canny_low=50, canny_high=150, eps_scale=2, min_area=2000):
    import numpy as np
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blurred, canny_low, canny_high)
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours[:20]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps_scale/100.0 * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > min_area:
                pts = approx.reshape(4,2).astype("float32")
                # order points (tl,tr,br,bl)
                s = pts.sum(axis=1); diff = np.diff(pts, axis=1)
                rect = np.zeros((4,2), dtype="float32")
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]
                dst = np.array([[0,0],[out_size[0]-1,0],[out_size[0]-1,out_size[1]-1],[0,out_size[1]-1]], dtype="float32")
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(frame_bgr, M, out_size)
                return warped, approx, edged
    # fallback: center-crop and resize
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
    return resized, None, edged if 'edged' in locals() else None

def overlay_text(img, lines, pos=(10,20), line_height=18):
    x,y = pos
    for i,l in enumerate(lines):
        cv2.putText(img, l, (x, y + i*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

def main_loop(cam_index=0):
    print("Interactive inspector: S=save, M=match, Q/ESC=quit")
    last_match_info = ""
    while True:
        try:
            frame = capture_frame(cam_index)
        except Exception as e:
            overlay = np.zeros((200,400,3), dtype=np.uint8)
            overlay_text(overlay, ["capture error: " + str(e)])
            cv2.imshow(WIN, overlay)
            key = cv2.waitKey(1000) & 0xFF
            if key in (ord('q'), 27):
                break
            continue

        cl, ch, eps_scale, min_area, show_norm = _read_trackbars()
        norm, approx, edged = runtime_normalize(frame, out_size=tuple(NORMALIZED_SIZE),
                                                canny_low=cl, canny_high=ch,
                                                eps_scale=eps_scale, min_area=min_area)
        display = norm if show_norm else frame.copy()

        # overlay status
        lines = [
            f"canny {cl}/{ch} eps_scale {eps_scale} min_area {min_area}",
            "S=save  M=match  Q=quit",
            last_match_info
        ]
        overlay_text(display, lines)

        # if found quad, draw it on display (small indicator)
        if approx is not None and not show_norm:
            cv2.polylines(display, [approx], True, (0,255,0), 2)

        cv2.imshow(WIN, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            fname = datetime.now().strftime("capture_%Y%m%dT%H%M%S.png")
            saved = normalize_and_save(norm, fname)
            print("Saved normalized crop to", saved)
            last_match_info = f"Saved {fname}"
        elif key == ord('m'):
            # compute phash and run matching flow on current normalized crop
            ph = compute_phash_bgr(norm)
            print("Query phash:", ph)
            rows = load_db_phashes()
            if not rows:
                print("No phashes in DB")
                last_match_info = "No DB phash"
                continue
            cands = top_phash_candidates(ph, rows, top_n=10)
            if not cands:
                print("No phash candidates")
                last_match_info = "No phash cands"
                continue
            d0, cid0, name0, ph0, desc0 = cands[0]
            print(f"Top phash h={d0} id={cid0} name={name0}")
            matched = False
            if d0 <= PHASH_STRICT_THRESHOLD:
                print("Strict accept:", cid0, name0)
                last_match_info = f"ACCEPT {name0} (h={d0})"
                matched = True
            else:
                for d, cid, name, phx, desc in cands:
                    desc_path = Path("data/scryfall_db/descriptors") / (desc if desc else f"{cid}.npz")
                    ok, good = verify_candidate_orb(norm