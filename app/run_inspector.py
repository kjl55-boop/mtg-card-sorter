"""
Interactive GUI inspector

- Live preview with trackbars to adjust detection parameters
- S to save normalized crop
- M to run matching on current normalized crop
- Q or ESC to quit
"""
import time
import cv2
from pathlib import Path
from datetime import datetime
from .capture import capture_frame, normalize_and_save, compute_phash_bgr
from .matcher import load_db_phashes, top_phash_candidates, verify_candidate_orb
from .config import NORMALIZED_SIZE, PHASH_STRICT_THRESHOLD
import numpy as np

DEBUG_DIR = Path("data/debug")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

WIN = "Inspector"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

# trackbar defaults
deftk = {
    "canny_low": 50,
    "canny_high": 150,
    "eps_scale_percent": 2,   # used as eps = percent/100 * peri
    "min_area": 2000,
    "show_norm": 1
}

cv2.createTrackbar("canny_low", WIN, deftk["canny_low"], 300, lambda v: None)
cv2.createTrackbar("canny_high", WIN, deftk["canny_high"], 400, lambda v: None)
cv2.createTrackbar("eps_scale_percent", WIN, deftk["eps_scale_percent"], 50, lambda v: None)
cv2.createTrackbar("min_area", WIN, deftk["min_area"], 20000, lambda v: None)
cv2.createTrackbar("show_norm", WIN, deftk["show_norm"], 1, lambda v: None)

def read_trackbar_vals():
    cl = cv2.getTrackbarPos("canny_low", WIN)
    ch = cv2.getTrackbarPos("canny_high", WIN)
    eps_pct = cv2.getTrackbarPos("eps_scale_percent", WIN)
    min_area = cv2.getTrackbarPos("min_area", WIN)
    show_norm = bool(cv2.getTrackbarPos("show_norm", WIN))
    return cl, ch, max(1, eps_pct) / 100.0, max(100, min_area), show_norm

def runtime_normalize(frame_bgr, out_size=tuple(NORMALIZED_SIZE), canny_low=50, canny_high=150, eps_scale=0.02, min_area=2000):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blurred, canny_low, canny_high)
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours[:20]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps_scale * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > min_area:
                pts = approx.reshape(4,2).astype("float32")
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
    # fallback
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
    print("Inspector: S=save, M=match, Q/ESC=quit")
    status_line = ""
    while True:
        try:
            frame = capture_frame(cam_index)
        except Exception as e:
            err = np.zeros((200,400,3), dtype=np.uint8)
            overlay_text(err, ["capture error: " + str(e)])
            cv2.imshow(WIN, err)
            k = cv2.waitKey(500) & 0xFF
            if k in (ord('q'), 27):
                break
            continue

        cl, ch, eps_scale, min_area, show_norm = read_trackbar_vals()
        norm, approx, edged = runtime_normalize(frame, out_size=tuple(NORMALIZED_SIZE),
                                                canny_low=cl, canny_high=ch,
                                                eps_scale=eps_scale, min_area=min_area)
        display = norm if show_norm else frame.copy()

        lines = [
            f"canny {cl}/{ch} eps {eps_scale:.3f} min_area {min_area}",
            "S=save  M=match  Q=quit",
            status_line
        ]
        overlay_text(display, lines)

        if approx is not None and not show_norm:
            cv2.polylines(display, [approx], True, (0,255,0), 2)

        cv2.imshow(WIN, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            fname = datetime.now().strftime("capture_%Y%m%dT%H%M%S.png")
            saved = normalize_and_save(norm, fname, canny_low=cl, canny_high=ch, eps_scale=eps_scale, min_area=min_area)
            print("Saved normalized crop to", saved)
            status_line = f"Saved {fname}"
        elif key == ord('m'):
            ph = compute_phash_bgr(norm)
            print("Query phash:", ph)
            rows = load_db_phashes()
            if not rows:
                print("No phashes in DB")
                status_line = "No DB phash"
                continue
            cands = top_phash_candidates(ph, rows, top_n=10)
            if not cands:
                print("No phash candidates")
                status_line = "No phash candidates"
                continue
            d0, cid0, name0, ph0, desc0 = cands[0]
            print(f"Top phash h={d0} id={cid0} name={name0}")
            matched = False
            if d0 <= PHASH_STRICT_THRESHOLD:
                print("Strict accept:", cid0, name0)
                status_line = f"ACCEPT {name0} (h={d0})"
                matched = True
            else:
                for d, cid, name, phx, desc in cands:
                    desc_path = Path("data/scryfall_db/descriptors") / (desc if desc else f"{cid}.npz")
                    ok, good = verify_candidate_orb(norm, desc_path)
                    print("Verify", name, "h=", d, "ok=", ok, "good=", good)
                    if ok:
                        status_line = f"VERIFIED {name} (h={d} good={good})"
                        matched = True
                        break
                if not matched:
                    status_line = f"No verified match (top h={d0})"
        elif key in (ord('q'), 27):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()
