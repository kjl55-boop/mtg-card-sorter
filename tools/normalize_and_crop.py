#!/usr/bin/env python3
"""
Normalize, deskew, and crop card images for phash testing.

Usage:
  python tools/normalize_and_crop.py            # processes data/debug -> data/debug_norm
  python tools/normalize_and_crop.py --input mydir --out outdir --w 400 --h 560

Requires: opencv-python-headless, Pillow, imagehash, numpy
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import imagehash

def four_point_transform(image, pts, w, h):
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts.astype("float32"), dst)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped

def largest_quad_contour(img_gray):
    # blur + canny then find contours
    blurred = cv2.GaussianBlur(img_gray, (5,5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # sort by area descending and try to approximate a quad
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours[:20]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 1000:
            return approx.reshape(4,2)
    return None

def order_points(pts):
    # order: tl, tr, br, bl
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def fallback_center_crop(img, w, h):
    H, W = img.shape[:2]
    cx, cy = W//2, H//2
    x0 = max(0, cx - w//2); y0 = max(0, cy - h//2)
    x1 = min(W, x0 + w); y1 = min(H, y0 + h)
    return img[y0:y1, x0:x1]

def compute_phash_from_cv(img_cv):
    # convert to PIL and compute phash
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    return str(imagehash.phash(img_pil))

def process_file(p, outdir, w, h, debug=False):
    img = cv2.imread(str(p))
    if img is None:
        print("FAILED to read:", p)
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    quad = largest_quad_contour(gray)
    if quad is not None:
        ordered = order_points(quad)
        warped = four_point_transform(img, ordered, w, h)
    else:
        # fallback: try simple threshold to find big contour or center crop
        warped = fallback_center_crop(img, w, h)
        # ensure size by resizing
        warped = cv2.resize(warped, (w,h), interpolation=cv2.INTER_AREA)
    outp = outdir / p.name
    cv2.imwrite(str(outp), warped)
    ph = compute_phash_from_cv(warped)
    if debug:
        print(f"{p.name}: wrote {outp} phash {ph}")
    return outp, ph

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", default="data/debug", help="Input dir")
    ap.add_argument("--out", "-o", default="data/debug_norm", help="Output dir")
    ap.add_argument("--w", type=int, default=400, help="Output width")
    ap.add_argument("--h", type=int, default=560, help="Output height")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in inp.iterdir() if p.suffix.lower() in (".png",".jpg",".jpeg")])
    if not images:
        print("No images found in", inp)
        return

    print(f"Processing {len(images)} images -> {out} ({args.w}x{args.h})")
    results = []
    for p in images:
        res = process_file(p, out, args.w, args.h, debug=args.debug)
        if res:
            results.append((p.name, str(res[0]), res[1]))
    # write simple summary
    for name, outpath, ph in results:
        print(f"{name} -> {Path(outpath).name}  phash={ph}")

if __name__ == "__main__":
    main()
