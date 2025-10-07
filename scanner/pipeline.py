# scanner/pipeline.py
import cv2
import numpy as np
from collections import OrderedDict
from typing import Optional, Tuple

def deskew_image(img_gray: np.ndarray) -> np.ndarray:
    if img_gray is None or img_gray.size == 0:
        return img_gray
    coords = np.column_stack(np.where(img_gray < 255))
    if coords.shape[0] < 10:
        return img_gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h,w) = img_gray.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(img_gray, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def order_pts(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")

def compute_pipeline_steps(img_bgr: np.ndarray, roi: Optional[Tuple[int,int,int,int]] = None) -> OrderedDict:
    """
    Returns OrderedDict of step_name -> image (single-channel for intermediate grayscale steps).
    """
    steps = OrderedDict()
    if roi is not None:
        x,y,w,h = roi
        img = img_bgr[y:y+h, x:x+w].copy()
    else:
        img = img_bgr.copy()

    # 01 gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    steps["01_gray"] = gray

    # 02 denoise
    denoised = cv2.GaussianBlur(gray, (3,3), 0)
    steps["02_denoised"] = denoised

    # 03 clahe
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    steps["03_clahe"] = enhanced

    # 04 adaptive threshold (binary inverted)
    th = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 10)
    steps["04_adaptivethresh"] = th

    # 05 morph close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,3))
    morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    steps["05_morph"] = morph

    # 06 candidate detection and crop/warp
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis = img.copy()
    candidate_crop = enhanced.copy()
    box_points = None
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 1000:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                box_points = approx.reshape(4,2)
                break
        if box_points is None:
            x,y,w,h = cv2.boundingRect(contours[0])
            candidate_crop = enhanced[y:y+h, x:x+w]
            cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
        else:
            pts = order_pts(box_points)
            cv2.polylines(vis, [box_points.astype(int)], True, (0,255,0), 3)
            (tl,tr,br,bl) = pts
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxW = int(max(widthA, widthB))
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxH = int(max(heightA, heightB))
            dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(pts, dst)
            warped = cv2.warpPerspective(enhanced, M, (maxW, maxH))
            candidate_crop = warped

    steps["06_candidate_vis"] = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
    steps["07_candidate_crop"] = candidate_crop

    # 08 deskew
    deskewed = deskew_image(candidate_crop)
    steps["08_deskewed"] = deskewed

    # 09 sharpen
    blur = cv2.GaussianBlur(deskewed, (0,0), sigmaX=1.0)
    sharp = cv2.addWeighted(deskewed, 1.5, blur, -0.5, 0)
    steps["09_sharp"] = sharp

    # 10 final threshold
    final = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 10)
    steps["10_final"] = final

    return steps