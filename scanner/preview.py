# scanner/preview.py
import cv2
import numpy as np
from math import ceil
from typing import Sequence, Optional, Tuple
from pathlib import Path
import pytesseract
import time

def stack_images_grid(scale, img_matrix, labels=None, ocr_texts=None):
    """
    img_matrix: 2D list of images
    labels: 2D list of strings (same shape as img_matrix)
    ocr_texts: list of (row, col, text) to overlay
    """
    rows = len(img_matrix)
    cols = len(img_matrix[0]) if rows > 0 else 0
    height, width = img_matrix[0][0].shape[:2]

    # Resize and label each image
    labeled_rows = []
    target_size = None  # will be set after first image

    for r in range(rows):
        row_imgs = []
        for c in range(cols):
            img = img_matrix[r][c]
            if img is None:
                img = np.zeros((100, 100, 3), dtype=np.uint8)  # fallback

            # Convert grayscale to BGR
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Resize to scale
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

            # Set target size from first image
            if target_size is None:
                target_size = img.shape[:2]  # (height, width)

            # Resize to match target size
            img = cv2.resize(img, (target_size[1], target_size[0]))

            # Add label if available
            label = labels[r][c] if labels else ""
            if label:
                cv2.putText(img, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            row_imgs.append(img)
        labeled_rows.append(row_imgs)

    # Overlay OCR text
    if ocr_texts:
        for r, c, text in ocr_texts:
            if r < len(labeled_rows) and c < len(labeled_rows[r]):
                img = labeled_rows[r][c]
                lines = text.strip().splitlines()
                for i, line in enumerate(lines[:3]):  # show up to 3 lines
                    y = 40 + i * 20
                    cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Stack into grid
    grid_img = cv2.vconcat([cv2.hconcat(row) for row in labeled_rows])
    return grid_img


def show_pipeline_grid_with_ocr(base_img, roi=None, cols=3, scale=0.45, debug_dir="debug_grid",
                                live=False, tesseract_config=r'--oem 1 --psm 6'):
    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    while True:
        steps = __import__("scanner.pipeline", fromlist=["compute_pipeline_steps"]).compute_pipeline_steps(base_img, roi=roi)
        names = list(steps.keys())
        imgs = [steps[n] for n in names]
        rows = (len(imgs) + cols - 1) // cols
        matrix = []
        labels = []
        for r in range(rows):
            row_imgs = []
            row_labels = []
            for c in range(cols):
                idx = r*cols + c
                if idx < len(imgs):
                    row_imgs.append(imgs[idx])
                    row_labels.append(names[idx])
                else:
                    row_imgs.append(np.zeros_like(imgs[0]))
                    row_labels.append("")
            matrix.append(row_imgs)
            labels.append(row_labels)
        ocr_texts = []
        for idx, name in enumerate(names):
            tile = imgs[idx]
            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY) if tile.ndim == 3 else tile
            text = pytesseract.image_to_string(gray, config=tesseract_config).strip()
            if text:
                r = idx // cols
                c = idx % cols
                ocr_texts.append((r, c, text))

        grid_img = stack_images_grid(scale, matrix, labels, ocr_texts)
        cv2.imshow("Pipeline Grid with OCR", grid_img)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('s'):
            ts = int(time.time())
            for i, (n, im) in enumerate(zip(names, imgs)):
                fname = Path(debug_dir) / f"{ts}_{i:02d}_{n}.png"
                cv2.imwrite(str(fname), im)
            with open(Path(debug_dir) / f"{ts}_ocr.txt", "w") as f:
                for r, c, text in ocr_texts:
                    f.write(f"[{r},{c}] {text.strip()}\n\n")
    cv2.destroyAllWindows()