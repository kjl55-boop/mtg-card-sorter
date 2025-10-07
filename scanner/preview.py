# scanner/preview.py
import cv2
import numpy as np
from math import ceil
from typing import Sequence, Optional, Tuple
from pathlib import Path
import pytesseract

# reuse stack_images_grid from earlier helper or include it here
def stack_images_grid(scale: float, img_matrix: Sequence[Sequence[np.ndarray]],
                      labels: Optional[Sequence[Sequence[str]]] = None,
                      ocr_text_for_tile: Optional[Tuple[int, int, str]] = None) -> np.ndarray:
    # (Implementation identical to the stack_images_grid provided earlier)
    # Paste the full function body from the helper you accepted.
    raise NotImplementedError("Paste stack_images_grid implementation here.")

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
        final_img = imgs[-1]
        ocr_text = pytesseract.image_to_string(final_img, config=tesseract_config)
        final_idx = len(imgs) - 1
        ocr_triple = (final_idx // cols, final_idx % cols, ocr_text)
        grid_img = stack_images_grid(scale, matrix, labels, ocr_triple)
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
                f.write(ocr_text)
    cv2.destroyAllWindows()