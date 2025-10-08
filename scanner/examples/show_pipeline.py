
import cv2
import pytesseract
from collections import OrderedDict
from scanner.capture import capture_once   # or load_image
from scanner.pipeline import compute_pipeline_steps
from scanner.preview import stack_images_grid

def run_and_show_pipeline_grid(source_img, cols=3, scale=0.45, tesseract_config='--oem 1 --psm 6'):
    # 1. Compute ordered steps
    steps = compute_pipeline_steps(source_img)  # OrderedDict or list of (name,img)
    if isinstance(steps, dict):
        names = list(steps.keys())
        imgs = [steps[n] for n in names]
    else:
        names, imgs = zip(*steps)

    # 2. Build rows/cols matrix and labels
    cols = int(cols)
    rows = (len(imgs) + cols - 1) // cols
    img_matrix = []
    label_matrix = []
    for r in range(rows):
        row_imgs = []
        row_labels = []
        for c in range(cols):
            idx = r * cols + c
            if idx < len(imgs):
                row_imgs.append(imgs[idx])
                row_labels.append(names[idx])
            else:
                # use a blank same-shape tile for missing cells
                blank = (255 * (imgs[0].astype('uint8')*0)).astype('uint8')
                row_imgs.append(blank)
                row_labels.append('')
        img_matrix.append(row_imgs)
        label_matrix.append(row_labels)

    # 3. Run OCR on the steps you care about (example: run on all)
    ocr_texts = []   # list of (row, col, text)
    for idx, name in enumerate(names):
        # pick a subset if desired: if name not in ('candidate_crop','final'): continue
        tile = imgs[idx]
        if tile.ndim == 3:
            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        else:
            gray = tile
        text = pytesseract.image_to_string(gray, config=tesseract_config).strip()
        if text:
            r = idx // cols
            c = idx % cols
            ocr_texts.append((r, c, text))

    # 4. Render grid with labels and OCR overlays
    grid = stack_images_grid(scale=scale, img_matrix=img_matrix, labels=label_matrix, ocr_texts=ocr_texts)
    cv2.imshow('Pipeline Grid', grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    img = capture_once()   # or cv2.imread('examples/sample.jpg')
    run_and_show_pipeline_grid(img, cols=3)
