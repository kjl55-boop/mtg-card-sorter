# main.py (place at project root, next to scanner/)
import cv2
import pytesseract
from scanner.capture import capture_once  # or: cv2.imread('examples/sample.jpg')
from scanner.pipeline import compute_pipeline_steps
from scanner.preview import stack_images_grid

def main():
    img = capture_once()  # returns BGR numpy array
    steps = compute_pipeline_steps(img)  # OrderedDict[name->image] or list of tuples

    # normalize to names, imgs lists
    if isinstance(steps, dict):
        names = list(steps.keys())
        imgs = [steps[n] for n in names]
    else:
        names, imgs = zip(*steps)

    # choose columns and build img_matrix + label_matrix
    cols = 3
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
                blank = 255 * (imgs[0].astype('uint8')*0)
                row_imgs.append(blank)
                row_labels.append('')
        img_matrix.append(row_imgs)
        label_matrix.append(row_labels)

    # run OCR on selected steps
    ocr_texts = []
    want = {'candidate_crop', 'final', 'deskewed'}  # pick step names you care about
    for idx, name in enumerate(names):
        if name not in want:
            continue
        tile = imgs[idx]
        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY) if tile.ndim == 3 else tile
        text = pytesseract.image_to_string(gray, config='--oem 1 --psm 6').strip()
        if text:
            r = idx // cols
            c = idx % cols
            ocr_texts.append((r, c, text))

    # render and show (your stack_images_grid must accept ocr_texts)
    grid = stack_images_grid(scale=0.45, img_matrix=img_matrix, labels=label_matrix, ocr_texts=ocr_texts)
    cv2.imshow('Pipeline Grid', grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()