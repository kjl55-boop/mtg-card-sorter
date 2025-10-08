# scanner/cli.py
import argparse
import cv2
from .capture import capture_with_rpicam_still
from .pipeline import compute_pipeline_steps
from .preview import show_pipeline_grid_with_ocr
from scanner.live_preview import run_live_preview


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--use-rpicam", action="store_true", help="Use rpicam-still wrapper")
    p.add_argument("--file", type=str, help="Path to image file to preview instead of capturing")
    args = p.parse_args()

    run_live_preview()

    '''if args.file:
        img = cv2.imread(args.file)
    else:
        img = capture_with_rpicam_still("capture_for_preview.jpg")

    steps = compute_pipeline_steps(img)
    show_pipeline_grid_with_ocr(img)'''

if __name__ == "__main__":
    main()