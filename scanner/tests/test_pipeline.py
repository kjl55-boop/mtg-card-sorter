# scanner/tests/test_pipeline.py
import numpy as np
from scanner.pipeline import deskew_image, compute_pipeline_steps

def test_deskew_noop_for_blank():
    img = np.full((100,200), 255, dtype=np.uint8)
    out = deskew_image(img)
    assert out.shape == img.shape

def test_compute_steps_shapes():
    # create a simple BGR test image
    img = np.zeros((480,640,3), dtype=np.uint8)
    steps = compute_pipeline_steps(img)
    assert isinstance(steps, dict)
    assert "01_gray" in steps and "10_final" in steps