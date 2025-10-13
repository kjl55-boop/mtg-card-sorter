"""
Persistent Camera abstraction with backend failover plus helper utilities.

Public API:
- Camera(...) : class for persistent usage
- init_camera(size) -> camera_handle (compat)
- grab_frame(cam_handle, timeout) -> np.ndarray (compat)
- close_camera(cam_handle) (compat)
- capture_frame(cam_index=0, timeout=2.0) -> np.ndarray (single-shot convenience)
- normalize_and_save(...)
- compute_phash_bgr(...)
"""

from pathlib import Path
import threading
import time
import tempfile
import subprocess
import os
import shutil
from time import sleep
from typing import Optional

import cv2
from PIL import Image
import imagehash
import numpy as np

from .crop import normalize_card_image

from . import utils
log = utils.get_logger("camera")

# NOTE: update these imports to match your config/__init__ layout
from .config import CAMERA_PREVIEW_SIZE, CAMERA_WARMUP_SEC, NORMALIZED_SIZE, DEBUG_DIR

# preserve previous behavior of ensuring debug dir exists (can move to app.__init__ later)
#Path(DEBUG_DIR).mkdir(parents=True, exist_ok=True)
OUT_W, OUT_H = NORMALIZED_SIZE

class CameraError(RuntimeError):
    pass

class Camera:
    """
    Persistent camera abstraction.
    Use start()/read()/stop() for lifecycle-managed capture.
    Optional background thread keeps latest frame for low-latency reads.
    """

    def __init__(self, preview_size=CAMERA_PREVIEW_SIZE, warmup_sec=CAMERA_WARMUP_SEC,
                 use_background_thread: bool = True, cam_index: int = 0):
        self.preview_size = preview_size
        self.warmup_sec = warmup_sec
        self.cam_index = cam_index
        self.use_background_thread = use_background_thread

        self._handle = None
        self.backend_name: Optional[str] = None
        self._running = False

        self._thread: Optional[threading.Thread] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    # ---------------- backends (keep heavy imports local) -----------------
    def _open_picamera2(self):
        try:
            from picamera2 import Picamera2
        except Exception:
            log.debug("picamera2 not available")
            return None
        try:
            pc2 = Picamera2()
            cfg = pc2.create_preview_configuration({"size": self.preview_size})
            pc2.configure(cfg)
            pc2.start()
            time.sleep(self.warmup_sec)
            self.backend_name = "picamera2"
            log.info("Selected backend: picamera2")
            return pc2
        except Exception:
            utils.log_exception(log, exc, "Failed to initialize Picamera2")
            return None

    def _open_libcamera_jpeg(self):
        if not shutil.which("libcamera-jpeg"):
            return None
        # we'll execute libcamera-jpeg per-read; mark backend sentinel
        self.backend_name = "libcamera-jpeg"
        return "libcamera-jpeg"

    def _open_opencv(self):
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            return None
        # warm-up reads
        for _ in range(2):
            cap.read()
        self.backend_name = "opencv"
        return cap

    # ---------------- lifecycle ------------------------------------------
    def start(self):
        if self._running:
            return
        self._handle = self._open_picamera2() or self._open_libcamera_jpeg() or self._open_opencv()
        if self._handle is None:
            raise CameraError("No capture backend available")
        self._running = True
        if self.use_background_thread and self.backend_name != "libcamera-jpeg":
            self._thread = threading.Thread(target=self._bg_loop, daemon=True)
            self._thread.start()

    def _bg_loop(self):
        while self._running:
            frame = self._read_once()
            if frame is not None:
                with self._lock:
                    self._frame = frame
            time.sleep(0.01)

    def _read_once(self):
        try:
            if self.backend_name == "picamera2":
                arr = self._handle.capture_array()
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            if self.backend_name == "opencv":
                ret, frame = self._handle.read()
                if not ret:
                    return None
                return frame
            # libcamera-jpeg handled synchronously in read()
            return None
        except Exception:
            return None

    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Return the latest frame. If background thread active, return buffered frame.
        Otherwise perform a synchronous capture depending on backend.
        """
        if not self._running:
            raise CameraError("Camera not started")

        if self.use_background_thread and self._thread is not None:
            start = time.time()
            while True:
                with self._lock:
                    frame = None if self._frame is None else self._frame.copy()
                if frame is not None:
                    log.debug("read: backend %s returned no frame", self.backend_name)
                    return frame
                if timeout is not None and (time.time() - start) >= timeout:
                    return None
                time.sleep(0.005)
        else:
            # synchronous fallback or libcamera-jpeg path
            if self.backend_name == "libcamera-jpeg":
                tmp = Path(tempfile.gettempdir()) / f"libcam_capture_{os.getpid()}.jpg"
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except Exception:
                        pass
                cmd = ["libcamera-jpeg", "-o", str(tmp), "-n", "--timeout", str(int((timeout or 2.0)*1000))]
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=max(5, (timeout or 2.0) + 3))
                    img = cv2.imread(str(tmp))
                except Exception:
                    img = None
                finally:
                    try:
                        tmp.unlink()
                    except Exception:
                        pass
                return img
            else:
                return self._read_once()

    def is_open(self) -> bool:
        return self._running

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        try:
            if self.backend_name == "picamera2" and self._handle is not None:
                self._handle.stop()
            if self.backend_name == "opencv" and self._handle is not None:
                self._handle.release()
        except Exception:
            pass
        self._handle = None
        self.backend_name = None

# ---------------- convenience single-shot API (keeps existing callers working) ----
def capture_frame(cam_index: int = 0, timeout: float = 2.0) -> np.ndarray:
    """
    Single-shot convenience which tries backends in order. Raises RuntimeError on failure.
    """
    cam = Camera(preview_size=CAMERA_PREVIEW_SIZE, warmup_sec=CAMERA_WARMUP_SEC,
                 use_background_thread=False, cam_index=cam_index)
    cam.start()
    try:
        img = cam.read(timeout=timeout)
        if img is None:
            raise RuntimeError("capture_frame: no frame captured")
        return img
    finally:
        cam.stop()

# ---------------- compatibility wrappers for existing inspector ------------------
def init_camera(preview_size: Optional[tuple] = None):
    c = Camera(preview_size=preview_size or CAMERA_PREVIEW_SIZE,
               warmup_sec=CAMERA_WARMUP_SEC,
               use_background_thread=True)
    c.start()
    return c

def grab_frame(cam, timeout: float = 1.0):
    return cam.read(timeout=timeout)

def close_camera(cam):
    try:
        cam.stop()
    except Exception:
        pass

# ---------------- helpers -----------------------------------------------------
def normalize_and_save(frame_bgr: np.ndarray, filename: str,
                       canny_low: int = 50, canny_high: int = 150,
                       eps_scale: float = 0.02, min_area: int = 2000) -> Path:
    norm = normalize_card_image(frame_bgr, size=(OUT_W, OUT_H),
                                canny_low=canny_low, canny_high=canny_high,
                                eps_scale=eps_scale, min_area=min_area)
    p = Path(DEBUG_DIR) / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), norm)
    return p

def compute_phash_bgr(frame_bgr: np.ndarray) -> str:
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    return str(imagehash.phash(pil))
