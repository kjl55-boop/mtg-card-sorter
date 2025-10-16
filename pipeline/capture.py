"""
Persistent Camera abstraction with backend failover plus helper utilities.

This simplified capture module intentionally omits any programmatic autofocus
or focus-locking logic so the camera will use its built-in/default focus and
white-balance behavior (sensor IMX708 or driver defaults).

Public API:
- Camera(...) : class for persistent usage
- init_camera(preview_size=None) -> Camera
- grab_frame(cam, timeout) -> np.ndarray
- close_camera(cam)
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
from typing import Optional, Tuple

import cv2
from PIL import Image
import imagehash
import numpy as np

from . import utils
log = utils.get_logger("camera")

from recognizer.crop import crop_card_from_box, find_card_contour
from config.config import CONFIG

# ensure debug dir exists
Path(CONFIG.debug_dir).mkdir(parents=True, exist_ok=True)
OUT_W, OUT_H = CONFIG.normalized_size


class CameraError(RuntimeError):
    pass


class Camera:
    """
    Persistent camera abstraction.
    Use start()/read()/stop() for lifecycle-managed capture.
    Optional background thread keeps latest frame for low-latency reads.

    Note: This version deliberately avoids any software focus control. The
    device (sensor/driver) default autofocus/white balance behavior remains in use.
    """

    def __init__(
        self,
        preview_size=CONFIG.preview_size,
        warmup_sec=CONFIG.camera_timeout,
        use_background_thread: bool = True,
        cam_index: int = 0,
    ):
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
            log.debug("picamera2 import failed or not installed")
            return None
        try:
            pc2 = Picamera2()
            cfg = pc2.create_preview_configuration({"size": self.preview_size})
            pc2.configure(cfg)
            pc2.start()
            self._run_autofocus(pc2)
            time.sleep(self.warmup_sec)
            self.backend_name = "picamera2"
            self._handle = pc2
            log.info("Camera backend selected: picamera2")
            return pc2
        except Exception as exc:
            utils.log_exception(log, exc, "Failed to initialize Picamera2")
            return None

    def _run_autofocus(self, pc2):
        try:
            from libcamera import controls
            pc2.set_controls({
                "AfMode": controls.AfModeEnum.Manual,
                "AfMetering": controls.AfMeteringEnum.Auto
            })
            log.info("Running autofocus cycle at startup...")
            success = pc2.autofocus_cycle()
            log.info("Autofocus successful." if success else "Autofocus failed.")
        except Exception as exc:
            utils.log_exception(log, exc, "Autofocus cycle failed")

    def _open_libcamera_jpeg(self):
        if not shutil.which("libcamera-jpeg"):
            log.debug("libcamera-jpeg not available on PATH")
            return None
        self.backend_name = "libcamera-jpeg"
        log.info("Camera backend available: libcamera-jpeg (synchronous)")
        return "libcamera-jpeg"

    def _open_opencv(self):
        try:
            cap = cv2.VideoCapture(self.cam_index, cv2.CAP_V4L2)
        except Exception as exc:
            utils.log_exception(log, exc, "VideoCapture initialization failed")
            return None
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            log.debug("OpenCV VideoCapture not opened for index %s", self.cam_index)
            return None
        # warm-up reads
        for _ in range(2):
            cap.read()
        self.backend_name = "opencv"
        self._handle = cap
        log.info("Camera backend selected: opencv (V4L2)")
        return cap

    # ---------------- lifecycle ------------------------------------------
    def start(self):
        if self._running:
            return
        self._handle = self._open_picamera2() or self._open_libcamera_jpeg() or self._open_opencv()
        if self._handle is None:
            log.error("No capture backend available")
            raise CameraError("No capture backend available")
        self._running = True
        log.debug("Camera started with backend=%s", self.backend_name)
        if self.use_background_thread and self.backend_name != "libcamera-jpeg":
            self._thread = threading.Thread(target=self._bg_loop, daemon=True)
            self._thread.start()
            log.debug("Background read thread started")

    def autofocus(self):
        if self.backend_name == "picamera2" and self._handle is not None:
            self._run_autofocus(self._handle)

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
                # picamera2 returns RGB arrays; convert to BGR for OpenCV compat
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            if self.backend_name == "opencv":
                ret, frame = self._handle.read()
                if not ret:
                    log.debug("OpenCV read returned no frame")
                    return None
                return frame
            # libcamera-jpeg handled synchronously in read()
            return None
        except Exception as exc:
            utils.log_exception(log, exc, "Exception during single read")
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
                    return frame
                if timeout is not None and (time.time() - start) >= timeout:
                    log.debug("read timed out waiting for frame")
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
                cmd = ["libcamera-jpeg", "-o", str(tmp), "-n", "--timeout", str(int((timeout or 2.0) * 1000))]
                try:
                    subprocess.run(
                        cmd,
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=max(5, (timeout or 2.0) + 3),
                    )
                    img = cv2.imread(str(tmp))
                except Exception as exc:
                    utils.log_exception(log, exc, "libcamera-jpeg capture failed")
                    img = None
                finally:
                    try:
                        tmp.unlink()
                    except Exception:
                        pass
                if img is None:
                    log.debug("libcamera-jpeg returned no image")
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
            log.debug("Background thread stopped")
        try:
            if self.backend_name == "picamera2" and self._handle is not None:
                try:
                    self._handle.stop()
                except Exception:
                    pass
            if self.backend_name == "opencv" and self._handle is not None:
                try:
                    self._handle.release()
                except Exception:
                    pass
            log.info("Camera backend stopped: %s", self.backend_name)
        except Exception as exc:
            utils.log_exception(log, exc, "Exception while stopping camera")
        self._handle = None
        self.backend_name = None


# ---------------- convenience single-shot API and module helpers ----------------

def capture_frame(cam_index: int = 0, timeout: float = 2.0) -> np.ndarray:
    """
    Single-shot convenience which tries backends in order. Raises RuntimeError on failure.
    """
    cam = Camera(preview_size=CONFIG.camera_preview_size, warmup_sec=CONFIG.camera_warmup_sec, use_background_thread=False, cam_index=cam_index)
    cam.start()
    try:
        img = cam.read(timeout=timeout)
        if img is None:
            log.error("capture_frame: no frame captured")
            raise RuntimeError("capture_frame: no frame captured")
        return img
    finally:
        cam.stop()


def init_camera(preview_size: Optional[tuple] = None) -> Camera:
    """
    Create, start, and return a Camera instance for long-running use.
    """
    c = Camera(preview_size=preview_size or CONFIG.camera_preview_size, warmup_sec=CONFIG.camera_warmup_sec, use_background_thread=True)
    c.start()
    return c


def grab_frame(cam: Camera, timeout: float = 1.0) -> Optional[np.ndarray]:
    """
    Read a frame from a running Camera instance. Returns None on timeout/errors.
    """
    try:
        return cam.read(timeout=timeout)
    except Exception as exc:
        utils.log_exception(log, exc, "grab_frame failed")
        return None


def close_camera(cam: Camera) -> None:
    """
    Stop and cleanup a Camera instance.
    """
    try:
        cam.stop()
    except Exception as exc:
        utils.log_exception(log, exc, "close_camera failed")


# ---------------- helpers -----------------------------------------------------
def normalize_and_save(
    frame_bgr: np.ndarray, filename: str, pad_x_pct: float = 0.02, pad_y_pct: float = 0.02, min_area: int = 2000
) -> Path:
    """
    Find the card in frame, deskew/crop using crop_card_from_box, resize to NORMALIZED_SIZE, and save to DEBUG_DIR.
    Returns the saved Path or raises RuntimeError on failure.
    """
    if frame_bgr is None:
        raise ValueError("normalize_and_save: input frame is None")

    box, _ = find_card_contour(frame_bgr, min_area=min_area)
    if box is None:
        log.warning("normalize_and_save: no card contour found")
        raise RuntimeError("normalize_and_save: no card contour found")

    card = crop_card_from_box(frame_bgr, box, pad_x_pct=pad_x_pct, pad_y_pct=pad_y_pct)
    if card is None or card.size == 0:
        log.warning("normalize_and_save: crop failed")
        raise RuntimeError("normalize_and_save: crop failed")

    norm = cv2.resize(card, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    p = Path(CONFIG.debug_dir) / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    written = utils.safe_imwrite(str(p), norm)
    if not written:
        log.warning("Failed to write normalized image to %s", p)
        raise RuntimeError(f"Failed to write normalized image to {p}")
    log.info("Saved normalized image to %s", p)
    return p


def compute_phash_bgr(frame_bgr: np.ndarray) -> str:
    """
    Compute perceptual hash (imagehash.phash) from an OpenCV BGR image and return hex string.
    """
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    return str(imagehash.phash(pil))
