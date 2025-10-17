# pipeline/camera/core.py

import cv2
import threading
import time
import tempfile
import subprocess
import os
import shutil
from pathlib import Path
from typing import Optional
import numpy as np

from config.config import CONFIG
from pipeline import utils

log = utils.get_logger("camera")

class CameraError(RuntimeError):
    pass

class Camera:
    """
    Persistent camera abstraction with backend failover.
    Supports threaded capture and synchronous fallback.
    Designed for use with phash-based recognition pipelines.
    """
    def __init__(self, preview_size=CONFIG.preview_size, warmup_sec=CONFIG.camera_timeout, use_background_thread=True, cam_index=0):
        self.preview_size = preview_size
        self.warmup_sec = warmup_sec
        self.cam_index = cam_index
        self.use_background_thread = use_background_thread
        self._handle = None
        self.backend_name = None
        self._running = False
        self._thread = None
        self._frame = None
        self._lock = threading.Lock()

    def _open_picamera2(self):
        try:
            from picamera2 import Picamera2
            from libcamera import controls
        except Exception:
            return None
        try:
            pc2 = Picamera2()
            cfg = pc2.create_preview_configuration({"size": self.preview_size})
            pc2.configure(cfg)
            pc2.start()
            pc2.set_controls({
                "AfMode": controls.AfModeEnum.Manual,
                "AfMetering": controls.AfMeteringEnum.Auto
            })
            pc2.autofocus_cycle()
            time.sleep(self.warmup_sec)
            self.backend_name = "picamera2"
            self._handle = pc2
            return pc2
        except Exception as exc:
            utils.log_exception(log, exc, "Failed to initialize Picamera2")
            return None

    def _open_libcamera_jpeg(self):
        if not shutil.which("libcamera-jpeg"):
            return None
        self.backend_name = "libcamera-jpeg"
        return "libcamera-jpeg"

    def _open_opencv(self):
        try:
            cap = cv2.VideoCapture(self.cam_index, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                return None
            for _ in range(2):
                cap.read()
            self.backend_name = "opencv"
            self._handle = cap
            return cap
        except Exception as exc:
            utils.log_exception(log, exc, "VideoCapture initialization failed")
            return None

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
                return frame if ret else None
            return None
        except Exception as exc:
            utils.log_exception(log, exc, "Exception during single read")
            return None

    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        if not self._running:
            raise CameraError("Camera not started")
        if self.use_background_thread and self._thread is not None:
            start = time.time()
            while True:
                with self._lock:
                    frame = None if self._frame is None else self._frame.copy()
                if frame is not None:
                    return frame
                if timeout and (time.time() - start) >= timeout:
                    return None
                time.sleep(0.005)
        elif self.backend_name == "libcamera-jpeg":
            tmp = Path(tempfile.gettempdir()) / f"libcam_capture_{os.getpid()}.jpg"
            cmd = ["libcamera-jpeg", "-o", str(tmp), "-n", "--timeout", str(int((timeout or 2.0) * 1000))]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=max(5, (timeout or 2.0) + 3))
                img = cv2.imread(str(tmp))
            except Exception as exc:
                utils.log_exception(log, exc, "libcamera-jpeg capture failed")
                img = None
            finally:
                try: tmp.unlink()
                except Exception: pass
            return img
        else:
            return self._read_once()

    def is_open(self) -> bool:
        return self._running

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        try:
            if self.backend_name == "picamera2" and self._handle:
                self._handle.stop()
            if self.backend_name == "opencv" and self._handle:
                self._handle.release()
        except Exception as exc:
            utils.log_exception(log, exc, "Exception while stopping camera")
        self._handle = None
        self.backend_name = None

    @property
    def backend(self) -> Optional[str]:
        return self.backend_name

