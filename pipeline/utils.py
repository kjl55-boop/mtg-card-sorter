# app/utils.py
"""
Small, dependency-light helpers: filesystem, image save, logging helpers.

Call configure_logging() from application entrypoint before other modules that log.
No side-effects at import time.
"""

from pathlib import Path
import logging
import time
import uuid
import pickle
from typing import Any, Optional
import cv2
import numpy as np

# Package logger name
PKG_LOGGER = "card_inspector"

def configure_logging(level: Optional[int] = None, logfile: Optional[str] = None):
    """
    Configure root package logger. Call once at application startup.
    If level is None the function will default to logging.INFO.
    """
    import logging

    # default to INFO when caller passes None
    if level is None:
        level = logging.INFO

    # accept string levels like "DEBUG"
    if isinstance(level, str):
        level = logging._nameToLevel.get(level.upper(), logging.INFO)

    logger = logging.getLogger(PKG_LOGGER)
    logger.setLevel(level)
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if logfile:
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a child logger for the package, e.g., get_logger('matcher')."""
    return logging.getLogger(PKG_LOGGER + (f".{name}" if name else ""))

def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _timestamped_name(prefix: str, tag: str = "", ext: str = ".png") -> str:
    ts = int(time.time())
    uid = uuid.uuid4().hex[:8]
    tag_part = f"_{tag}" if tag else ""
    return f"{ts}_{uid}{tag_part}{ext}"

def safe_imwrite(path: str, img: np.ndarray) -> bool:
    """
    Write image to disk, ensuring parent directory exists.
    Returns True on success, False on failure.
    """
    try:
        p = Path(path)
        ensure_dir(p.parent)
        # cv2.imwrite returns boolean success in many builds; check it
        ok = cv2.imwrite(str(p), img)
        return bool(ok)
    except Exception:
        get_logger("utils").exception("safe_imwrite failed for %s", path)
        return False

def save_debug_image(img: np.ndarray, tag: str = "debug", directory: Optional[Path] = None) -> Optional[str]:
    """
    Save image to directory with timestamped filename and return path string or None on failure.
    """
    try:
        if directory is None:
            # avoid importing app.__init__ here; caller should pass DEBUG_DIR from app package
            raise RuntimeError("save_debug_image requires a directory argument")
        directory = ensure_dir(directory)
        fname = _timestamped_name(prefix="dbg", tag=tag, ext=".png")
        path = directory / fname
        ok = safe_imwrite(str(path), img)
        return str(path) if ok else None
    except Exception:
        get_logger("utils").exception("save_debug_image failed")
        return None

def log_exception(logger: Optional[logging.Logger], exc: Exception, message: Optional[str] = None):
    """
    Convenience to log an exception with context. Use get_logger(...) to obtain logger.
    """
    log = logger if logger is not None else get_logger("utils")
    if message:
        log.exception(message)
    else:
        log.exception("Unhandled exception: %s", exc)

# atomic pickle helpers
def safe_pickle_save(path: Path, obj: Any) -> bool:
    try:
        ensure_dir(path.parent)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        return True
    except Exception:
        get_logger("utils").exception("safe_pickle_save failed for %s", path)
        return False

def safe_pickle_load(path: Path) -> Any:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        get_logger("utils").exception("safe_pickle_load failed for %s", path)
        return None

# Simple image conversion helpers
def pil_to_bgr(pil_img) -> np.ndarray:
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def bgr_to_pil(bgr_img):
    from PIL import Image
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))
