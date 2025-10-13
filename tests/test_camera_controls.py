#!/usr/bin/env python3
"""
Interactive camera controls tester for Raspberry Pi (Picamera2 / v4l2).

Features
- Probe Picamera2 controls (if picamera2 is installed and usable)
- Probe v4l2 controls via v4l2-ctl (if available)
- Preview live frames (uses Picamera2 capture_array or OpenCV VideoCapture)
- Capture a single frame to disk
- Interactive command loop to try set/get operations

Usage
- Run on the Pi where the camera is attached:
    python3 test_camera_controls.py

Commands in the interactive shell:
- help                       Display available commands
- probe_pc2                  Print picamera2 controls (if Picamera2 present)
- probe_v4l2 /dev/video0     List v4l2 ctrls for a device
- set_v4l2 /dev/video0 key val
- set_pc2 key val            Try pc2.set_controls({key: cast(val)})
- get_pc2                    Print pc2.get_controls() if available
- preview                    Start a short live preview (press q to exit window)
- capture out.jpg            Capture one frame to file
- exit / quit                Exit program

Notes
- v4l2-ctl must be on PATH for v4l2 commands. Example: apt install v4l-utils
- Picamera2 must be installed for Picamera2 commands; code handles absence gracefully.
- Use the probe commands first to see what keys your system exposes.
"""
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import cv2
except Exception:
    cv2 = None

# optional imports
try:
    from picamera2 import Picamera2
    from picamera2 import controls as pc2_controls  # may or may not exist
except Exception:
    Picamera2 = None
    pc2_controls = None

PROMPT = "camtest> "


def run_cmd(cmd: str, timeout: float = 5.0) -> subprocess.CompletedProcess:
    return subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=timeout)


# ---------------- v4l2 helpers ----------------
def v4l2_list_ctrls(device: str = "/dev/video0") -> Optional[str]:
    if shutil_which("v4l2-ctl") is None:
        print("v4l2-ctl not found on PATH (install v4l-utils).")
        return None
    try:
        cp = run_cmd(f"v4l2-ctl -d {shlex.quote(device)} --list-ctrls", timeout=5.0)
        if cp.returncode != 0:
            print("v4l2-ctl returned error:", cp.stderr.strip())
            return None
        print(cp.stdout.strip())
        return cp.stdout
    except Exception as exc:
        print("v4l2-ctl invocation failed:", exc)
        return None


def v4l2_set_ctrl(device: str, ctrl: str, value: Any) -> bool:
    if shutil_which("v4l2-ctl") is None:
        print("v4l2-ctl not found on PATH.")
        return False
    try:
        cmd = f"v4l2-ctl -d {shlex.quote(device)} --set-ctrl={ctrl}={value}"
        cp = run_cmd(cmd, timeout=4.0)
        if cp.returncode != 0:
            print("v4l2-ctl error:", cp.stderr.strip())
            return False
        print("v4l2-ctl OK")
        return True
    except Exception as exc:
        print("v4l2-ctl invocation failed:", exc)
        return False


# ---------------- Picamera2 helpers ----------------
def probe_picamera2_controls(pc2) -> None:
    if pc2 is None:
        print("Picamera2 not available in this environment.")
        return
    get_controls = getattr(pc2, "get_controls", None)
    controls_attr = getattr(pc2, "controls", None)
    if callable(get_controls):
        try:
            ctrls = get_controls()
            print("Picamera2.get_controls keys:", list(ctrls.keys()))
            for k, v in ctrls.items():
                print(f"  {k} = {v!r}")
            return
        except Exception as exc:
            print("pc2.get_controls() failed:", exc)
    if isinstance(controls_attr, dict):
        print("Picamera2.controls keys:", list(controls_attr.keys()))
        for k, v in controls_attr.items():
            print(f"  {k} = {v!r}")
        return
    print("No get_controls/controls attribute available on Picamera2 handle; probe returned nothing.")


def pc2_set_controls(pc2, key: str, raw_val: str) -> bool:
    if pc2 is None:
        print("Picamera2 not available.")
        return False
    # Try to cast numeric values, boolean, or leave as string
    val: Any = try_cast_value(raw_val)
    try:
        pc2.set_controls({key: val})
        print("pc2.set_controls OK")
        return True
    except Exception as exc:
        print("pc2.set_controls failed:", exc)
        return False


def pc2_get_controls(pc2) -> None:
    if pc2 is None:
        print("Picamera2 not available.")
        return
    get_controls = getattr(pc2, "get_controls", None)
    if callable(get_controls):
        try:
            ctrls = get_controls()
            for k, v in ctrls.items():
                print(f"{k} = {v!r}")
            return
        except Exception as exc:
            print("pc2.get_controls() failed:", exc)
            return
    print("No pc2.get_controls available.")


# ---------------- utility helpers ----------------
def try_cast_value(s: str) -> Any:
    # try int, float, bool, else string
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return s


def shutil_which(name: str) -> Optional[str]:
    from shutil import which

    return which(name)


# ---------------- preview / capture ----------------
def get_preview_frame_picamera2(pc2, timeout: float = 2.0):
    try:
        arr = pc2.capture_array(timeout=timeout)
        # Picamera2 returns RGB; convert to BGR for OpenCV
        if cv2 is not None:
            import numpy as _np

            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr
    except Exception as exc:
        print("Picamera2 capture_array failed:", exc)
        return None


def preview_with_picamera2(duration: float = 10.0, preview_size=(640, 360)):
    if Picamera2 is None:
        print("Picamera2 not available.")
        return
    pc2 = Picamera2()
    pc2.configure(pc2.create_preview_configuration({"size": preview_size}))
    pc2.start()
    print("Picamera2 started preview; press 'q' in window to exit early.")
    start = time.time()
    try:
        while True:
            frame = get_preview_frame_picamera2(pc2)
            if frame is None:
                print("No frame")
                break
            if cv2 is None:
                print("OpenCV not installed; cannot show preview.")
                break
            cv2.imshow("preview", frame)
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
            if (time.time() - start) > duration:
                break
    finally:
        try:
            pc2.stop()
        except Exception:
            pass
        if cv2 is not None:
            cv2.destroyAllWindows()


def preview_with_opencv(device: int = 0, duration: float = 10.0):
    if cv2 is None:
        print("OpenCV not installed; cannot preview.")
        return
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("OpenCV cannot open device", device)
        return
    print("OpenCV preview started; press 'q' to exit early.")
    start = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("frame read failed")
                break
            cv2.imshow("preview", frame)
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
            if (time.time() - start) > duration:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def capture_one_picamera2(pc2, out_path: str):
    if Picamera2 is None:
        print("Picamera2 not available.")
        return False
    try:
        arr = pc2.capture_array()
        if cv2 is None:
            try:
                from PIL import Image

                im = Image.fromarray(arr)
                im.save(out_path)
                print("Saved", out_path)
                return True
            except Exception as exc:
                print("Failed to save with PIL:", exc)
                return False
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, bgr)
        print("Saved", out_path)
        return True
    except Exception as exc:
        print("capture failed:", exc)
        return False


def capture_one_opencv(device: int, out_path: str):
    if cv2 is None:
        print("OpenCV not available.")
        return False
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("OpenCV cannot open device", device)
        return False
    try:
        ret, frame = cap.read()
        if not ret:
            print("capture read failed")
            return False
        cv2.imwrite(out_path, frame)
        print("Saved", out_path)
        return True
    finally:
        cap.release()


# ---------------- interactive loop ----------------
def interactive():
    print("Camera controls tester")
    pc2 = None
    if Picamera2 is not None:
        try:
            pc2 = Picamera2()
            pc2.configure(pc2.create_preview_configuration({"size": (640, 360)}))
            pc2.start()
            time.sleep(0.2)
            print("Picamera2 started and ready")
        except Exception as exc:
            print("Failed to start Picamera2:", exc)
            pc2 = None

    try:
        while True:
            try:
                cmd = input(PROMPT).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not cmd:
                continue
            parts = cmd.split()
            if parts[0] in ("exit", "quit"):
                break
            if parts[0] == "help":
                print(
                    "\n".join(
                        [
                            "Commands:",
                            "  help",
                            "  probe_pc2",
                            "  get_pc2",
                            "  set_pc2 KEY VALUE",
                            "  probe_v4l2 /dev/video0",
                            "  set_v4l2 /dev/video0 KEY VALUE",
                            "  preview  (tries Picamera2 preview first, falls back to OpenCV device 0)",
                            "  capture out.jpg  (captures one frame)",
                            "  quit",
                        ]
                    )
                )
                continue
            if parts[0] == "probe_pc2":
                probe_picamera2_controls(pc2)
                continue
            if parts[0] == "get_pc2":
                pc2_get_controls(pc2)
                continue
            if parts[0] == "set_pc2":
                if len(parts) < 3:
                    print("usage: set_pc2 KEY VALUE")
                    continue
                key = parts[1]
                val = " ".join(parts[2:])
                pc2_set_controls(pc2, key, val)
                continue
            if parts[0] == "probe_v4l2":
                dev = parts[1] if len(parts) > 1 else "/dev/video0"
                v4l2_list_ctrls(dev)
                continue
            if parts[0] == "set_v4l2":
                if len(parts) < 4:
                    print("usage: set_v4l2 /dev/video0 KEY VALUE")
                    continue
                dev = parts[1]
                key = parts[2]
                val = parts[3]
                v4l2_set_ctrl(dev, key, val)
                continue
            if parts[0] == "preview":
                if pc2 is not None:
                    preview_with_picamera2(duration=60.0)
                else:
                    preview_with_opencv(0, duration=60.0)
                continue
            if parts[0] == "capture":
                out = parts[1] if len(parts) > 1 else "capture.jpg"
                if pc2 is not None:
                    capture_one_picamera2(pc2, out)
                else:
                    capture_one_opencv(0, out)
                continue
            print("Unknown command. Type help for list.")
    finally:
        if pc2 is not None:
            try:
                pc2.stop()
            except Exception:
                pass
        print("Exiting interactive tester.")


if __name__ == "__main__":
    interactive()
