#!/usr/bin/env python3
"""
Live preview tester with interactive focus controls for Raspberry Pi camera.

- Uses Picamera2 preview_array() when available, falls back to OpenCV VideoCapture.
- Uses v4l2-ctl to probe/set focus controls when available.
- Keybindings:
    q : quit
    c : capture current frame -> out.jpg
    p : probe picamera2 controls (printed to terminal)
    g : probe v4l2 controls for /dev/video0 (printed to terminal)
    a : toggle v4l2 autofocus (focus_auto)
    + : increase focus_absolute by step
    - : decrease focus_absolute by step
    0 : reset focus_absolute to baseline
    h : print help
"""
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

# optional Picamera2
try:
    from picamera2 import Picamera2
except Exception:
    Picamera2 = None

DEVICE = "/dev/video0"
OUT_FILE = "out.jpg"
PREVIEW_SIZE = (1280, 720)  # preview resolution to use
FOCUS_STEP = 10             # change step for focus_absolute
FOCUS_BASELINE = 100        # baseline focus value for reset (tune as needed)


def run_cmd(cmd: str, timeout: float = 3.0) -> subprocess.CompletedProcess:
    return subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=timeout)


def has_v4l2_ctl() -> bool:
    from shutil import which
    return which("v4l2-ctl") is not None


def v4l2_list_ctrls(device: str = DEVICE) -> Optional[str]:
    if not has_v4l2_ctl():
        print("v4l2-ctl not found on PATH; install v4l-utils")
        return None
    try:
        cp = run_cmd(f"v4l2-ctl -d {shlex.quote(device)} --list-ctrls")
        if cp.returncode != 0:
            print("v4l2-ctl error:", cp.stderr.strip())
            return None
        print(cp.stdout.strip())
        return cp.stdout
    except Exception as exc:
        print("v4l2-ctl invocation failed:", exc)
        return None


def v4l2_get(device: str, ctrl: str) -> Optional[int]:
    if not has_v4l2_ctl():
        return None
    try:
        cp = run_cmd(f"v4l2-ctl -d {shlex.quote(device)} --get-ctrl={ctrl}")
        if cp.returncode != 0:
            return None
        out = cp.stdout.strip()
        # format: ctrl_name: value
        if ":" in out:
            return int(out.split(":")[-1].strip())
        return int(out.strip())
    except Exception:
        return None


def v4l2_set(device: str, ctrl: str, val) -> bool:
    if not has_v4l2_ctl():
        print("v4l2-ctl not found; cannot set control")
        return False
    try:
        cp = run_cmd(f"v4l2-ctl -d {shlex.quote(device)} --set-ctrl={ctrl}={val}")
        if cp.returncode != 0:
            print("v4l2-ctl set error:", cp.stderr.strip())
            return False
        return True
    except Exception as exc:
        print("v4l2-ctl invocation failed:", exc)
        return False


def probe_picamera2_controls(pc2) -> None:
    if pc2 is None:
        print("Picamera2 not available")
        return
    get_controls = getattr(pc2, "get_controls", None)
    controls_attr = getattr(pc2, "controls", None)
    try:
        if callable(get_controls):
            ctrls = get_controls()
            print("Picamera2.get_controls keys:", list(ctrls.keys()))
            for k, v in ctrls.items():
                print(f"  {k} = {v!r}")
            return
        if isinstance(controls_attr, dict):
            print("Picamera2.controls keys:", list(controls_attr.keys()))
            for k, v in controls_attr.items():
                print(f"  {k} = {v!r}")
            return
    except Exception as exc:
        print("Picamera2 probe failed:", exc)
    print("No Picamera2 controls available")


def start_picamera2(preview_size=PREVIEW_SIZE, warmup=0.2):
    if Picamera2 is None:
        return None
    try:
        pc2 = Picamera2()
        cfg = pc2.create_preview_configuration({"size": preview_size})
        pc2.configure(cfg)
        pc2.start()
        time.sleep(warmup)
        return pc2
    except Exception as exc:
        print("Picamera2 start failed:", exc)
        try:
            pc2.stop()
        except Exception:
            pass
        return None


def capture_and_save(frame, name=OUT_FILE):
    if frame is None:
        print("No frame to save")
        return False
    if cv2 is None:
        print("OpenCV not available; cannot save")
        return False
    cv2.imwrite(name, frame)
    print("Saved", name)
    return True


def print_help():
    print(
        "\nPreview controls:\n"
        "  q : quit\n"
        "  c : capture current frame -> out.jpg\n"
        "  p : probe picamera2 controls (terminal)\n"
        "  g : probe v4l2 controls for /dev/video0 (terminal)\n"
        "  a : toggle v4l2 autofocus (focus_auto)\n"
        "  + : increase focus_absolute by step\n"
        "  - : decrease focus_absolute by step\n"
        "  0 : reset focus_absolute to baseline\n"
        "  h : print this help\n"
    )


def main():
    use_pc2 = False
    pc2 = None
    cap = None

    if Picamera2 is not None:
        pc2 = start_picamera2()
        if pc2 is not None:
            use_pc2 = True
            print("Using Picamera2 preview")
        else:
            print("Picamera2 not usable; falling back to OpenCV if available")

    if not use_pc2:
        if cv2 is None:
            print("OpenCV not available; cannot preview")
            return
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not cap.isOpened():
            print("OpenCV cannot open device 0")
            return
        # set preview size for OpenCV capture if supported
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_SIZE[1])
        print("Using OpenCV preview from /dev/video0")

    print_help()
    focus_auto_state = v4l2_get(DEVICE, "focus_auto")
    if focus_auto_state is not None:
        print("Initial focus_auto:", focus_auto_state)
    focus_val = v4l2_get(DEVICE, "focus_absolute")
    if focus_val is not None:
        print("Initial focus_absolute:", focus_val)
    else:
        focus_val = FOCUS_BASELINE

    window = "camera preview"
    if cv2 is not None:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = None
            if use_pc2:
                try:
                    arr = pc2.capture_array()
                    # convert RGB->BGR
                    frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if cv2 is not None else arr
                except Exception as exc:
                    print("Picamera2 capture_array failed:", exc)
                    # try to fall back to OpenCV
                    use_pc2 = False
                    try:
                        pc2.stop()
                    except Exception:
                        pass
                    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_SIZE[0])
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_SIZE[1])
            else:
                ret, frame = cap.read()
                if not ret:
                    print("OpenCV read failed")
                    break

            if frame is None:
                print("No frame available")
                break

            if cv2 is not None:
                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xFF
            else:
                # no GUI, we can't interact
                key = ord("q")

            if key == ord("q"):
                break
            if key == ord("c"):
                capture_and_save(frame, OUT_FILE)
            if key == ord("p"):
                probe_picamera2_controls(pc2)
            if key == ord("g"):
                v4l2_list_ctrls(DEVICE)
            if key == ord("a"):
                # toggle autofocus via v4l2 focus_auto if available
                fa = v4l2_get(DEVICE, "focus_auto")
                if fa is None:
                    print("focus_auto not available on device via v4l2")
                else:
                    new = 0 if fa == 1 else 1
                    ok = v4l2_set(DEVICE, "focus_auto", new)
                    if ok:
                        print("Set focus_auto ->", new)
                    else:
                        print("Failed to set focus_auto")
            if key in (ord("+"), ord("=")):
                # increase focus
                cur = v4l2_get(DEVICE, "focus_absolute") or focus_val
                new = cur + FOCUS_STEP
                if v4l2_set(DEVICE, "focus_absolute", new):
                    focus_val = new
                    print("focus_absolute ->", focus_val)
            if key == ord("-"):
                cur = v4l2_get(DEVICE, "focus_absolute") or focus_val
                new = max(0, cur - FOCUS_STEP)
                if v4l2_set(DEVICE, "focus_absolute", new):
                    focus_val = new
                    print("focus_absolute ->", focus_val)
            if key == ord("0"):
                if v4l2_set(DEVICE, "focus_absolute", FOCUS_BASELINE):
                    focus_val = FOCUS_BASELINE
                    print("Reset focus_absolute ->", focus_val)
            if key == ord("h"):
                print_help()

    finally:
        try:
            if use_pc2 and pc2 is not None:
                pc2.stop()
        except Exception:
            pass
        if cap is not None:
            cap.release()
        if cv2 is not None:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
