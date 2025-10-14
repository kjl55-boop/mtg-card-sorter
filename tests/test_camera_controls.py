#!/usr/bin/env python3
"""
Live preview tester with interactive focus controls, Picamera2 tuning, multishot sharpness
selection, and an interactive manual focus sweep.

Keybindings (preview window):
  q : quit
  c : capture current frame -> out.jpg
  m : single multishot burst -> best.jpg
  M : continuous multishot bursts until stopped
  ] : increase burst count (N)
  [ : decrease burst count (N)
  > : increase inter-frame delay (seconds)
  < : decrease inter-frame delay (seconds)
  p : probe picamera2 controls (terminal)
  g : probe v4l2 controls for /dev/video0 (terminal)
  a : toggle v4l2 autofocus (focus_auto) if supported
  + / - / 0 : v4l2 focus_absolute tweaks
  t : apply Picamera2 tuned preview controls (best-effort)
  r : revert Picamera2 controls to neutral (best-effort)
  s : interactive manual focus sweep (rotate lens between positions, press Enter)
  h : print help
"""
from pathlib import Path
import shlex
import subprocess
import sys
import time
import csv

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

# Config
DEVICE = "/dev/video0"
OUT_FILE = "out.jpg"
BEST_FILE = "best.jpg"
PREVIEW_SIZE = (1280, 720)   # preview resolution
FOCUS_STEP = 10
FOCUS_BASELINE = 100

# Multishot defaults (adjustable)
MULTISHOT_N = 8
MULTISHOT_DELAY = 0.05  # seconds between frames

# Sweep defaults
SWEEP_POSITIONS = 8
SWEEP_SHOTS_PER_POS = 12
SWEEP_DELAY = 0.05

# ---------------------------------------------------------------------------
# Helpers: shell / v4l2 / picamera2 probes and sets
# ---------------------------------------------------------------------------

def run_cmd(cmd: str, timeout: float = 3.0) -> subprocess.CompletedProcess:
    return subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=timeout)


def has_v4l2_ctl() -> bool:
    from shutil import which
    return which("v4l2-ctl") is not None


def v4l2_list_ctrls(device: str = DEVICE):
    if not has_v4l2_ctl():
        print("v4l2-ctl not found on PATH; install v4l-utils")
        return
    try:
        cp = run_cmd(f"v4l2-ctl -d {shlex.quote(device)} --list-ctrls")
        if cp.returncode != 0:
            print("v4l2-ctl error:", cp.stderr.strip())
            return
        print(cp.stdout.strip())
    except Exception as exc:
        print("v4l2-ctl invocation failed:", exc)


def v4l2_get(device: str, ctrl: str):
    if not has_v4l2_ctl():
        return None
    try:
        cp = run_cmd(f"v4l2-ctl -d {shlex.quote(device)} --get-ctrl={ctrl}")
        if cp.returncode != 0:
            return None
        out = cp.stdout.strip()
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


def probe_picamera2_controls(pc2):
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


# ---------------------------------------------------------------------------
# Picamera2 tuning functions (best-effort, safe)
# ---------------------------------------------------------------------------

def apply_picamera2_tuned_controls(pc2):
    """Best-effort: set tuned preview controls on Picamera2 instance."""
    if pc2 is None:
        print("Picamera2 not available; cannot apply tuned controls")
        return False
    try:
        from picamera2 import controls as pc2_controls
    except Exception:
        pc2_controls = None

    # Try to reconfigure preview to ensure proper pipeline; ignore failures
    try:
        cfg = pc2.create_preview_configuration({"size": PREVIEW_SIZE})
        pc2.configure(cfg)
        try:
            pc2.start()
        except Exception:
            pass
    except Exception:
        pass

    ctrl_payload = {}

    # AF mode
    try:
        if pc2_controls and hasattr(pc2_controls, "AfModeEnum"):
            enum = pc2_controls.AfModeEnum
            if hasattr(enum, "Continuous"):
                ctrl_payload["AfMode"] = enum.Continuous
            elif hasattr(enum, "Auto"):
                ctrl_payload["AfMode"] = enum.Auto
    except Exception as exc:
        print("AF enum handling failed:", exc)

    # AWB
    try:
        if pc2_controls and hasattr(pc2_controls, "AwbModeEnum") and hasattr(pc2_controls.AwbModeEnum, "Auto"):
            ctrl_payload["AwbMode"] = pc2_controls.AwbModeEnum.Auto
        else:
            ctrl_payload["AwbEnable"] = True
    except Exception as exc:
        print("AWB control handling failed:", exc)

    # Noise reduction - try draft namespace if present
    try:
        if pc2_controls and hasattr(pc2_controls, "draft") and hasattr(pc2_controls.draft, "NoiseReductionModeEnum"):
            nr_enum = pc2_controls.draft.NoiseReductionModeEnum
            if hasattr(nr_enum, "HighQuality"):
                ctrl_payload["NoiseReductionMode"] = nr_enum.HighQuality
    except Exception as exc:
        print("Noise reduction handling failed:", exc)

    # Image tuning numeric controls
    # Values chosen to mirror your example
    for k, v in (("Sharpness", 2.0), ("Contrast", 1.5), ("Saturation", 1.5)):
        try:
            ctrl_payload[k] = v
        except Exception:
            pass

    if not ctrl_payload:
        print("No Picamera2 controls available to set")
        return False

    try:
        pc2.set_controls(ctrl_payload)
        print("Applied Picamera2 tuned controls:", ", ".join(ctrl_payload.keys()))
        return True
    except Exception as exc:
        print("pc2.set_controls failed:", exc)
        return False


def revert_picamera2_controls(pc2):
    """Best-effort: attempt to revert controls to safe defaults."""
    if pc2 is None:
        print("Picamera2 not available; cannot revert controls")
        return False
    try:
        from picamera2 import controls as pc2_controls
    except Exception:
        pc2_controls = None

    payload = {}
    # Try to set AF back to continuous/auto if available
    try:
        if pc2_controls and hasattr(pc2_controls, "AfModeEnum"):
            enum = pc2_controls.AfModeEnum
            if hasattr(enum, "Continuous"):
                payload["AfMode"] = enum.Continuous
            elif hasattr(enum, "Auto"):
                payload["AfMode"] = enum.Auto
    except Exception:
        pass

    # Re-enable AWB
    try:
        if pc2_controls and hasattr(pc2_controls, "AwbModeEnum") and hasattr(pc2_controls.AwbModeEnum, "Auto"):
            payload["AwbMode"] = pc2_controls.AwbModeEnum.Auto
        else:
            payload["AwbEnable"] = True
    except Exception:
        pass

    # Neutral numeric tuning
    for k, v in (("Sharpness", 0.0), ("Contrast", 1.0), ("Saturation", 1.0)):
        payload[k] = v

    try:
        pc2.set_controls(payload)
        print("Reverted Picamera2 controls to defaults")
        return True
    except Exception as exc:
        print("Failed to revert controls:", exc)
        return False


# ---------------------------------------------------------------------------
# Capture, scoring, multishot
# ---------------------------------------------------------------------------

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


def sharpness_score(img):
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def do_multishot_capture_from_pc2(pc2, n, delay):
    frames = []
    for i in range(n):
        try:
            arr = pc2.capture_array()
            frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            print("pc2 capture failed:", exc)
            return frames
        frames.append(frame)
        time.sleep(delay)
    return frames


def do_multishot_capture_from_cv2(cap, n, delay):
    frames = []
    for i in range(n):
        ret, frame = cap.read()
        if not ret:
            print("OpenCV read failed at frame", i)
            break
        frames.append(frame)
        time.sleep(delay)
    return frames


def run_multishot(pc2, cap, n, delay, out_path=BEST_FILE):
    print(f"Multishot: N={n} delay={delay:.3f}s")
    frames = None
    if pc2 is not None:
        frames = do_multishot_capture_from_pc2(pc2, n, delay)
        if not frames:
            print("Picamera2 multishot capture returned no frames; falling back to OpenCV")
            frames = None
    if frames is None:
        if cap is None:
            print("No capture backend available for multishot")
            return False
        frames = do_multishot_capture_from_cv2(cap, n, delay)
    if not frames:
        print("No frames captured in multishot")
        return False
    scores = [sharpness_score(f) for f in frames]
    for i, s in enumerate(scores):
        print(f"  frame {i:02d}: score {s:.2f}")
    best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
    print("Best frame:", best_idx, "score:", scores[best_idx])
    cv2.imwrite(out_path, frames[best_idx])
    print("Saved best frame to", out_path)
    return True


# ---------------------------------------------------------------------------
# Interactive manual focus sweep
# ---------------------------------------------------------------------------

def focus_sweep_interactive(pc2, cap, shots_per_position=SWEEP_SHOTS_PER_POS, delay=SWEEP_DELAY, positions=SWEEP_POSITIONS, out_dir="focus_sweep"):
    """
    Interactive manual sweep:
    - User manually rotates lens between positions.
    - Press Enter to capture a multishot burst at current lens setting.
    - Type 'q' then Enter to quit early.
    - Saves best image per position and summary CSV.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    summary = []
    print("FOCUS SWEEP MODE")
    print("  Instructions:")
    print("   - For each mechanical lens position: rotate lens, place card, press Enter to capture.")
    print("   - Type 'q' + Enter to quit early.")
    print()
    pos = 0
    while pos < positions:
        cmd = input(f"Position {pos+1}/{positions}: rotate lens to next pos, then press Enter (or 'q' to quit): ")
        if cmd.strip().lower() == "q":
            break
        pos += 1
        print(f"Capturing {shots_per_position} frames at position {pos} ...")
        # capture frames
        frames = []
        if pc2 is not None:
            for i in range(shots_per_position):
                try:
                    arr = pc2.capture_array()
                    frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                except Exception as exc:
                    print("pc2 capture failed:", exc)
                    break
                frames.append(frame)
                time.sleep(delay)
        else:
            for i in range(shots_per_position):
                ret, frame = cap.read()
                if not ret:
                    print("cap read failed")
                    break
                frames.append(frame)
                time.sleep(delay)
        if not frames:
            print("No frames captured for this position, skipping")
            continue
        # score and pick best
        scores = [sharpness_score(f) for f in frames]
        best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        best_frame = frames[best_idx]
        fname = f"pos{pos:02d}_best.jpg"
        fpath = Path(out_dir) / fname
        cv2.imwrite(str(fpath), best_frame)
        print(f"Saved best frame for pos {pos}: {fpath}  score={scores[best_idx]:.2f}")
        summary.append((pos, scores[best_idx], str(fpath)))
    # write CSV
    csv_path = Path(out_dir) / "summary.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["position", "best_score", "best_image"])
        for r in summary:
            writer.writerow(r)
    print("Sweep complete. Summary written to", csv_path)
    return summary


# ---------------------------------------------------------------------------
# UI / main loop
# ---------------------------------------------------------------------------

def print_help():
    print(
        "\nPreview controls:\n"
        "  q : quit\n"
        "  c : capture current frame -> out.jpg\n"
        "  m : single multishot burst -> best.jpg\n"
        "  M : continuous multishot bursts until stopped\n"
        "  ] : increase burst count (N)\n"
        "  [ : decrease burst count (N)\n"
        "  > : increase inter-frame delay (seconds)\n"
        "  < : decrease inter-frame delay (seconds)\n"
        "  p : probe picamera2 controls (terminal)\n"
        "  g : probe v4l2 controls for /dev/video0 (terminal)\n"
        "  a : toggle v4l2 autofocus (focus_auto)\n"
        "  + / - / 0 : v4l2 focus_absolute tweaks\n"
        "  t : apply Picamera2 tuned controls\n"
        "  r : revert Picamera2 controls\n"
        "  s : start interactive manual focus sweep\n"
        "  h : print this help\n"
    )


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


def main():
    global MULTISHOT_N, MULTISHOT_DELAY
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

    continuous_multishot = False

    try:
        while True:
            frame = None
            if use_pc2:
                try:
                    arr = pc2.capture_array()
                    frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                except Exception as exc:
                    print("Picamera2 capture_array failed:", exc)
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
                # overlay status
                status = f"N={MULTISHOT_N} delay={MULTISHOT_DELAY:.3f}s"
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xFF
            else:
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
                fa = v4l2_get(DEVICE, "focus_auto")
                if fa is None:
                    print("focus_auto not available on device via v4l2")
                else:
                    new = 0 if fa == 1 else 1
                    ok = v4l2_set(DEVICE, "focus_auto", new)
                    print("Set focus_auto ->", new if ok else "FAILED")
            if key in (ord("+"), ord("=")):
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
            if key == ord("]"):
                MULTISHOT_N = max(1, MULTISHOT_N + 1)
                print("MULTISHOT_N ->", MULTISHOT_N)
            if key == ord("["):
                MULTISHOT_N = max(1, MULTISHOT_N - 1)
                print("MULTISHOT_N ->", MULTISHOT_N)
            if key == ord(">"):
                MULTISHOT_DELAY = MULTISHOT_DELAY + 0.01
                print("MULTISHOT_DELAY ->", MULTISHOT_DELAY)
            if key == ord("<"):
                MULTISHOT_DELAY = max(0.0, MULTISHOT_DELAY - 0.01)
                print("MULTISHOT_DELAY ->", MULTISHOT_DELAY)
            if key == ord("m"):
                run_multishot(pc2 if use_pc2 else None, cap, MULTISHOT_N, MULTISHOT_DELAY, BEST_FILE)
            if key == ord("M"):
                continuous_multishot = not continuous_multishot
                print("Continuous multishot ->", continuous_multishot)
            if continuous_multishot:
                run_multishot(pc2 if use_pc2 else None, cap, MULTISHOT_N, MULTISHOT_DELAY, BEST_FILE)
                time.sleep(0.1)
            if key == ord("t"):
                ok = apply_picamera2_tuned_controls(pc2 if use_pc2 else None)
                print("Tuned controls applied?" , ok)
            if key == ord("r"):
                ok = revert_picamera2_controls(pc2 if use_pc2 else None)
                print("Revert controls applied?", ok)
            if key == ord("s"):
                print("Starting interactive focus sweep. This will pause preview until sweep completes.")
                focus_sweep_interactive(pc2 if use_pc2 else None, cap, shots_per_position=MULTISHOT_N, delay=MULTISHOT_DELAY, positions=SWEEP_POSITIONS, out_dir="focus_sweep")
                print("Returned from focus sweep; resuming preview.")
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
