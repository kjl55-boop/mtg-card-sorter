#!/usr/bin/env python3
"""
Live preview tester with interactive focus controls, Picamera2 tuning, multishot sharpness
selection, interactive manual focus sweep, rpicam-like setting application, and high-res capture.

Keybindings (preview window):
  q : quit
  c : capture current frame -> out.jpg
  z : single rpicam-still capture (if available)
  Z : multishot using rpicam-still (if available)
  k : capture a high-resolution still using Picamera2 (fallback to rpicam-still)
  m : single multishot burst -> best.jpg
  M : continuous multishot bursts until stopped
  ] : increase burst count (N)
  [ : decrease burst count (N)
  > : increase inter-frame delay (seconds)
  < : decrease inter-frame delay (seconds)
  p : probe picamera2 controls (terminal)
  g : probe v4l2 controls for /dev/video0 (terminal)
  a : toggle v4l2 autofocus (focus_auto)
  + / - / 0 : v4l2 focus_absolute tweaks
  t : apply Picamera2 tuned controls (best-effort)
  y : apply rpicam-still like settings (still-size, exposure/gain hints, AF/AWB/NR/tonal)
  r : revert all tuned/rpicam-like Picamera2 settings to neutral
  s : start interactive manual focus sweep
  h : print help
  f : triggers autofocus using AfMode=Manual, AfMetering=Window, and a centered AfWindow region.
"""
from pathlib import Path
import shlex
import subprocess
import sys
import time
import csv
import shutil
import datetime

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
PREVIEW_SIZE = (1280, 720)   # preview resolution for live preview
FOCUS_STEP = 10
FOCUS_BASELINE = 100

# Multishot defaults (adjustable)
MULTISHOT_N = 8
MULTISHOT_DELAY = 0.05  # seconds between frames

# Sweep defaults
SWEEP_POSITIONS = 8
SWEEP_SHOTS_PER_POS = 12
SWEEP_DELAY = 0.05

# rpicam defaults to mimic
RPICAM_STILL_SIZE = (4056, 3040)  # conservative full-sensor still resolution
RPICAM_STILL_TIMEOUT_MS = 1000

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
    """Best-effort: set tuned preview controls on Picamera2 instance (your earlier example)."""
    if pc2 is None:
        print("Picamera2 not available; cannot apply tuned controls")
        return False
    try:
        from picamera2 import controls as pc2_controls
    except Exception:
        pc2_controls = None

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

    try:
        if pc2_controls and hasattr(pc2_controls, "AfModeEnum"):
            enum = pc2_controls.AfModeEnum
            if hasattr(enum, "Continuous"):
                ctrl_payload["AfMode"] = enum.Continuous
            elif hasattr(enum, "Auto"):
                ctrl_payload["AfMode"] = enum.Auto
    except Exception as exc:
        print("AF enum handling failed:", exc)

    try:
        if pc2_controls and hasattr(pc2_controls, "AwbModeEnum") and hasattr(pc2_controls.AwbModeEnum, "Auto"):
            ctrl_payload["AwbMode"] = pc2_controls.AwbModeEnum.Auto
        else:
            ctrl_payload["AwbEnable"] = True
    except Exception as exc:
        print("AWB control handling failed:", exc)

    try:
        if pc2_controls and hasattr(pc2_controls, "draft") and hasattr(pc2_controls.draft, "NoiseReductionModeEnum"):
            nr_enum = pc2_controls.draft.NoiseReductionModeEnum
            if hasattr(nr_enum, "HighQuality"):
                ctrl_payload["NoiseReductionMode"] = nr_enum.HighQuality
    except Exception as exc:
        print("Noise reduction handling failed:", exc)

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
    """Best-effort: attempt to revert Picamera2 tuned controls to safe defaults."""
    if pc2 is None:
        print("Picamera2 not available; cannot revert controls")
        return False
    try:
        from picamera2 import controls as pc2_controls
    except Exception:
        pc2_controls = None

    payload = {}
    try:
        if pc2_controls and hasattr(pc2_controls, "AfModeEnum"):
            enum = pc2_controls.AfModeEnum
            if hasattr(enum, "Continuous"):
                payload["AfMode"] = enum.Continuous
            elif hasattr(enum, "Auto"):
                payload["AfMode"] = enum.Auto
    except Exception:
        pass

    try:
        if pc2_controls and hasattr(pc2_controls, "AwbModeEnum") and hasattr(pc2_controls.AwbModeEnum, "Auto"):
            payload["AwbMode"] = pc2_controls.AwbModeEnum.Auto
        else:
            payload["AwbEnable"] = True
    except Exception:
        pass

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
# rpicam-like settings (apply/revert) to mirror rpicam-still pipeline behavior
# ---------------------------------------------------------------------------

def apply_rpicam_like_settings(pc2,
                              still_size=RPICAM_STILL_SIZE,
                              shutter_us: int | None = None,
                              gain: float | None = None,
                              af_mode_prefer="Continuous"):
    """
    Best-effort: apply settings intended to match rpicam-still output.
    - still_size: target still/preview resolution to configure
    - shutter_us: microseconds for shutter (None leaves AE auto)
    - gain: numeric gain/ISO hint (None leaves AGC)
    - af_mode_prefer: "Continuous" or "Auto"
    Returns True if at least one control appears to have been applied.
    """
    if pc2 is None:
        print("Picamera2 not available")
        return False

    applied = {}
    try:
        # attempt to reconfigure preview/still pipeline similar to rpicam-still
        try:
            cfg = pc2.create_preview_configuration({"size": still_size})
            pc2.configure(cfg)
            try:
                pc2.start()
            except Exception:
                pass
        except Exception:
            pass

        try:
            from picamera2 import controls as pc2_controls
        except Exception:
            pc2_controls = None

        ctrl_payload = {}

        # Exposure / shutter controls mapping
        if shutter_us is not None:
            for k in ("Shutter", "ExposureTime", "ExposureTimeUs", "SensorExposureTime"):
                try:
                    ctrl_payload[k] = int(shutter_us)
                    applied[k] = shutter_us
                    break
                except Exception:
                    pass

        # Gain mapping
        if gain is not None:
            for k in ("AnalogueGain", "Gain", "ExposureValue", "ISO"):
                try:
                    ctrl_payload[k] = float(gain)
                    applied[k] = gain
                    break
                except Exception:
                    pass

        # AF mode
        try:
            if pc2_controls and hasattr(pc2_controls, "AfModeEnum"):
                enum = pc2_controls.AfModeEnum
                if hasattr(enum, af_mode_prefer):
                    ctrl_payload["AfMode"] = getattr(enum, af_mode_prefer)
                    applied["AfMode"] = af_mode_prefer
                elif hasattr(enum, "Continuous"):
                    ctrl_payload["AfMode"] = enum.Continuous
                    applied["AfMode"] = "Continuous"
        except Exception:
            pass

        # AWB
        try:
            if pc2_controls and hasattr(pc2_controls, "AwbModeEnum") and hasattr(pc2_controls.AwbModeEnum, "Auto"):
                ctrl_payload["AwbMode"] = pc2_controls.AwbModeEnum.Auto
                applied["AwbMode"] = "Auto"
            else:
                ctrl_payload["AwbEnable"] = True
                applied["AwbEnable"] = True
        except Exception:
            pass

        # Noise reduction (draft)
        try:
            if pc2_controls and hasattr(pc2_controls, "draft") and hasattr(pc2_controls.draft, "NoiseReductionModeEnum"):
                nr_enum = pc2_controls.draft.NoiseReductionModeEnum
                if hasattr(nr_enum, "HighQuality"):
                    ctrl_payload["NoiseReductionMode"] = nr_enum.HighQuality
                    applied["NoiseReductionMode"] = "HighQuality"
        except Exception:
            pass

        # tonal tuning
        for k, v in (("Sharpness", 2.0), ("Contrast", 1.5), ("Saturation", 1.5)):
            try:
                ctrl_payload[k] = v
                applied[k] = v
            except Exception:
                pass

        if not ctrl_payload:
            print("No Picamera2-compatible controls found to apply rpicam-like settings")
            return False

        try:
            pc2.set_controls(ctrl_payload)
            print("Applied rpicam-like controls:", ", ".join(f"{k}={v}" for k, v in applied.items()))
            return True
        except Exception as exc:
            print("pc2.set_controls failed:", exc)
            ok_any = False
            for k, v in ctrl_payload.items():
                try:
                    pc2.set_controls({k: v})
                    print(f"  set {k} OK")
                    ok_any = True
                except Exception as e2:
                    print(f"  set {k} failed:", e2)
            return ok_any

    except Exception as exc:
        print("apply_rpicam_like_settings error:", exc)
        return False


def revert_rpicam_like_settings(pc2):
    """Try to revert rpicam-like settings to neutral defaults."""
    if pc2 is None:
        print("Picamera2 not available")
        return False
    try:
        from picamera2 import controls as pc2_controls
    except Exception:
        pc2_controls = None

    payload = {}
    try:
        if pc2_controls and hasattr(pc2_controls, "AfModeEnum"):
            enum = pc2_controls.AfModeEnum
            if hasattr(enum, "Continuous"):
                payload["AfMode"] = enum.Continuous
        if pc2_controls and hasattr(pc2_controls, "AwbModeEnum") and hasattr(pc2_controls.AwbModeEnum, "Auto"):
            payload["AwbMode"] = pc2_controls.AwbModeEnum.Auto
        else:
            payload["AwbEnable"] = True
    except Exception:
        pass

    for k, v in (("Sharpness", 0.0), ("Contrast", 1.0), ("Saturation", 1.0)):
        payload[k] = v

    try:
        pc2.set_controls(payload)
        print("Reverted rpicam-like controls to neutral")
        return True
    except Exception as exc:
        print("Failed to revert controls:", exc)
        return False


# ---------------------------------------------------------------------------
# rpicam-still invocation helpers (single-shot + multishot path)
# ---------------------------------------------------------------------------

def has_rpicam_still() -> bool:
    return shutil.which("rpicam-still") is not None


def capture_with_rpicam(out_path: str = "out.jpg", timeout_ms: int = RPICAM_STILL_TIMEOUT_MS) -> bool:
    if not has_rpicam_still():
        print("rpicam-still not found on PATH")
        return False
    cmd = ["rpicam-still", "-o", out_path, "-t", str(int(timeout_ms))]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=(timeout_ms / 1000.0) + 5.0)
        print(f"rpicam-still capture saved -> {out_path}")
        return True
    except subprocess.CalledProcessError as exc:
        print("rpicam-still failed:", exc)
        return False
    except Exception as exc:
        print("rpicam-still invocation error:", exc)
        return False


def multishot_with_rpicam(out_basename: str = "rpicam_best.jpg", n: int = 5, delay: float = 0.05, working_dir: str = "/tmp/rpicam_multishot"):
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    captured = []
    for i in range(n):
        fname = Path(working_dir) / f"rpicam_{i:03d}.jpg"
        ok = capture_with_rpicam(str(fname), timeout_ms=int(max(200, delay * 1000)))
        if not ok:
            print(f"rpicam-still failed at shot {i}, skipping remaining shots")
            break
        time.sleep(0.02)
        img = None
        if cv2 is not None:
            img = cv2.imread(str(fname))
        captured.append((str(fname), img))
        time.sleep(delay)
    if not captured:
        print("No rpicam captures produced")
        return False
    best_idx = None
    best_score = -1.0
    for idx, (p, img) in enumerate(captured):
        if img is None:
            continue
        s = sharpness_score(img)
        print(f"rpicam shot {idx}: {p} score={s:.2f}")
        if s > best_score:
            best_score = s
            best_idx = idx
    if best_idx is None:
        last = captured[-1][0]
        dst = Path(out_basename)
        shutil.copy(last, dst)
        print(f"No readable frames for scoring; copied last frame to {dst}")
        return True
    best_path = Path(captured[best_idx][0])
    dst = Path(out_basename)
    shutil.copy(best_path, dst)
    print(f"Saved best rpicam frame {best_path} -> {dst} (score={best_score:.2f})")
    return True


# ---------------------------------------------------------------------------
# Capture, scoring, multishot (OpenCV/Picamera2)
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
        scores = [sharpness_score(f) for f in frames]
        best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        best_frame = frames[best_idx]
        fname = f"pos{pos:02d}_best.jpg"
        fpath = Path(out_dir) / fname
        cv2.imwrite(str(fpath), best_frame)
        print(f"Saved best frame for pos {pos}: {fpath}  score={scores[best_idx]:.2f}")
        summary.append((pos, scores[best_idx], str(fpath)))
    csv_path = Path(out_dir) / "summary.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["position", "best_score", "best_image"])
        for r in summary:
            writer.writerow(r)
    print("Sweep complete. Summary written to", csv_path)
    return summary


# ---------------------------------------------------------------------------
# High-resolution capture helper (Picamera2 with fallback)
# ---------------------------------------------------------------------------

def capture_highres_with_picamera2(pc2, out_path="out_highres.jpg", still_size=RPICAM_STILL_SIZE):
    if pc2 is None:
        print("Picamera2 not available for highres capture")
        return False
    try:
        # stop camera before reconfigure (required by some Picamera2 builds)
        try:
            pc2.stop()
        except Exception:
            pass

        # save previous config if available
        try:
            prev_cfg = pc2.configuration
        except Exception:
            prev_cfg = None

        # configure still pipeline
        try:
            still_cfg = pc2.create_still_configuration({"size": still_size})
        except Exception:
            still_cfg = pc2.create_preview_configuration({"size": still_size})
        try:
            pc2.configure(still_cfg)
        except Exception as e:
            print("configure(still) failed:", e)

        # start camera after configure
        try:
            pc2.start()
        except Exception:
            pass

        # apply rpicam-like controls
        try:
            apply_rpicam_like_settings(pc2, still_size=still_size, shutter_us=None, gain=None, af_mode_prefer="Continuous")
        except Exception:
            pass

        time.sleep(0.25)  # let AWB/AF/ISP settle

        # capture_array on your build does not accept timeout; call it without timeout
        arr = pc2.capture_array()
        if arr is None:
            raise RuntimeError("capture_array returned None")

        if cv2 is None:
            print("OpenCV required to save high-res array")
            return False
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, bgr)
        print("Saved high-res still ->", out_path)

        # restore preview config
        try:
            if prev_cfg is not None:
                # stop before reconfigure
                try:
                    pc2.stop()
                except Exception:
                    pass
                pc2.configure(prev_cfg)
                try:
                    pc2.start()
                except Exception:
                    pass
        except Exception:
            pass

        return True
    except Exception as exc:
        print("High-res capture with Picamera2 failed:", exc)
        return False



# ---------------------------------------------------------------------------
# UI / main loop
# ---------------------------------------------------------------------------

def print_help():
    print(
        "\nPreview controls:\n"
        "  q : quit\n"
        "  c : capture current frame -> out.jpg\n"
        "  z : single rpicam-still capture (if available)\n"
        "  Z : multishot using rpicam-still (if available)\n"
        "  k : capture high-res still (Picamera2 attempt, fallback to rpicam-still)\n"
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
        "  y : apply rpicam-still like settings\n"
        "  r : revert Picamera2 tuned and rpicam-like settings to neutral\n"
        "  s : start interactive manual focus sweep\n"
        "  h : print this help\n"
        "  f: triggers autofocus using AfMode=Manual, AfMetering=Window, and a centered AfWindow region.\n"
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
            if key == ord("y"):
                ok = apply_rpicam_like_settings(pc2 if use_pc2 else None, still_size=RPICAM_STILL_SIZE, shutter_us=None, gain=None, af_mode_prefer="Continuous")
                print("Applied rpicam-like settings?", ok)
            if key == ord("r"):
                # single reset key: revert both tuned and rpicam-like Picamera2 controls
                ok1 = revert_picamera2_controls(pc2 if use_pc2 else None)
                ok2 = revert_rpicam_like_settings(pc2 if use_pc2 else None)
                print("Reverted tuned controls:", ok1, "reverted rpicam-like:", ok2)
            if key == ord("z"):
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                outname = f"rpicam_{ts}.jpg"
                ok = capture_with_rpicam(outname, timeout_ms=RPICAM_STILL_TIMEOUT_MS)
                print("rpicam-still single capture saved:" if ok else "rpicam-still capture failed")
            if key == ord("Z"):
                print(f"Running rpicam multishot N={MULTISHOT_N} delay={MULTISHOT_DELAY:.3f}s")
                ok = multishot_with_rpicam(out_basename="rpicam_best.jpg", n=MULTISHOT_N, delay=MULTISHOT_DELAY)
                print("rpicam multishot done:", ok)
            if key == ord("k"):
                print("Attempting high-res capture via Picamera2 (fallback to rpicam-still if needed)")
                ok = capture_highres_with_picamera2(pc2 if use_pc2 else None, out_path="out_highres.jpg", still_size=RPICAM_STILL_SIZE) if use_pc2 else False
                if not ok:
                    print("Picamera2 high-res capture failed or unavailable; falling back to rpicam-still CLI")
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    outname = f"rpicam_highres_{ts}.jpg"
                    ok2 = capture_with_rpicam(outname, timeout_ms=RPICAM_STILL_TIMEOUT_MS)
                    print("Fallback rpicam-still saved:" if ok2 else "Fallback rpicam-still failed")
            if key == ord("s"):
                print("Starting interactive focus sweep. This will pause preview until sweep completes.")
                focus_sweep_interactive(pc2 if use_pc2 else None, cap, shots_per_position=MULTISHOT_N, delay=MULTISHOT_DELAY, positions=SWEEP_POSITIONS, out_dir="focus_sweep")
                print("Returned from focus sweep; resuming preview.")
            if key == ord("h"):
                print_help()
            if key == ord("f"):
                if use_pc2 and pc2 is not None:
                    print("Triggering autofocus cycle...")
                    try:
                        from libcamera import controls
                        pc2.set_controls({
                            "AfMode": controls.AfModeEnum.Manual,
                            "AfMetering": controls.AfMeteringEnum.Auto
                        })
                        success = pc2.autofocus_cycle()
                        print("Autofocus successful." if success else "Autofocus failed.")
                    except Exception as exc:
                        print("Autofocus cycle error:", exc)
                else:
                    print("Picamera2 not available or not running")





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
