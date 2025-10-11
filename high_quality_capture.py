# insert near the top of app/run_inspector.py with other imports
from picamera2 import Picamera2
from libcamera import controls
import time, cv2, os

# High-quality capture helper
def high_quality_capture_and_save(out_path: str, target_size=(1920,1080),
                                  warmup=0.6, settle=0.4,
                                  lock_awb=False, colour_gains=None,
                                  prefer_xbgr=True):
    """
    Temporarily switch Picamera2 to a still/high-quality config, capture, save as a BGR PNG,
    then return True on success.
    - out_path: full path to PNG to write
    - target_size: desired still resolution
    - lock_awb: if True, disable AWB after warmup and set colour_gains (tuple red,blue)
    - colour_gains: (red_gain, blue_gain) used if lock_awb True
    - prefer_xbgr: try XBGR8888 pipeline to avoid color conversion; fallback to RGB888 -> cvtColor
    """
    pc = None
    try:
        pc = Picamera2()
        # Try XBGR first if requested (some IPAs support it natively)
        if prefer_xbgr:
            try:
                cfg = pc.create_still_configuration(main={"size": target_size, "format": "XBGR8888"})
                pc.configure(cfg)
            except Exception:
                cfg = pc.create_still_configuration(main={"size": target_size, "format": "RGB888"})
                pc.configure(cfg)
        else:
            cfg = pc.create_still_configuration(main={"size": target_size, "format": "RGB888"})
            pc.configure(cfg)

        pc.start()
        # let AE/AWB warm up
        time.sleep(warmup)
        # optionally lock AWB and set manual colour gains for repeatable neutral white
        if lock_awb:
            cg = colour_gains if colour_gains is not None else (1.15, 1.0)
            try:
                # Disable AWB and set manual ColourGains (red_gain, blue_gain)
                pc.set_controls({"AwbEnable": False, "ColourGains": cg})
                time.sleep(0.05)
            except Exception:
                pass

        # let pipeline settle a bit if requested
        time.sleep(settle)

        arr = pc.capture_array()
        if arr is None:
            return False, "capture_array returned None"

        # If we captured XBGR8888, arr is likely in BGR ordering already; detect by dtype/shape
        # Heuristic: check channel means — if red mean is much lower than blue mean, we still convert from RGB->BGR
        try:
            r_mean = int(arr[:,:,0].mean())
            b_mean = int(arr[:,:,2].mean())
        except Exception:
            r_mean = b_mean = 0

        wrote = False
        # If we configured XBGR and arr shape is 4 channels, drop alpha if present
        if arr.ndim == 3 and arr.shape[2] == 4:
            # assume XBGR ordering; OpenCV expects BGR so write directly after dropping alpha
            bgr = arr[:, :, :3]  # X B G R or X R G B depending on pipeline; if wrong, fallback below
            # Verify by quick channel-check heuristic and convert if needed
            try:
                # if R mean much less than B mean, convert RGB->BGR
                if r_mean < b_mean:
                    # assume arr is RGB888-like even if 4 channels; convert using cvtColor if possible
                    bgr = cv2.cvtColor(arr[:,:,:3], cv2.COLOR_RGB2BGR)
                cv2.imwrite(out_path, bgr)
                wrote = True
            except Exception:
                pass

        if not wrote:
            # Most consistent path: assume RGB888 and convert to BGR for OpenCV
            try:
                if arr.ndim == 3 and arr.shape[2] >= 3:
                    bgr = cv2.cvtColor(arr[:,:,:3], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(out_path, bgr)
                    wrote = True
                else:
                    # fallback: save raw bytes using numpy
                    import numpy as np
                    cv2.imwrite(out_path, np.ascontiguousarray(arr))
                    wrote = True
            except Exception as e:
                return False, f"save failed: {e}"

        return wrote, out_path
    except Exception as ex:
        return False, str(ex)
    finally:
        try:
            if pc is not None:
                pc.stop()
        except Exception:
            pass
