#!/usr/bin/env python3
"""
web/app.py

FastAPI scaffold for live capture, matching, and simple organization UI.
Expose endpoints:
- GET  /           -> serves static index.html
- GET  /preview    -> MJPEG stream from camera (polls controller.capture_once)
- POST /capture    -> trigger capture_and_match, save to store, return JSON record
- GET  /captures   -> list recent captures (metadata)
- GET  /image/{rec_id}/{filename} -> serve image file from record dir
- POST /captures/{rec_id}/accept -> set accepted flag and optional note
"""

import io
import json
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Response, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from config.config import CONFIG
from pipeline import utils
from scripts import controller
from tools import store as store_mod

# logging
utils.configure_logging(level=CONFIG.log_level)
log = utils.get_logger("web_app")

# init store and DB connection (singleton for the app)
DB_PATH = Path(CONFIG.data_dir or "data") / "captures.db"
_conn = store_mod.init_store(str(DB_PATH))

# create FastAPI app
app = FastAPI()
# mount static files (the UI)
static_dir = Path(__file__).parent / "static"
if not static_dir.exists():
    static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


########################################
# Utility: MJPEG generator using camera
########################################

# single lock so only one stream client accesses camera at a time
_stream_lock = threading.Lock()

def _frame_to_jpeg_bytes(frame):
    import cv2
    ret, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ret:
        return None
    return buf.tobytes()

def mjpeg_generator():
    """
    Generator that yields MJPEG frames from controller.capture_once.
    Keeps streaming until client disconnects.
    """
    try:
        # continuous loop until client disconnects
        while True:
            with _stream_lock:
                card = controller.capture_once()
            if card is None:
                # fallback: small sleep and continue to avoid busy-loop
                time.sleep(0.05)
                continue
            jpeg = _frame_to_jpeg_bytes(card)
            if jpeg is None:
                time.sleep(0.05)
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            # throttle to ~8-10 fps (adjust as needed)
            time.sleep(0.12)
    except GeneratorExit:
        log.debug("MJPEG generator client disconnected")
    except Exception:
        log.exception("MJPEG generator error")


########################################
# Endpoints
########################################

@app.get("/", response_class=HTMLResponse)
def index():
    index_file = static_dir / "index.html"
    if not index_file.exists():
        # a minimal fallback page
        return HTMLResponse("<html><body><h3>Put index.html in web/static/</h3></body></html>")
    return HTMLResponse(index_file.read_text(encoding="utf8"))

@app.get("/preview")
def preview():
    """
    MJPEG stream: client can open /preview as img src or in browser.
    """
    return StreamingResponse(mjpeg_generator(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.post("/capture")
def capture_and_save():
    """
    Trigger capture + match (uses controller.capture_and_match).
    Save result to store and return the record id + metadata.
    """
    # perform capture_and_match; it retries per CONFIG
    try:
        with _stream_lock:
            out = controller.capture_and_match()
    except Exception as e:
        log.exception("capture_and_match error")
        raise HTTPException(status_code=500, detail=str(e))

    if not out:
        # no card captured or no candidate
        return JSONResponse({"ok": False, "reason": "no_capture_or_no_candidate"}, status_code=200)

    card_img = out.get("card")
    result = out.get("result") or out.get("match") or out

    # save into DB via store
    try:
        rec_id = store_mod.save_capture_record(card_img, {"result": result}, conn=_conn)
    except Exception as e:
        log.exception("Failed to save capture")
        raise HTTPException(status_code=500, detail="save_failed: " + str(e))

    rec = store_mod.get_capture(rec_id, conn=_conn)
    return JSONResponse({"ok": True, "id": rec_id, "record": rec})

@app.get("/captures")
def captures_list(limit: Optional[int] = 50):
    rows = store_mod.list_recent_captures(limit=limit, conn=_conn)
    return JSONResponse({"ok": True, "captures": rows})

@app.get("/image/{rec_id}/{filename}")
def serve_image(rec_id: str, filename: str):
    # find file under data/captures/{rec_id}/
    rec_dir = Path(CONFIG.data_dir or "data") / "captures" / rec_id
    if not rec_dir.exists():
        raise HTTPException(status_code=404, detail="record not found")
    path = rec_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path)

@app.post("/captures/{rec_id}/accept")
async def accept_capture(rec_id: str, request: Request):
    body = {}
    try:
        body = await request.json()
    except Exception:
        body = {}
    accepted = body.get("accepted", True)
    note = body.get("note")
    ok = store_mod.set_capture_accept(rec_id, accepted=bool(accepted), note=note, conn=_conn)
    if not ok:
        raise HTTPException(status_code=404, detail="record not found or update failed")
    rec = store_mod.get_capture(rec_id, conn=_conn)
    return JSONResponse({"ok": True, "record": rec})

########################################
# Startup / Shutdown hooks
########################################

@app.on_event("shutdown")
def shutdown_event():
    try:
        store_mod.close_store(_conn)
    except Exception:
        pass
    # close shared camera if created by controller
    try:
        camera = controller._get_camera(shared=True)
        # controller holds a shared camera instance; ensure we close if present
        from pipeline.camera import api as camera_api
        camera_api.close_camera(camera)
    except Exception:
        pass

########################################
# Run via `python web/app.py` for dev
########################################

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
