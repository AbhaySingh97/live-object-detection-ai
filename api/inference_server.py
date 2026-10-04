import base64
import os
import sys
import time
from typing import List, Optional
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.camera import CameraStream
from app.detector import YOLODetector

app = FastAPI(title="YOLO Object Detection AI Service", version="1.0.0")

# Enable CORS for React frontend (localhost:3000, localhost:5173) and Node.js backend (localhost:5000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
MODEL_PATH = "models/best.pt" if os.path.exists("models/best.pt") else "models/yolo11n.pt"
detector = YOLODetector(weights_path=MODEL_PATH, conf_threshold=0.35)

# Camera stream instance (lazy initialized)
camera_stream: Optional[CameraStream] = None


import threading
import urllib.request
import logging

logger = logging.getLogger("uvicorn")


def keep_awake_daemon():
    """Background heartbeat daemon to keep Render service awake 24/7.
    Render free tier sleeps after 15 minutes of inactivity.
    This worker pings the public endpoint every 10 minutes (600s).
    """
    time.sleep(60)  # Wait for initial boot
    render_url = os.environ.get("RENDER_EXTERNAL_URL", "https://live-object-detection-ai.onrender.com")
    ping_url = f"{render_url.rstrip('/')}/health"
    logger.info(f"[Keep-Awake] Heartbeat worker active. Target: {ping_url}")

    while True:
        try:
            time.sleep(600)  # Ping every 10 minutes
            req = urllib.request.Request(
                ping_url,
                headers={"User-Agent": "Render-KeepAlive-Worker/1.0"},
            )
            with urllib.request.urlopen(req, timeout=25) as resp:
                if resp.status == 200:
                    logger.info(f"[Keep-Awake] Self-ping successful: {ping_url}")
        except Exception as err:
            logger.warning(f"[Keep-Awake] Ping exception (will retry in 10m): {err}")


@app.on_event("startup")
def on_startup():
    t = threading.Thread(target=keep_awake_daemon, daemon=True, name="KeepAwakeWorker")
    t.start()


# In-memory events database
events_db: List[dict] = []


def record_event(data: dict, filename: str):
    event = {
        "id": int(time.time() * 1000),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "classes": [d["class_name"] for d in data.get("detections", [])],
        "count": data.get("count", 0),
        "latency_ms": data.get("latency_ms", 0),
        "filename": filename,
    }
    events_db.insert(0, event)
    if len(events_db) > 500:
        events_db.pop()
    return event["id"]


@app.get("/health")
@app.get("/api/health")
def health_check():
    return {
        "server": "ObjectVision Cloud AI Backend",
        "ai_service": "online",
        "model_details": {
            "status": "online",
            "model_path": MODEL_PATH,
            "classes_count": len(detector.class_names),
            "classes": list(detector.class_names.values()),
        },
        "stored_events_count": len(events_db),
    }


@app.get("/classes")
@app.get("/api/classes")
def get_classes():
    return {"classes": list(detector.class_names.values())}


@app.post("/detect")
@app.post("/api/detect")
async def detect_image(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    conf: float = Form(0.35),
    classes: Optional[str] = Form(None),
):
    target_file = file or image
    if target_file is None:
        return JSONResponse(status_code=400, content={"error": "No image file provided"})

    contents = await target_file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Failed to decode image"})

    # Memory-safe downscaling for phone/high-res photos (Render 512MB RAM optimization)
    h, w = frame.shape[:2]
    max_dim = 1024
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    detector.conf_threshold = conf
    detector.target_classes = [c.strip() for c in classes.split(",")] if classes else None

    start_t = time.perf_counter()
    detections = detector.detect(frame)
    latency_ms = (time.perf_counter() - start_t) * 1000

    annotated = detector.annotate(frame, detections)
    _, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    b64_img = base64.b64encode(buffer).decode("utf-8")

    det_list = [
        {
            "class_name": d.class_name,
            "confidence": round(d.confidence, 4),
            "bbox": list(d.bbox),
            "center": list(d.center),
        }
        for d in detections
    ]

    result = {
        "success": True,
        "count": len(det_list),
        "latency_ms": round(latency_ms, 2),
        "detections": det_list,
        "annotated_image": f"data:image/jpeg;base64,{b64_img}",
    }

    event_id = record_event(result, target_file.filename or "image.jpg")
    result["logged_event_id"] = event_id
    return result


@app.get("/events")
@app.get("/api/events")
def get_events():
    return {"total": len(events_db), "events": events_db[:50]}


@app.delete("/events")
@app.delete("/api/events")
def clear_events():
    global events_db
    events_db = []
    return {"success": True, "message": "Event logs cleared"}


@app.get("/stats")
@app.get("/api/stats")
def get_stats():
    class_frequencies = {}
    total_objects = 0
    total_latency = 0.0

    for e in events_db:
        total_objects += e.get("count", 0)
        total_latency += e.get("latency_ms", 0)
        for c in e.get("classes", []):
            class_frequencies[c] = class_frequencies.get(c, 0) + 1

    count = len(events_db)
    avg_latency = round(total_latency / count, 1) if count > 0 else 0

    return {
        "total_inferences": count,
        "total_objects_detected": total_objects,
        "avg_latency_ms": avg_latency,
        "class_frequencies": class_frequencies,
    }


def generate_video_frames():
    """Generator for live MJPEG webcam streaming."""
    global camera_stream
    if camera_stream is None:
        camera_stream = CameraStream(source=0, width=1280, height=720, fps=30)
        camera_stream.start()

    while True:
        ret, frame = camera_stream.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        detections = detector.detect(frame)
        annotated = detector.annotate(frame, detections)

        _, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.get("/video_feed")
@app.get("/api/video_feed")
def video_feed():
    """MJPEG stream endpoint for real-time video playback in HTML/React."""
    return StreamingResponse(
        generate_video_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
