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


class Base64Payload(BaseModel):
    image: str
    conf: Optional[float] = 0.35
    classes: Optional[List[str]] = None


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "model_path": MODEL_PATH,
        "classes_count": len(detector.class_names),
        "classes": list(detector.class_names.values()),
    }


@app.get("/classes")
def get_classes():
    return {"classes": list(detector.class_names.values())}


@app.post("/detect")
async def detect_image(
    file: Optional[UploadFile] = File(None),
    conf: float = Form(0.35),
    classes: Optional[str] = Form(None),
):
    if file is None:
        return JSONResponse(status_code=400, content={"error": "No image file provided"})

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Failed to decode image"})

    detector.conf_threshold = conf
    detector.target_classes = [c.strip() for c in classes.split(",")] if classes else None

    start_t = time.perf_counter()
    detections = detector.detect(frame)
    latency_ms = (time.perf_counter() - start_t) * 1000

    annotated = detector.annotate(frame, detections)
    _, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
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

    return {
        "success": True,
        "count": len(det_list),
        "latency_ms": round(latency_ms, 2),
        "detections": det_list,
        "annotated_image": f"data:image/jpeg;base64,{b64_img}",
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
def video_feed():
    """MJPEG stream endpoint for real-time video playback in HTML/React."""
    return StreamingResponse(
        generate_video_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
