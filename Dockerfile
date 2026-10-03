# Lightweight Python Docker container for YOLO AI Inference Microservice
FROM python:3.11-slim

# Install system dependencies for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir ultralytics opencv-python-headless fastapi uvicorn python-multipart pydantic

# Copy application code and trained model
COPY app/ app/
COPY api/ api/
COPY models/best.pt models/best.pt

EXPOSE 8000
ENV PORT=8000

CMD ["sh", "-c", "uvicorn api.inference_server:app --host 0.0.0.0 --port ${PORT:-8000}"]
