# Smart Object Detection System (OpenCV + YOLO)

An end-to-end, modular real-time object detection, tracking, and event monitoring system built using **Ultralytics YOLO** and **OpenCV**, designed according to the *OpenCV Object Detection Complete Build Roadmap*.

- 🌐 **Live Web Application**: [https://live-object-detection-ai.vercel.app](https://live-object-detection-ai.vercel.app)
- 🐙 **GitHub Repository**: [https://github.com/AbhaySingh97/live-object-detection-ai](https://github.com/AbhaySingh97/live-object-detection-ai)
- ⚡ **Auto CI/CD**: Any push to the `main` branch automatically builds and deploys to Vercel.

---

## 🌐 Full-Stack MERN Architecture

The system features a **Production MERN + AI Microservice Architecture** (Phase 14 of the roadmap):

```
                    ┌─────────────────────────┐
                    │  React Client (Vite)    │  http://localhost:3000
                    │  Dark Theme UI & Canvas │
                    └────────────┬────────────┘
                                 │ REST / Proxy
                    ┌────────────▼────────────┐
                    │  Node.js / Express API  │  http://localhost:5000
                    │  Event Logger & Store   │
                    └────────────┬────────────┘
                                 │ REST / Multipart
                    ┌────────────▼────────────┐
                    │  Python AI Microservice │  http://localhost:8000
                    │  FastAPI + YOLO + OpenCV│  (models/best.pt)
                    └─────────────────────────┘
```

### 🚀 Running the MERN Stack:
1. Double click **`run_mern.bat`** (or start the 3 services):
   - **AI Microservice**: `python -m uvicorn api.inference_server:app --port 8000`
   - **Express Backend**: `cd server && node server.js`
   - **React Frontend**: `cd client && npm run dev`
2. Open your browser: **`http://localhost:3000`**

---

## 📁 System Architecture & Directory Structure

Following **Phase 11 (Modular Separation of Concerns)**:

```
object-detection/
├── app/
│   ├── __init__.py
│   ├── main.py          # Real-time pipeline orchestrator & interactive HUD
│   ├── camera.py        # Robust threaded camera capture & reconnect logic
│   ├── detector.py      # YOLO inference wrapper & custom HUD visual annotator
│   ├── tracker.py       # Trajectory tracking, dwell time, and motion trails
│   ├── alerts.py        # Line crossing (IN/OUT), zone intrusion & CSV logger
│   └── config.py        # Application configuration dataclass & YAML parser
├── models/
│   └── yolo11n.pt       # Pretrained weights / fine-tuned best.pt
├── data/
│   ├── raw/             # Raw captured footage and snapshots
│   ├── dataset/         # Structured YOLO dataset (train, val, test)
│   │   ├── images/      # [train/, val/, test/]
│   │   ├── labels/      # [train/, val/, test/]
│   │   └── data.yaml    # YOLO dataset metadata & class mapping
│   └── test/            # Sample benchmark images & test outputs
├── training/
│   ├── __init__.py
│   ├── train.py         # Transfer learning & fine-tuning script with smoke tests
│   └── evaluate.py      # Model validation (mAP@50, mAP@50-95, precision, recall)
├── notebooks/
│   └── kaggle_training_template.ipynb # Ready-to-upload GPU training notebook
├── tests/
│   ├── test_config.py   # Configuration unit tests
│   ├── test_detector.py # YOLO inference & annotation tests
│   ├── test_tracker.py  # Track lifecycle & motion history tests
│   └── test_alerts.py   # Geometry & alert logic tests
├── config.yaml          # Default configuration parameters
├── requirements.txt     # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

Dependencies are pre-installed in your environment. To install in a new virtual environment:

```bash
pip install -r requirements.txt
```

### 2. Run Real-Time Webcam Detection

Start the application with default settings (webcam source `0`):

```bash
python app/main.py
```

### 3. Run on a Video File or Image Stream

```bash
# Process a video file
python app/main.py --source "path/to/video.mp4"

# Filter specific classes (e.g. only detect person and car)
python app/main.py --source 0 --classes person car

# Adjust confidence threshold
python app/main.py --source 0 --conf 0.45

# Headless mode (no UI window, ideal for servers)
python app/main.py --source 0 --no-gui
```

### 4. Interactive Keyboard Controls

| Key | Action |
| :--- | :--- |
| **`q`** or **`ESC`** | Quit application |
| **`SPACE`** | Pause / Resume video stream |
| **`s`** | Save frame snapshot to `data/test/snapshot_*.jpg` |
| **`t`** | Toggle motion trails on/off |
| **`l`** | Toggle virtual counting line on/off |
| **`z`** | Toggle restricted zone on/off |
| **`h`** | Toggle Heads-Up Display (HUD) overlay |

---

## 🎯 Application Features & Modules

### 1. Robust Camera Stream (`app/camera.py`)
- Background threaded capture prevents frame drops and UI freezes during deep learning inference.
- Automatic reconnection on device disconnect or network frame loss.
- Rolling average FPS calculation.

### 2. YOLO Detector (`app/detector.py`)
- Seamlessly loads Ultralytics models (`yolo11n.pt`, `yolov8n.pt`, or custom `best.pt`).
- Returns clean, structured `Detection` objects with bounding box, class, confidence, and track ID.
- Professional HUD visualization with corner accents and high-contrast color badges.

### 3. Trajectory Tracking (`app/tracker.py`)
- Persistent object IDs across frames.
- Smooth fading motion trails.
- Dwell time computation (calculates exact seconds an object has stayed in the field of view).

### 4. Event Logic & Alerts (`app/alerts.py`)
- **Virtual Line Crossing**: Directional counting (`IN` vs `OUT`).
- **Restricted Zone Intrusion**: Detects when objects enter sensitive polygonal areas.
- **Dwell Time Alarms**: Alerts when an object stays inside a restricted zone longer than the threshold.
- **Event Logging**: Logs all events to `data/detections_log.csv` and displays active top-right alert banners.

---

## 🏋️ Model Training & Kaggle Workflow

### Training Strategy (Transfer Learning)

Do not train from scratch. Fine-tune from pretrained YOLO weights (`yolo11n.pt`):

1. **Prepare Dataset (`data/dataset/`)**:
   - Organize images into `images/train`, `images/val`, and labels into `labels/train`, `labels/val`.
   - Update `data/dataset/data.yaml` with your target classes.

2. **Kaggle GPU Training**:
   - Upload `notebooks/kaggle_training_template.ipynb` to Kaggle.
   - Attach your dataset under `/kaggle/input/`.
   - Run the **3-epoch Smoke Test** to verify dataset formatting and paths.
   - Run the **50-epoch baseline fine-tuning**.
   - Download the generated `best.pt` model weights.

3. **Deploy Locally**:
   - Place downloaded weights at `models/best.pt`.
   - Run local detection:
     ```bash
     python app/main.py --model models/best.pt
     ```

### Local Training & Evaluation Scripts

```bash
# Run a quick smoke test
python training/train.py --data data/dataset/data.yaml --smoke-test

# Run full local training
python training/train.py --data data/dataset/data.yaml --epochs 50 --imgsz 640

# Evaluate model metrics
python training/evaluate.py --model models/best.pt --split val
```

---

## 🧪 Automated Testing

Run the test suite using `pytest`:

```bash
python -m pytest tests/
```
