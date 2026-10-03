import io
import os
import sys
import time
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.alerts import AlertManager
from app.detector import YOLODetector
from app.tracker import ObjectTracker

# Page config
st.set_page_config(
    page_title="Smart Object Detection Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e222d;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2ecc71;
        margin-bottom: 10px;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector(model_path: str, conf: float, iou: float):
    """Cache loaded YOLO model to avoid re-initializing on each interaction."""
    return YOLODetector(weights_path=model_path, conf_threshold=conf, iou_threshold=iou)


def main():
    # Sidebar
    st.sidebar.image("https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Reverse.svg", width=180)
    st.sidebar.title("🎯 Control Panel")
    st.sidebar.markdown("---")

    # Model selector
    available_models = []
    if os.path.exists("models/best.pt"):
        available_models.append("models/best.pt (Fine-Tuned VOC)")
    if os.path.exists("models/yolo11n.pt"):
        available_models.append("models/yolo11n.pt (Base COCO)")

    if not available_models:
        available_models = ["models/yolo11n.pt"]

    selected_model_label = st.sidebar.selectbox("Select Model Weights", available_models)
    model_path = selected_model_label.split(" ")[0]

    # Hyperparameters
    conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.10, max_value=0.95, value=0.35, step=0.05)
    iou_threshold = st.sidebar.slider("IoU (NMS) Threshold", min_value=0.20, max_value=0.80, value=0.45, step=0.05)

    # Load detector
    detector = load_detector(model_path, conf_threshold, iou_threshold)
    class_names = list(detector.class_names.values())

    # Class filter
    selected_classes = st.sidebar.multiselect(
        "Filter Specific Classes",
        options=class_names,
        default=[],
        help="Leave empty to detect all classes",
    )
    detector.target_classes = selected_classes if selected_classes else None

    # Application Logic Toggles
    st.sidebar.markdown("---")
    st.sidebar.subheader("Analytics & Overlays")
    enable_trails = st.sidebar.checkbox("Show Tracking Trails", value=True)
    enable_line = st.sidebar.checkbox("Virtual Counting Line", value=True)
    enable_zone = st.sidebar.checkbox("Restricted Zone", value=False)

    # Main dashboard header
    st.title("🎯 Smart Object Detection & Monitoring System")
    st.caption("End-to-End Deep Learning Vision Application • Powered by Ultralytics YOLO & OpenCV")

    # Header metric indicators
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Active Model", os.path.basename(model_path))
    with col2:
        st.metric("Total Supported Classes", len(class_names))
    with col3:
        st.metric("Confidence Filter", f"{conf_threshold:.0%}")
    with col4:
        st.metric("System Mode", "Online (Localhost)")

    st.markdown("---")

    # Dashboard Tabs
    tab_image, tab_camera, tab_logs = st.tabs([
        "🖼️ Image & Benchmark Detection",
        "📹 Live Camera & Video Capture",
        "📊 Detection Logs & Analytics",
    ])

    # ---------------- TAB 1: IMAGE INFERENCE ----------------
    with tab_image:
        st.subheader("Test Detection on Images")
        img_col1, img_col2 = st.columns([1, 1])

        with img_col1:
            uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])
            use_sample = st.button("Load Benchmark Sample (`data/test/bus.jpg`)")

        test_img_path = None
        input_image = None

        if uploaded_file is not None:
            input_image = Image.open(uploaded_file).convert("RGB")
        elif use_sample or os.path.exists("data/test/bus.jpg"):
            test_img_path = "data/test/bus.jpg"
            input_image = Image.open(test_img_path).convert("RGB")

        if input_image is not None:
            # Convert PIL to BGR OpenCV format
            open_cv_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)

            # Run inference
            start_t = time.perf_counter()
            detections = detector.detect(open_cv_image)
            latency = (time.perf_counter() - start_t) * 1000

            # Annotate
            annotated_bgr = detector.annotate(open_cv_image, detections)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            with img_col1:
                st.image(input_image, caption="Original Input", use_container_width=True)

            with img_col2:
                st.image(annotated_rgb, caption=f"YOLO Detection Result ({len(detections)} objects, {latency:.1f}ms)", use_container_width=True)

            # Detection breakdown table
            if detections:
                st.markdown("### Detected Objects Summary")
                det_records = []
                for idx, d in enumerate(detections, 1):
                    det_records.append({
                        "Object #": idx,
                        "Class": d.class_name.capitalize(),
                        "Confidence": f"{d.confidence:.1%}",
                        "Bounding Box (x1, y1, x2, y2)": str(d.bbox),
                        "Center (cx, cy)": str(d.center),
                    })
                df_det = pd.DataFrame(det_records)
                st.dataframe(df_det, use_container_width=True)
            else:
                st.info("No objects detected above the confidence threshold. Try lowering the threshold in the sidebar.")

    # ---------------- TAB 2: LIVE CAMERA ----------------
    with tab_camera:
        st.subheader("Live Camera Stream & Video Analysis")
        st.write("You can use your browser webcam below or run the dedicated desktop OpenCV window.")

        cam_col1, cam_col2 = st.columns([2, 1])

        with cam_col1:
            picture = st.camera_input("Take a snapshot from your webcam for real-time analysis:")
            if picture:
                img_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
                frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

                # Inference
                detections = detector.detect(frame)
                annotated = detector.annotate(frame, detections)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                st.image(annotated_rgb, caption=f"Live Detection: {len(detections)} objects found!", use_container_width=True)

        with cam_col2:
            st.info("### 💻 Launch Full OpenCV Desktop Window")
            st.markdown("""
            To run the continuous real-time video pipeline with 30+ FPS, multi-object tracking, and virtual lines:
            ```bash
            python app/main.py --model models/best.pt
            ```
            **Keyboard Shortcuts:**
            - `q`: Quit
            - `SPACE`: Pause / Resume
            - `s`: Save frame snapshot
            - `t`: Toggle motion trails
            - `l`: Toggle counting line
            - `z`: Toggle restricted zone
            """)

    # ---------------- TAB 3: LOGS & ANALYTICS ----------------
    with tab_logs:
        st.subheader("Event Monitoring & Alert History")
        log_file = "data/detections_log.csv"

        if os.path.exists(log_file):
            try:
                df_logs = pd.read_csv(log_file)
                if not df_logs.empty:
                    st.dataframe(df_logs.tail(50), use_container_width=True)
                    st.download_button(
                        label="📥 Download CSV Log",
                        data=open(log_file, "r").read(),
                        file_name="detections_log.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("Log file is currently empty. Events will appear here as objects cross lines or enter zones.")
            except Exception as e:
                st.warning(f"Could not load log: {e}")
        else:
            st.info("No logs created yet. Run `python app/main.py` with tracking enabled to generate event logs.")


if __name__ == "__main__":
    main()
