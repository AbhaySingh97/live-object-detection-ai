import numpy as np
from app.detector import YOLODetector, Detection


def test_detector_inference_on_synthetic_image():
    # Initialize detector with default nano weights
    detector = YOLODetector(weights_path="yolo11n.pt", conf_threshold=0.25)
    assert detector.model is not None
    assert len(detector.class_names) > 0

    # Create synthetic test frame
    synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(synthetic_frame)

    # Detections should be a list of Detection objects
    assert isinstance(detections, list)
    assert detector.last_inference_time_ms >= 0


def test_detector_annotation():
    detector = YOLODetector(weights_path="yolo11n.pt")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_detection = Detection(
        bbox=(50, 50, 200, 200),
        confidence=0.92,
        class_id=0,
        class_name="person",
        track_id=1,
    )

    annotated = detector.annotate(frame, [mock_detection])
    assert annotated.shape == frame.shape
    # Frame should have non-zero pixels from bounding boxes and badge
    assert np.any(annotated > 0)
