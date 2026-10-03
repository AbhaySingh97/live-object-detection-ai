import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """Represents a single detected object in a frame."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coordinates
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None

    @property
    def center(self) -> Tuple[int, int]:
        """Return center point (cx, cy) of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    @property
    def area(self) -> int:
        """Return box area in pixels."""
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


class YOLODetector:
    """
    High-level YOLO object detector wrapper.
    Handles weights loading, inference, filtering, tracking, and visualization.
    """

    # High-contrast, clean distinct colors for annotations
    PALETTE = [
        (46, 204, 113),   # Emerald Green
        (52, 152, 219),   # Peter River Blue
        (155, 89, 182),   # Amethyst Purple
        (241, 196, 15),   # Sun Yellow
        (230, 126, 34),   # Carrot Orange
        (231, 76, 60),    # Alizarin Red
        (26, 188, 156),   # Turquoise
        (243, 156, 18),   # Orange
        (211, 84, 0),     # Pumpkin
        (192, 57, 43),    # Pomegranate
    ]

    def __init__(
        self,
        weights_path: str = "models/yolo11n.pt",
        device: Optional[str] = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        target_classes: Optional[List[str]] = None,
    ):
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes
        self.device = device
        self.last_inference_time_ms = 0.0

        # Load YOLO model (auto-downloads official weights if file doesn't exist locally)
        self.model = self._load_model()
        self.class_names: Dict[int, str] = self.model.names if hasattr(self.model, "names") else {}

    def _load_model(self) -> YOLO:
        """Load YOLO model checkpoint with fallback."""
        if not os.path.exists(self.weights_path):
            # Check if model exists in root or fallback to yolo11n.pt
            base_name = os.path.basename(self.weights_path)
            if os.path.exists(base_name):
                return YOLO(base_name)
            # If path points to models/yolo11n.pt, download to models/ directory
            os.makedirs(os.path.dirname(os.path.abspath(self.weights_path)), exist_ok=True)
            print(f"[YOLODetector] Model weights '{self.weights_path}' not found locally. Loading/downloading '{base_name}'...")
            return YOLO(base_name)
        return YOLO(self.weights_path)

    def detect(
        self,
        frame: np.ndarray,
        track: bool = False,
        tracker_type: str = "bytetrack.yaml",
    ) -> List[Detection]:
        """
        Run inference on a single BGR frame.
        
        Args:
            frame: BGR numpy image from OpenCV.
            track: Whether to enable built-in multi-object tracking.
            tracker_type: Tracker configuration file.
            
        Returns:
            List of Detection objects.
        """
        start_t = time.perf_counter()

        # Run inference or tracking
        if track:
            results = self.model.track(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                tracker=tracker_type,
                persist=True,
                verbose=False,
            )
        else:
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )

        self.last_inference_time_ms = (time.perf_counter() - start_t) * 1000.0

        detections: List[Detection] = []
        if not results or len(results) == 0:
            return detections

        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(cls_ids)

        for box, conf, cls_id, track_id in zip(xyxy, confs, cls_ids, track_ids):
            class_name = self.class_names.get(cls_id, f"class_{cls_id}")

            # Filter if target_classes specified
            if self.target_classes and class_name not in self.target_classes:
                continue

            x1, y1, x2, y2 = [int(v) for v in box]
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=class_name,
                    track_id=int(track_id) if track_id is not None else None,
                )
            )

        return detections

    def annotate(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        show_confidence: bool = True,
        show_track_id: bool = True,
    ) -> np.ndarray:
        """
        Draw clean, high-visibility bounding boxes and tags on frame.
        """
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = self.PALETTE[det.class_id % len(self.PALETTE)]

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Draw corner accents for professional HUD feel
            corner_len = min(16, max(6, (x2 - x1) // 6), max(6, (y2 - y1) // 6))
            thick = 3
            # Top-left
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color, thick)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color, thick)
            # Top-right
            cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), color, thick)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), color, thick)
            # Bottom-left
            cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), color, thick)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), color, thick)
            # Bottom-right
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), color, thick)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), color, thick)

            # Prepare label text
            label_parts = []
            if show_track_id and det.track_id is not None:
                label_parts.append(f"#{det.track_id}")
            label_parts.append(det.class_name.capitalize())
            if show_confidence:
                label_parts.append(f"{det.confidence:.0%}")
            label = " | ".join(label_parts)

            # Draw label badge background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            badge_y1 = max(0, y1 - text_h - 8)
            badge_y2 = y1
            badge_x1 = x1
            badge_x2 = x1 + text_w + 10

            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
            cv2.putText(
                annotated,
                label,
                (badge_x1 + 5, badge_y2 - 4),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA,
            )

        return annotated
