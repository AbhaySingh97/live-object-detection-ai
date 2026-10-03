import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import yaml


@dataclass
class ModelConfig:
    weights_path: str = "models/best.pt" if os.path.exists("models/best.pt") else "models/yolo11n.pt"
    device: Optional[str] = None           # 'cuda', 'cpu', or None for auto
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    target_classes: Optional[List[str]] = None  # None for all classes, or e.g. ["person", "car"]


@dataclass
class CameraConfig:
    """Camera and Video Stream Configuration."""
    source: str = "0"                      # Webcam index as string e.g. "0" or video path/RTSP URL
    width: int = 1280
    height: int = 720
    fps: int = 30
    buffer_size: int = 1                   # Minimize latency for live streams


@dataclass
class TrackingConfig:
    """Object Tracking Configuration."""
    enabled: bool = True
    tracker_type: str = "bytetrack.yaml"   # "bytetrack.yaml" or "botsort.yaml"
    track_history_length: int = 30         # Number of historical points to draw trails
    max_lost_frames: int = 30


@dataclass
class AlertConfig:
    """Virtual Lines, Zones, and Event Alert Configuration."""
    line_crossing_enabled: bool = True
    # (x1, y1), (x2, y2) normalized (0-1) or absolute pixels. If tuple of floats < 1.0, scaled to resolution
    line_points: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (0.1, 0.5),
        (0.9, 0.5),
    )
    zone_detection_enabled: bool = False
    zone_polygon: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
    )
    dwell_time_threshold_sec: float = 5.0
    log_file: str = "data/detections_log.csv"


@dataclass
class DisplayConfig:
    """OpenCV UI and Visualization Configuration."""
    window_name: str = "Smart Object Detection System (OpenCV + YOLO)"
    draw_boxes: bool = True
    draw_trails: bool = True
    draw_lines: bool = True
    draw_zones: bool = True
    draw_hud: bool = True


@dataclass
class AppConfig:
    """Global Application Configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "AppConfig":
        """Load configuration from a YAML file."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        config = cls()
        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "camera" in data:
            config.camera = CameraConfig(**data["camera"])
        if "tracking" in data:
            config.tracking = TrackingConfig(**data["tracking"])
        if "alerts" in data:
            alert_data = data["alerts"]
            if "line_points" in alert_data and isinstance(alert_data["line_points"], list):
                # Ensure line points is tuple of tuples
                alert_data["line_points"] = (
                    tuple(alert_data["line_points"][0]),
                    tuple(alert_data["line_points"][1]),
                )
            if "zone_polygon" in alert_data and isinstance(alert_data["zone_polygon"], list):
                alert_data["zone_polygon"] = [tuple(p) for p in alert_data["zone_polygon"]]
            config.alerts = AlertConfig(**alert_data)
        if "display" in data:
            config.display = DisplayConfig(**data["display"])
        return config

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to a YAML file."""
        os.makedirs(os.path.dirname(os.path.abspath(yaml_path)), exist_ok=True)
        import dataclasses
        import json

        # Convert dataclass through json to convert all tuples to pure JSON/YAML-compatible lists
        raw_dict = dataclasses.asdict(self)
        clean_dict = json.loads(json.dumps(raw_dict))

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(clean_dict, f, default_flow_style=False, sort_keys=False)
