import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple
import cv2
import numpy as np
from app.detector import Detection


@dataclass
class TrackedObject:
    """State of an individually tracked object across consecutive frames."""
    track_id: int
    class_id: int
    class_name: str
    first_seen_time: float = field(default_factory=time.time)
    last_seen_time: float = field(default_factory=time.time)
    history: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=30))
    current_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    current_confidence: float = 0.0

    @property
    def dwell_time(self) -> float:
        """Dwell time in seconds since first detection."""
        return self.last_seen_time - self.first_seen_time

    @property
    def centroid(self) -> Tuple[int, int]:
        """Latest center point."""
        if self.history:
            return self.history[-1]
        x1, y1, x2, y2 = self.current_bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)


class ObjectTracker:
    """
    Maintains persistent track state, movement trails, and lifecycle for tracked objects.
    """

    def __init__(self, history_length: int = 30, max_inactive_seconds: float = 2.0):
        self.history_length = history_length
        self.max_inactive_seconds = max_inactive_seconds
        self.tracks: Dict[int, TrackedObject] = {}

    def update(self, detections: List[Detection]) -> Dict[int, TrackedObject]:
        """
        Update tracking state with new detections from the current frame.
        """
        now = time.time()
        active_ids = set()

        for det in detections:
            if det.track_id is None:
                continue

            active_ids.add(det.track_id)
            center = det.center

            if det.track_id not in self.tracks:
                # New track
                obj = TrackedObject(
                    track_id=det.track_id,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    first_seen_time=now,
                    last_seen_time=now,
                    history=deque(maxlen=self.history_length),
                    current_bbox=det.bbox,
                    current_confidence=det.confidence,
                )
                obj.history.append(center)
                self.tracks[det.track_id] = obj
            else:
                # Existing track
                obj = self.tracks[det.track_id]
                obj.last_seen_time = now
                obj.current_bbox = det.bbox
                obj.current_confidence = det.confidence
                obj.class_name = det.class_name
                obj.history.append(center)

        # Remove inactive / lost tracks
        to_delete = [
            t_id for t_id, obj in self.tracks.items()
            if (now - obj.last_seen_time) > self.max_inactive_seconds
        ]
        for t_id in to_delete:
            del self.tracks[t_id]

        return self.tracks

    def draw_trails(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
        """
        Draw smooth fading motion trails of active tracks on the frame.
        """
        annotated = frame.copy()

        for obj in self.tracks.values():
            pts = list(obj.history)
            if len(pts) < 2:
                continue

            for i in range(1, len(pts)):
                thickness = int(np.sqrt(self.history_length / float(i + 1)) * 1.5) + 1
                cv2.line(annotated, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)

            # Draw current centroid dot
            cv2.circle(annotated, pts[-1], 4, (0, 0, 255), -1, cv2.LINE_AA)

        return annotated
