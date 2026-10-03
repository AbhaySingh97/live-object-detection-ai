import csv
from datetime import datetime
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import cv2
import numpy as np
from app.tracker import TrackedObject


@dataclass
class AlertEvent:
    timestamp: str
    track_id: int
    class_name: str
    event_type: str  # "LINE_CROSS_IN", "LINE_CROSS_OUT", "ZONE_INTRUSION", "DWELL_TIME"
    confidence: float
    details: str


def _ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
    """Return True if points A, B, C are in counter-clockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def _intersect(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float], D: Tuple[float, float]) -> bool:
    """Return True if line segments AB and CD intersect."""
    return (_ccw(A, C, D) != _ccw(B, C, D)) and (_ccw(A, B, C) != _ccw(A, B, D))


class AlertManager:
    """
    Manages virtual line crossing, polygonal zone detection,
    dwell time alerts, event logging, and on-screen HUD warnings.
    """

    def __init__(
        self,
        line_points: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        zone_polygon: Optional[List[Tuple[float, float]]] = None,
        dwell_time_threshold_sec: float = 5.0,
        log_file: str = "data/detections_log.csv",
    ):
        self.raw_line = line_points or ((0.1, 0.5), (0.9, 0.5))
        self.raw_zone = zone_polygon or [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
        self.dwell_time_threshold_sec = dwell_time_threshold_sec
        self.log_file = log_file

        # Counters
        self.count_in = 0
        self.count_out = 0
        self.counted_ids: Set[int] = set()
        self.active_intrusions: Set[int] = set()
        self.alerted_dwell_ids: Set[int] = set()

        # Recent alerts for on-screen HUD (list of tuples: (timestamp, message, color))
        self.recent_alerts: List[Tuple[float, str, Tuple[int, int, int]]] = []

        # Ensure log directory and CSV headers
        os.makedirs(os.path.dirname(os.path.abspath(self.log_file)), exist_ok=True)
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Track_ID", "Class_Name", "Event_Type", "Confidence", "Details"])

    def _scale_point(self, pt: Tuple[float, float], w: int, h: int) -> Tuple[int, int]:
        """Convert normalized (0-1) coordinates to absolute pixels if needed."""
        x, y = pt
        if x <= 1.0 and y <= 1.0:
            return int(x * w), int(y * h)
        return int(x), int(y)

    def get_scaled_line(self, w: int, h: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        p1 = self._scale_point(self.raw_line[0], w, h)
        p2 = self._scale_point(self.raw_line[1], w, h)
        return p1, p2

    def get_scaled_zone(self, w: int, h: int) -> np.ndarray:
        pts = [self._scale_point(pt, w, h) for pt in self.raw_zone]
        return np.array(pts, dtype=np.int32)

    def log_event(self, event: AlertEvent, banner_color: Tuple[int, int, int] = (0, 0, 255)) -> None:
        """Append event to CSV log and active HUD alert banner."""
        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                event.timestamp,
                event.track_id,
                event.class_name,
                event.event_type,
                f"{event.confidence:.2f}",
                event.details,
            ])
        msg = f"[{event.event_type}] ID #{event.track_id} ({event.class_name})"
        self.recent_alerts.append((time.time(), msg, banner_color))
        if len(self.recent_alerts) > 5:
            self.recent_alerts.pop(0)

    def process(
        self,
        tracks: Dict[int, TrackedObject],
        frame_width: int,
        frame_height: int,
        check_line: bool = True,
        check_zone: bool = True,
    ) -> None:
        """
        Evaluate line crossing, zone intrusion, and dwell time for all active tracks.
        """
        line_p1, line_p2 = self.get_scaled_line(frame_width, frame_height)
        zone_pts = self.get_scaled_zone(frame_width, frame_height)
        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for track_id, obj in tracks.items():
            pts = list(obj.history)
            if len(pts) < 2:
                continue

            prev_pt = pts[-2]
            curr_pt = pts[-1]

            # 1. Virtual Line Crossing
            if check_line and track_id not in self.counted_ids:
                if _intersect(prev_pt, curr_pt, line_p1, line_p2):
                    self.counted_ids.add(track_id)
                    # Determine direction based on y-movement
                    if curr_pt[1] > prev_pt[1]:
                        self.count_in += 1
                        event_type = "LINE_CROSS_IN"
                    else:
                        self.count_out += 1
                        event_type = "LINE_CROSS_OUT"

                    self.log_event(
                        AlertEvent(
                            timestamp=now_ts,
                            track_id=track_id,
                            class_name=obj.class_name,
                            event_type=event_type,
                            confidence=obj.current_confidence,
                            details=f"Crossed line at {curr_pt}",
                        ),
                        banner_color=(0, 200, 0) if event_type == "LINE_CROSS_IN" else (0, 165, 255),
                    )

            # 2. Zone Intrusion
            if check_zone:
                inside = cv2.pointPolygonTest(zone_pts, (float(curr_pt[0]), float(curr_pt[1])), False) >= 0
                if inside:
                    if track_id not in self.active_intrusions:
                        self.active_intrusions.add(track_id)
                        self.log_event(
                            AlertEvent(
                                timestamp=now_ts,
                                track_id=track_id,
                                class_name=obj.class_name,
                                event_type="ZONE_INTRUSION",
                                confidence=obj.current_confidence,
                                details="Entered restricted zone",
                            ),
                            banner_color=(0, 0, 255),
                        )

                    # 3. Dwell Time inside zone
                    if obj.dwell_time >= self.dwell_time_threshold_sec and track_id not in self.alerted_dwell_ids:
                        self.alerted_dwell_ids.add(track_id)
                        self.log_event(
                            AlertEvent(
                                timestamp=now_ts,
                                track_id=track_id,
                                class_name=obj.class_name,
                                event_type="DWELL_TIME_EXCEEDED",
                                confidence=obj.current_confidence,
                                details=f"Lingered inside zone for {obj.dwell_time:.1f}s",
                            ),
                            banner_color=(0, 128, 255),
                        )
                else:
                    self.active_intrusions.discard(track_id)

    def draw_overlays(
        self,
        frame: np.ndarray,
        draw_line: bool = True,
        draw_zone: bool = True,
    ) -> np.ndarray:
        """
        Draw line and zone boundaries on frame.
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Draw virtual counting line
        if draw_line:
            p1, p2 = self.get_scaled_line(w, h)
            cv2.line(annotated, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                annotated,
                f"COUNT LINE [IN: {self.count_in} | OUT: {self.count_out}]",
                (p1[0], max(20, p1[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Draw restricted zone
        if draw_zone:
            zone_pts = self.get_scaled_zone(w, h)
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [zone_pts], (0, 0, 180))
            # Alpha blend for semi-transparent zone
            cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)
            cv2.polylines(annotated, [zone_pts], isClosed=True, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            top_pt = zone_pts[0]
            cv2.putText(
                annotated,
                "RESTRICTED ZONE",
                (top_pt[0] + 5, top_pt[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # Draw recent alert banners in top-right
        now = time.time()
        # Keep banners visible for 4 seconds
        active_banners = [b for b in self.recent_alerts if (now - b[0]) < 4.0]
        banner_y = 30
        for _, msg, color in active_banners[-3:]:
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            bx1 = w - tw - 25
            bx2 = w - 10
            by1 = banner_y - 20
            by2 = banner_y + 8
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, -1)
            cv2.putText(annotated, msg, (bx1 + 8, banner_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            banner_y += 35

        return annotated
