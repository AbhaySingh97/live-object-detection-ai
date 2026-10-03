import os
import tempfile
from collections import deque
from app.alerts import AlertManager, _intersect
from app.tracker import TrackedObject


def test_line_intersection():
    p1 = (0, 50)
    p2 = (100, 50)

    # Segment crossing the horizontal line
    a = (50, 40)
    b = (50, 60)
    assert _intersect(p1, p2, a, b) is True

    # Segment that does not cross
    c = (50, 10)
    d = (50, 40)
    assert _intersect(p1, p2, c, d) is False


def test_alert_manager_crossing_and_zone():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        log_csv = tmp.name

    try:
        manager = AlertManager(
            line_points=((0.0, 0.5), (1.0, 0.5)),
            zone_polygon=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)],
            dwell_time_threshold_sec=1.0,
            log_file=log_csv,
        )

        # Create a mock tracked object moving downward across y=250 (in 500x500 frame)
        obj = TrackedObject(
            track_id=1,
            class_id=0,
            class_name="person",
            history=deque([(250, 200), (250, 300)]),
            current_bbox=(200, 250, 300, 350),
            current_confidence=0.9,
        )

        tracks = {1: obj}
        manager.process(tracks, frame_width=500, frame_height=500, check_line=True, check_zone=True)

        # Verify line cross
        assert manager.count_in == 1
        assert 1 in manager.counted_ids
        # Verify zone intrusion (center 250,300 is inside [100,100] to [400,400])
        assert 1 in manager.active_intrusions
        assert len(manager.recent_alerts) >= 1
    finally:
        if os.path.exists(log_csv):
            os.remove(log_csv)
