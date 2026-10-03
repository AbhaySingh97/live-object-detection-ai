import time
from app.detector import Detection
from app.tracker import ObjectTracker


def test_tracker_lifecycle():
    tracker = ObjectTracker(history_length=10, max_inactive_seconds=0.5)

    # Frame 1: object with track_id 1
    det1 = Detection(
        bbox=(100, 100, 200, 200),
        confidence=0.85,
        class_id=0,
        class_name="person",
        track_id=1,
    )
    tracks = tracker.update([det1])
    assert 1 in tracks
    assert tracks[1].class_name == "person"
    assert tracks[1].centroid == (150, 150)
    assert len(tracks[1].history) == 1

    # Frame 2: object moves
    det2 = Detection(
        bbox=(120, 120, 220, 220),
        confidence=0.88,
        class_id=0,
        class_name="person",
        track_id=1,
    )
    tracks = tracker.update([det2])
    assert tracks[1].centroid == (170, 170)
    assert len(tracks[1].history) == 2

    # Inactivity cleanup test
    time.sleep(0.6)
    tracks = tracker.update([])
    assert 1 not in tracks
