import argparse
from datetime import datetime
import os
import sys
import time
import cv2
import numpy as np

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.alerts import AlertManager
from app.camera import CameraStream
from app.config import AppConfig
from app.detector import YOLODetector
from app.tracker import ObjectTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Smart Object Detection System (OpenCV + YOLO)")
    parser.add_argument("--source", type=str, default=None, help="Video source (0 for webcam, or video/RTSP path)")
    parser.add_argument("--model", type=str, default=None, help="Path to YOLO weights (.pt)")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold (0.0 to 1.0)")
    parser.add_argument("--classes", nargs="+", default=None, help="Filter specific classes (e.g. person car bottle)")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config YAML file")
    parser.add_argument("--no-gui", action="store_true", help="Run in headless mode without GUI window")
    parser.add_argument("--save", type=str, default=None, help="Path to save annotated output video (.mp4)")
    return parser.parse_args()


def draw_hud(frame: np.ndarray, fps: float, latency_ms: float, class_counts: dict, paused: bool) -> np.ndarray:
    """Draw professional translucent heads-up display overlay on top-left of frame."""
    annotated = frame.copy()
    h, w = frame.shape[:2]

    # HUD Box dimensions
    hud_w = 260
    num_classes = len(class_counts)
    hud_h = 100 + (num_classes * 22)

    # Semi-transparent dark background
    overlay = annotated.copy()
    cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
    cv2.rectangle(annotated, (10, 10), (10 + hud_w, 10 + hud_h), (70, 70, 70), 1)

    # Status & FPS
    status_text = "PAUSED" if paused else "LIVE"
    status_color = (0, 165, 255) if paused else (0, 255, 0)
    cv2.circle(annotated, (25, 32), 6, status_color, -1)
    cv2.putText(annotated, f"SYSTEM {status_text}", (38, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(annotated, f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(annotated, (20, 73), (hud_w, 73), (90, 90, 90), 1)

    # Detected classes breakdown
    y = 95
    cv2.putText(annotated, f"Active Objects: {sum(class_counts.values())}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
    y += 22

    for cname, count in class_counts.items():
        cv2.putText(annotated, f"  • {cname.capitalize()}: {count}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
        y += 22

    return annotated


def run_pipeline():
    args = parse_args()

    # Load configuration
    if args.config and os.path.exists(args.config):
        config = AppConfig.from_yaml(args.config)
    else:
        config = AppConfig()

    # Apply CLI overrides
    if args.source is not None:
        config.camera.source = args.source
    if args.model is not None:
        config.model.weights_path = args.model
    if args.conf is not None:
        config.model.conf_threshold = args.conf
    if args.classes is not None:
        config.model.target_classes = args.classes

    print("=" * 60)
    print(" Smart Object Detection System (OpenCV + YOLO)")
    print("=" * 60)
    print(f" Source        : {config.camera.source}")
    print(f" Model Weights : {config.model.weights_path}")
    print(f" Confidence    : {config.model.conf_threshold}")
    print(f" Target Classes: {config.model.target_classes or 'ALL (COCO)'}")
    print(f" Headless Mode : {args.no_gui}")
    print("=" * 60)

    # Initialize modules according to Phase 11 architecture
    detector = YOLODetector(
        weights_path=config.model.weights_path,
        device=config.model.device,
        conf_threshold=config.model.conf_threshold,
        iou_threshold=config.model.iou_threshold,
        target_classes=config.model.target_classes,
    )

    tracker = ObjectTracker(history_length=config.tracking.track_history_length)

    alert_mgr = AlertManager(
        line_points=config.alerts.line_points,
        zone_polygon=config.alerts.zone_polygon,
        dwell_time_threshold_sec=config.alerts.dwell_time_threshold_sec,
        log_file=config.alerts.log_file,
    )

    stream = CameraStream(
        source=config.camera.source,
        width=config.camera.width,
        height=config.camera.height,
        fps=config.camera.fps,
    )

    # Setup video writer if save requested
    writer = None
    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, config.camera.fps, (config.camera.width, config.camera.height))

    # Controls state
    draw_trails = config.display.draw_trails
    draw_lines = config.display.draw_lines
    draw_zones = config.display.draw_zones
    draw_hud_overlay = config.display.draw_hud
    paused = False

    if not args.no_gui:
        cv2.namedWindow(config.display.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(config.display.window_name, 1024, 576)

    print("\n[Controls] q: Quit | s: Save Snapshot | t: Toggle Trails | l: Toggle Line | z: Toggle Zone | h: Toggle HUD | SPACE: Pause\n")

    try:
        stream.start()
        # Small warm-up pause for webcam hardware initialisation
        time.sleep(0.5)

        last_valid_frame = None

        while True:
            if not paused:
                ret, frame = stream.read()
                if not ret or frame is None:
                    # For video files, EOF signals completion
                    if not stream.threaded:
                        print("\nReached end of video stream.")
                        break
                    time.sleep(0.01)
                    continue
                last_valid_frame = frame
            else:
                frame = last_valid_frame.copy() if last_valid_frame is not None else None
                if frame is None:
                    time.sleep(0.05)
                    continue

            h, w = frame.shape[:2]

            # 1. Detection + Tracking with YOLO
            detections = detector.detect(
                frame,
                track=config.tracking.enabled,
                tracker_type=config.tracking.tracker_type,
            )

            # 2. Tracking state update
            active_tracks = tracker.update(detections)

            # 3. Application Logic: line crossing, zone intrusion, dwell time
            alert_mgr.process(
                tracks=active_tracks,
                frame_width=w,
                frame_height=h,
                check_line=draw_lines and config.alerts.line_crossing_enabled,
                check_zone=draw_zones and config.alerts.zone_detection_enabled,
            )

            # 4. Visualization & Annotations
            annotated = detector.annotate(frame, detections)

            if draw_trails and config.tracking.enabled:
                annotated = tracker.draw_trails(annotated)

            if draw_lines or draw_zones:
                annotated = alert_mgr.draw_overlays(annotated, draw_line=draw_lines, draw_zone=draw_zones)

            if draw_hud_overlay:
                # Count classes in current frame
                class_counts = {}
                for det in detections:
                    class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
                annotated = draw_hud(
                    annotated,
                    fps=stream.measured_fps,
                    latency_ms=detector.last_inference_time_ms,
                    class_counts=class_counts,
                    paused=paused,
                )

            # Save frame to output video if enabled
            if writer is not None:
                writer.write(cv2.resize(annotated, (config.camera.width, config.camera.height)))

            # 5. Display GUI
            if not args.no_gui:
                cv2.imshow(config.display.window_name, annotated)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):  # 'q' or ESC
                    print("\nExiting application...")
                    break
                elif key == ord(" "):      # Space to Pause/Resume
                    paused = not paused
                    print(f"Stream {'paused' if paused else 'resumed'}.")
                elif key == ord("s"):      # Save snapshot
                    os.makedirs("data/test", exist_ok=True)
                    snap_name = f"data/test/snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(snap_name, annotated)
                    print(f"Snapshot saved: {snap_name}")
                elif key == ord("t"):      # Toggle trails
                    draw_trails = not draw_trails
                elif key == ord("l"):      # Toggle line
                    draw_lines = not draw_lines
                elif key == ord("z"):      # Toggle zones
                    draw_zones = not draw_zones
                elif key == ord("h"):      # Toggle HUD
                    draw_hud_overlay = not draw_hud_overlay
            else:
                # Headless mode loop throttler
                time.sleep(0.005)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        stream.stop()
        if writer is not None:
            writer.release()
        if not args.no_gui:
            cv2.destroyAllWindows()
        print("Application terminated cleanly.")


if __name__ == "__main__":
    run_pipeline()
