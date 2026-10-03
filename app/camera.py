import logging
import threading
import time
from typing import Optional, Tuple, Union
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraStream:
    """
    Robust camera and video capture stream.
    Supports webcam index, video file paths, and RTSP streams.
    Provides threaded non-blocking capture for real-time cameras,
    and synchronous frame-by-frame playback for video files.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        threaded: bool = True,
        auto_reconnect: bool = True,
    ):
        # Parse source: if numeric string like "0", parse to integer
        if isinstance(source, str) and source.isdigit():
            self.source = int(source)
        else:
            self.source = source

        self.width = width
        self.height = height
        self.target_fps = fps
        self.threaded = threaded and not isinstance(self.source, str)  # Default threaded for webcam
        if isinstance(self.source, str) and not self.source.startswith(("rtsp://", "http://", "https://")):
            # Video file: process sequentially to avoid skipping frames
            self.threaded = False

        self.auto_reconnect = auto_reconnect
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        self._current_frame: Optional[np.ndarray] = None
        self._frame_available = False
        self._fps_history = []
        self._last_time = time.time()
        self.measured_fps = 0.0

        self._init_capture()

    def _init_capture(self) -> bool:
        """Initialize or re-initialize OpenCV VideoCapture."""
        if self.cap is not None:
            self.cap.release()

        logger.info(f"Connecting to video source: {self.source}")
        if isinstance(self.source, int):
            # Windows DirectShow backend for faster webcam startup
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Fallback to default backend
                self.cap = cv2.VideoCapture(self.source)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            logger.warning(f"Failed to open video source: {self.source}")
            return False

        if isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Video capture initialized successfully: {actual_w}x{actual_h}")
        return True

    def start(self) -> "CameraStream":
        """Start threaded capture if enabled."""
        if self.is_running:
            return self

        self.is_running = True
        if self.threaded:
            self.thread = threading.Thread(target=self._capture_worker, daemon=True)
            self.thread.start()
        return self

    def _capture_worker(self) -> None:
        """Background thread loop for continuous reading."""
        consecutive_failures = 0
        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                if self.auto_reconnect:
                    logger.info("Attempting camera reconnect...")
                    time.sleep(1.0)
                    self._init_capture()
                    continue
                else:
                    break

            ret, frame = self.cap.read()
            if ret and frame is not None:
                consecutive_failures = 0
                with self.lock:
                    self._current_frame = frame
                    self._frame_available = True
            else:
                consecutive_failures += 1
                if consecutive_failures > 30 and self.auto_reconnect:
                    logger.warning("Excessive frame drop. Reconnecting...")
                    self._init_capture()
                    consecutive_failures = 0
                time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the latest frame.
        Returns:
            (success: bool, frame: np.ndarray)
        """
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        if dt > 0:
            current_instant_fps = 1.0 / dt
            self._fps_history.append(current_instant_fps)
            if len(self._fps_history) > 30:
                self._fps_history.pop(0)
            self.measured_fps = sum(self._fps_history) / len(self._fps_history)

        if self.threaded:
            with self.lock:
                if self._frame_available and self._current_frame is not None:
                    frame = self._current_frame.copy()
                    return True, frame
                return False, None
        else:
            if self.cap is None or not self.cap.isOpened():
                return False, None
            ret, frame = self.cap.read()
            return ret, frame

    def get_resolution(self) -> Tuple[int, int]:
        """Return (width, height) of the stream."""
        if self.cap is not None and self.cap.isOpened():
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        return self.width, self.height

    def stop(self) -> None:
        """Stop capture and release resources."""
        self.is_running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        logger.info("Camera capture stopped and released.")

    def __enter__(self) -> "CameraStream":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
