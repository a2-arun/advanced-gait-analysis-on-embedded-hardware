"""Two-camera capture for multi-angle gait analysis.

Each camera's blocking cv2.VideoCapture.read() runs in its own background
thread so one camera stalling (a common USB-bus hiccup once two devices
share the Jetson Nano's ports) doesn't stall the other. Each thread keeps
only the latest frame, never a backlog, so callers always get the most
recent pair instead of catching up on stale ones.

Both feeds run at config.yaml's camera.resolution (480p / 640x480 by
default) rather than 720p - see docs/DUAL_CAMERA.md for the USB-bandwidth
and CPU reasoning. MJPEG is requested explicitly for the same reason: it's
roughly 3-5x smaller over USB than raw YUYV at the same resolution, which
matters once two cameras are competing for the same bus.
"""

import threading
import time
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class _CameraStream:
    """Background-threaded single camera reader that always holds the
    latest frame."""

    def __init__(self, device_id, resolution: Tuple[int, int], fps_target: int, name: str):
        self.device_id = device_id
        self.resolution = resolution
        self.fps_target = fps_target
        self.name = name
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def open(self) -> None:
        if isinstance(self.device_id, str):
            self._cap = cv2.VideoCapture(self.device_id, cv2.CAP_GSTREAMER)
        else:
            self._cap = cv2.VideoCapture(self.device_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open '{self.name}' camera (device {self.device_id})")

        if not isinstance(self.device_id, str):
            # Only applies to plain USB/V4L2 capture - a GStreamer pipeline
            # string already picks its own format.
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        width, height = self.resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps_target)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "Camera '%s' (%s) opened: %dx%d @ %.1f fps (requested %dx%d @ %d)",
            self.name, self.device_id, actual_w, actual_h, actual_fps, width, height, self.fps_target,
        )

        self._stop.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.01)

    def latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class DualCameraManager:
    """Opens two USB webcams (e.g. front + side angle) and yields synced
    frame pairs, both at config.yaml's camera.resolution.

    Needs camera.device_id and camera.secondary_device_id both set in
    config.yaml. Use plain CameraManager for a single-camera setup.
    """

    def __init__(self, config: Optional[dict] = None):
        cam_config = (config or load_config())["camera"]
        primary_id = cam_config["device_id"]
        secondary_id = cam_config.get("secondary_device_id")
        if secondary_id is None:
            raise ValueError(
                "camera.secondary_device_id is not set in config.yaml - "
                "DualCameraManager needs two device IDs. Use CameraManager for one camera."
            )
        resolution = tuple(cam_config["resolution"])
        fps_target = cam_config["fps_target"]

        self.primary = _CameraStream(primary_id, resolution, fps_target, name="primary")
        self.secondary = _CameraStream(secondary_id, resolution, fps_target, name="secondary")

    def open(self) -> None:
        self.primary.open()
        self.secondary.open()

    def close(self) -> None:
        self.primary.close()
        self.secondary.close()

    def __enter__(self) -> "DualCameraManager":
        self.open()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def read_pair(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return the latest (primary_frame, secondary_frame). Either may be
        None briefly after open() while that camera's first frame arrives."""
        return self.primary.latest_frame(), self.secondary.latest_frame()

    def frame_pairs(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield synced frame pairs at roughly fps_target, skipping until
        both cameras have delivered at least one frame."""
        interval = 1.0 / self.primary.fps_target if self.primary.fps_target else 0.0
        while True:
            primary_frame, secondary_frame = self.read_pair()
            if primary_frame is not None and secondary_frame is not None:
                yield primary_frame, secondary_frame
            if interval:
                time.sleep(interval)
