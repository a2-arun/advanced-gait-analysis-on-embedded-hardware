"""Thin OpenCV webcam wrapper - just enough to keep scripts from repeating
VideoCapture boilerplate and to centralize the config-driven camera settings."""

from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CameraManager:
    def __init__(self, config: Optional[dict] = None):
        cam_config = (config or load_config())["camera"]
        self.device_id = cam_config["device_id"]
        self.resolution: Tuple[int, int] = tuple(cam_config["resolution"])
        self.fps_target = cam_config["fps_target"]
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        if isinstance(self.device_id, str):
            # GStreamer pipeline string, e.g. a Jetson CSI camera (see config.yaml)
            self._cap = cv2.VideoCapture(self.device_id, cv2.CAP_GSTREAMER)
        else:
            self._cap = cv2.VideoCapture(self.device_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera device {self.device_id}")

        width, height = self.resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps_target)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "Camera %s opened: %dx%d @ %.1f fps (requested %dx%d @ %d)",
            self.device_id, actual_w, actual_h, actual_fps, width, height, self.fps_target,
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "CameraManager":
        self.open()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            raise RuntimeError("Camera not opened - use CameraManager as a context manager")
        ok, frame = self._cap.read()
        return frame if ok else None

    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames until the camera stops delivering them or the
        caller breaks out of the loop."""
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame
