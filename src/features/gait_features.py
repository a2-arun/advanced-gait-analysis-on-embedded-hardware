"""Rolls live camera frames into captured walks for the gait model.

A live camera has no "end of file" - this module accumulates per-frame
MediaPipe landmarks, with the time each frame was seen, until there's enough
walking to embed. Timestamps matter: the camera's effective frame rate drifts
with load, and GaitModel resamples every walk to the 25 fps the model was
trained at.
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.pose.pose_extraction import GaitFeatureExtractor


@dataclass
class CapturedSequence:
    poses: np.ndarray          # (T, 33, 3) MediaPipe normalized landmarks, valid frames only
    timestamps: np.ndarray     # (T,) seconds, when each of those frames was captured
    frame_size: Tuple[int, int]  # (width, height) in pixels
    pose_quality: float        # fraction of frames with a detected pose


class LiveSequenceBuffer:
    """Accumulates pose landmarks from consecutive frames and yields a
    captured walk once enough valid frames have been seen.

    Not thread-safe; intended to be driven from a single capture loop.
    """

    def __init__(self, extractor: GaitFeatureExtractor, min_valid_frames: int = 60):
        self.extractor = extractor
        self.frames_needed = min_valid_frames

        self._landmarks: List[np.ndarray] = []
        self._timestamps: List[float] = []
        self._total_frames_seen = 0
        self.last_landmarks: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._landmarks = []
        self._timestamps = []
        self._total_frames_seen = 0

    @property
    def valid_frame_count(self) -> int:
        return len(self._landmarks)

    def add_frame(self, frame: np.ndarray) -> Optional[CapturedSequence]:
        """Feed one BGR camera frame. Returns a CapturedSequence once the
        buffer has enough frames, else None."""
        self._total_frames_seen += 1
        landmarks = self.extractor.extract_pose_from_frame(frame)
        self.last_landmarks = landmarks

        if landmarks is not None:
            self._landmarks.append(landmarks)
            self._timestamps.append(time.monotonic())

        if len(self._landmarks) < self.frames_needed:
            return None

        result = CapturedSequence(
            poses=np.array(self._landmarks, dtype=np.float32),
            timestamps=np.array(self._timestamps),
            frame_size=(frame.shape[1], frame.shape[0]),
            pose_quality=len(self._landmarks) / max(self._total_frames_seen, 1),
        )
        self.reset()
        return result
