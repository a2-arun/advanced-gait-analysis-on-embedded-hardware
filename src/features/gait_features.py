"""Rolls live camera frames into model-ready 78-dim gait sequences.

The offline video pipeline (src/pose/pose_extraction.py:process_video) reads
a whole file and normalizes once. A live camera has no "end of file" - this
module is the streaming equivalent: accumulate per-frame landmarks and, once
there's enough motion, hand back a (sequence_length, 78) array through the
same GaitFeatureExtractor.sequence_to_model_input() used offline, so the
model always sees the same feature distribution regardless of source.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.pose.pose_extraction import GaitFeatureExtractor


@dataclass
class CapturedSequence:
    model_input: np.ndarray       # (sequence_length, 78)
    raw_pose_sequence: np.ndarray  # (T, 33, 3) before resampling
    pose_quality: float           # fraction of frames with a detected pose


class LiveSequenceBuffer:
    """Accumulates pose landmarks from consecutive frames and yields a
    complete gait sequence once enough valid frames have been captured.

    Not thread-safe; intended to be driven from a single capture loop.
    """

    def __init__(
        self,
        extractor: GaitFeatureExtractor,
        sequence_length: int = 60,
        min_valid_frames: int = 20,
        max_buffer_frames: int = 120,
    ):
        self.extractor = extractor
        self.sequence_length = sequence_length
        self.min_valid_frames = min_valid_frames
        self.max_buffer_frames = max_buffer_frames

        self._landmarks: List[np.ndarray] = []
        self._total_frames_seen = 0
        self.last_landmarks: Optional[np.ndarray] = None

    @property
    def frames_needed(self) -> int:
        return max(self.min_valid_frames, self.sequence_length // 2)

    def reset(self) -> None:
        self._landmarks = []
        self._total_frames_seen = 0

    @property
    def valid_frame_count(self) -> int:
        return len(self._landmarks)

    def add_frame(self, frame: np.ndarray) -> Optional[CapturedSequence]:
        """Feed one BGR camera frame. Returns a CapturedSequence once the
        buffer has enough frames to build a full gait sequence, else None.
        """
        self._total_frames_seen += 1
        landmarks = self.extractor.extract_pose_from_frame(frame)
        self.last_landmarks = landmarks

        if landmarks is not None:
            self._landmarks.append(landmarks)

        ready = (
            len(self._landmarks) >= self.frames_needed
            or len(self._landmarks) >= self.max_buffer_frames
        )

        if not ready:
            return None

        return self._flush()

    def _flush(self) -> CapturedSequence:
        raw_sequence = np.array(self._landmarks)
        pose_quality = len(self._landmarks) / max(self._total_frames_seen, 1)
        model_input = self.extractor.sequence_to_model_input(raw_sequence)

        result = CapturedSequence(
            model_input=model_input,
            raw_pose_sequence=raw_sequence,
            pose_quality=pose_quality,
        )
        self.reset()
        return result


def average_sequences(sequences: List[np.ndarray]) -> np.ndarray:
    """Average multiple (sequence_length, 78) captures into one signature,
    e.g. for enrollment from several walk passes."""
    if not sequences:
        raise ValueError("Cannot average an empty list of sequences")
    return np.mean(np.stack(sequences, axis=0), axis=0).astype(np.float32)
