"""Drives the "walk in front of the camera a few times" enrollment flow:
capture N gait sequences, average them into one signature, store it.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from src.camera.camera_manager import CameraManager
from src.database.gait_database import GaitDatabase
from src.features.gait_features import CapturedSequence, LiveSequenceBuffer, average_sequences
from src.pose.pose_extraction import GaitFeatureExtractor
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EnrollmentOutcome:
    person_name: str
    person_id: str
    num_sequences: int
    average_quality: float


def enroll_from_camera(
    person_name: str,
    config: Optional[dict] = None,
    on_sequence_captured: Optional[Callable[[int, CapturedSequence], None]] = None,
) -> EnrollmentOutcome:
    """Capture config['enrollment']['min_sequences']..max_sequences walk
    passes from the live camera and enroll the averaged signature.

    on_sequence_captured(index, sequence) fires after each captured pass,
    e.g. so a CLI can print progress or a UI can show a checkmark.
    """
    config = config or load_config()
    enrollment_cfg = config["enrollment"]
    sequence_cfg = config["sequence"]

    extractor = GaitFeatureExtractor(
        min_detection_confidence=config["camera"]["min_detection_confidence"],
        min_tracking_confidence=config["camera"]["min_tracking_confidence"],
        sequence_length=sequence_cfg["length"],
    )

    captured: List[CapturedSequence] = []

    try:
        with CameraManager(config) as camera:
            buffer = LiveSequenceBuffer(
                extractor,
                sequence_length=sequence_cfg["length"],
                min_valid_frames=sequence_cfg["min_valid_frames"],
            )

            for frame in camera.frames():
                result = buffer.add_frame(frame)
                if result is None:
                    continue

                if result.pose_quality < enrollment_cfg["quality_threshold"]:
                    logger.warning(
                        "Discarding low-quality pass (%.0f%% valid frames < %.0f%% required)",
                        result.pose_quality * 100,
                        enrollment_cfg["quality_threshold"] * 100,
                    )
                    continue

                captured.append(result)
                if on_sequence_captured:
                    on_sequence_captured(len(captured), result)

                if len(captured) >= enrollment_cfg["max_sequences"]:
                    break
    finally:
        extractor.close()

    if len(captured) < enrollment_cfg["min_sequences"]:
        raise RuntimeError(
            f"Only captured {len(captured)} usable sequence(s), "
            f"need at least {enrollment_cfg['min_sequences']}. Try again with "
            "better lighting/framing, or lower enrollment.quality_threshold."
        )

    signature = average_sequences([c.model_input for c in captured])
    average_quality = float(np.mean([c.pose_quality for c in captured]))

    with GaitDatabase(config) as db:
        person_id = db.enroll_identity(
            person_name=person_name,
            signature=signature,
            num_samples=len(captured),
            quality_score=average_quality,
        )

    return EnrollmentOutcome(
        person_name=person_name,
        person_id=person_id,
        num_sequences=len(captured),
        average_quality=average_quality,
    )
