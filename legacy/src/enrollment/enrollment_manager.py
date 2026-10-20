"""Drives the "walk in front of the camera a few times" enrollment flow:
capture N gait sequences, average them into one signature, store it.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

import cv2
import numpy as np

from src.camera.camera_manager import CameraManager
from src.database.gait_database import GaitDatabase
from src.features.gait_features import CapturedSequence, LiveSequenceBuffer, average_sequences
from src.pose.pose_extraction import GaitFeatureExtractor
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.preview import GREEN, ORANGE, draw_preview

logger = get_logger(__name__)

PREVIEW_WINDOW = "Enrollment - press q to cancel"


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
    show_preview: bool = False,
) -> EnrollmentOutcome:
    """Capture config['enrollment']['min_sequences']..max_sequences walk
    passes from the live camera and enroll the averaged signature.

    on_sequence_captured(index, sequence) fires after each captured pass,
    e.g. so a CLI can print progress or a UI can show a checkmark.
    show_preview opens an OpenCV window with the skeleton and progress;
    pressing q there cancels enrollment.
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
    cancelled = False
    flash_text, flash_color, flash_frames_left = "", GREEN, 0

    try:
        with CameraManager(config) as camera:
            buffer = LiveSequenceBuffer(
                extractor,
                sequence_length=sequence_cfg["length"],
                min_valid_frames=sequence_cfg["min_valid_frames"],
            )

            for frame in camera.frames():
                result = buffer.add_frame(frame)

                if result is not None:
                    if result.pose_quality < enrollment_cfg["quality_threshold"]:
                        logger.warning(
                            "Discarding low-quality pass (%.0f%% valid frames < %.0f%% required)",
                            result.pose_quality * 100,
                            enrollment_cfg["quality_threshold"] * 100,
                        )
                        flash_text = "Pass rejected ({:.0%}) - walk again".format(result.pose_quality)
                        flash_color = ORANGE
                    else:
                        captured.append(result)
                        if on_sequence_captured:
                            on_sequence_captured(len(captured), result)
                        flash_text = "Pass {} captured ({:.0%})".format(len(captured), result.pose_quality)
                        flash_color = GREEN
                    flash_frames_left = 45

                if show_preview:
                    needed = buffer.frames_needed
                    draw_preview(
                        frame, buffer.last_landmarks, PREVIEW_WINDOW,
                        ["Pass {}/{}".format(len(captured), enrollment_cfg["max_sequences"]),
                         "Frames {}/{}".format(min(buffer.valid_frame_count, needed), needed)],
                        flash_text if flash_frames_left > 0 else "", flash_color,
                    )
                    flash_frames_left -= 1
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        cancelled = True
                        break

                if len(captured) >= enrollment_cfg["max_sequences"]:
                    break
    finally:
        extractor.close()
        if show_preview:
            cv2.destroyAllWindows()

    if cancelled:
        raise RuntimeError("Cancelled from the preview window - nothing was saved.")

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
