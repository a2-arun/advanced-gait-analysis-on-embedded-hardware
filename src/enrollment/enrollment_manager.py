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

logger = get_logger(__name__)

PREVIEW_WINDOW = "Enrollment - press q to cancel"
_BONES = [(11, 12), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26),
          (26, 28), (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32)]
_GREEN, _RED, _WHITE = (0, 200, 0), (0, 0, 255), (255, 255, 255)


def _draw_preview(frame, landmarks, passes_done, passes_max, frames_in_buffer,
                  frames_needed, flash_text):
    h, w = frame.shape[:2]
    tracking = landmarks is not None
    color = _GREEN if tracking else _RED

    if tracking:
        pts = {i: (int(landmarks[i][0] * w), int(landmarks[i][1] * h))
               for i in GaitFeatureExtractor.GAIT_LANDMARKS}
        for a, b in _BONES:
            cv2.line(frame, pts[a], pts[b], _GREEN, 2)
        for p in pts.values():
            cv2.circle(frame, p, 4, _WHITE, -1)

    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 6)
    status = "TRACKING" if tracking else "NO POSE - get full body in view"
    lines = [
        "Pass {}/{}".format(passes_done, passes_max),
        "Frames {}/{}".format(min(frames_in_buffer, frames_needed), frames_needed),
        status,
    ]
    for i, text in enumerate(lines):
        y = 30 + i * 28
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    color if i == 2 else _WHITE, 2)

    if flash_text:
        cv2.putText(frame, flash_text, (12, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
        cv2.putText(frame, flash_text, (12, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, _GREEN, 2)

    cv2.imshow(PREVIEW_WINDOW, frame)


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
    flash_text, flash_frames_left = "", 0

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
                    else:
                        captured.append(result)
                        if on_sequence_captured:
                            on_sequence_captured(len(captured), result)
                        flash_text = "Pass {} captured ({:.0%})".format(len(captured), result.pose_quality)
                    flash_frames_left = 45

                if show_preview:
                    _draw_preview(
                        frame, buffer.last_landmarks, len(captured),
                        enrollment_cfg["max_sequences"], buffer.valid_frame_count,
                        buffer.frames_needed, flash_text if flash_frames_left > 0 else "",
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
