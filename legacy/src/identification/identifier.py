"""Live identification loop: camera -> pose -> gait sequence -> model -> event.

This is the Phase 6 "put it all together" module. scripts/identify.py is a
thin CLI shell around GaitIdentifier.run().
"""

import time
from typing import Callable, Optional

import cv2

from src.camera.camera_manager import CameraManager
from src.database.gait_database import GaitDatabase
from src.events.identity_event import build_event, emit_event
from src.features.gait_features import CapturedSequence, LiveSequenceBuffer
from src.model.gait_model import GaitModel, IdentificationResult
from src.pose.pose_extraction import GaitFeatureExtractor
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.preview import GREEN, ORANGE, draw_preview

logger = get_logger(__name__)

PREVIEW_WINDOW = "Identification - press q to stop"


class GaitIdentifier:
    def __init__(self, config: Optional[dict] = None, device_id: str = "laptop"):
        self.config = config or load_config()
        self.device_id = device_id

        self.model = GaitModel(self.config)
        self.extractor = GaitFeatureExtractor(
            min_detection_confidence=self.config["camera"]["min_detection_confidence"],
            min_tracking_confidence=self.config["camera"]["min_tracking_confidence"],
            sequence_length=self.config["sequence"]["length"],
        )
        self.db = GaitDatabase(self.config)

    def close(self) -> None:
        self.extractor.close()
        self.db.close()

    def __enter__(self) -> "GaitIdentifier":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def identify_sequence(self, captured: CapturedSequence) -> IdentificationResult:
        gallery = self.db.get_gallery()
        start = time.perf_counter()
        result = self.model.identify(captured.model_input, gallery)
        processing_time_ms = (time.perf_counter() - start) * 1000

        self.db.log_identification_event(
            status="IDENTIFIED" if result.is_identified else "UNKNOWN",
            identified_person_name=result.identified_name,
            top_k=result.top_k,
            threshold_used=result.threshold,
            processing_time_ms=processing_time_ms,
            pose_quality_score=captured.pose_quality,
            num_enrolled_identities=len(gallery),
            device_id=self.device_id,
            camera_device_id=self.config["camera"]["device_id"],
        )

        event = build_event(result, device_id=self.device_id)
        emit_event(event, self.config)

        return result

    def run(
        self,
        on_result: Optional[Callable[[IdentificationResult], None]] = None,
        max_iterations: Optional[int] = None,
        show_preview: bool = False,
    ) -> None:
        """Continuously capture gait sequences from the camera and identify
        each one. Blocks until the camera stops, max_iterations is hit, or q
        is pressed in the preview window - intended for scripts/identify.py,
        not for embedding in a UI event loop."""
        sequence_cfg = self.config["sequence"]
        gallery_size = len(self.db.get_gallery())
        banner, banner_color = "", GREEN

        try:
            with CameraManager(self.config) as camera:
                buffer = LiveSequenceBuffer(
                    self.extractor,
                    sequence_length=sequence_cfg["length"],
                    min_valid_frames=sequence_cfg["min_valid_frames"],
                )

                iterations = 0
                for frame in camera.frames():
                    captured = buffer.add_frame(frame)

                    if captured is not None:
                        result = self.identify_sequence(captured)
                        if on_result:
                            on_result(result)
                        if result.is_identified:
                            banner = "{} ({:.2f})".format(result.identified_name, result.similarity)
                            banner_color = GREEN
                        else:
                            banner = "UNKNOWN ({:.2f})".format(result.similarity)
                            banner_color = ORANGE
                        iterations += 1

                    if show_preview:
                        needed = buffer.frames_needed
                        draw_preview(
                            frame, buffer.last_landmarks, PREVIEW_WINDOW,
                            ["Enrolled: {}".format(gallery_size),
                             "Frames {}/{}".format(min(buffer.valid_frame_count, needed), needed)],
                            banner, banner_color,
                        )
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            return

                    if max_iterations is not None and iterations >= max_iterations:
                        return
        finally:
            if show_preview:
                cv2.destroyAllWindows()
