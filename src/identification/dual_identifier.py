"""Two-camera live identification: run pose extraction on both angles and
identify from whichever produced the higher-quality gait sequence.

This is NOT multi-view fusion into a single feature vector - the checkpoint
was trained on single-view 78-dim sequences, and feeding it anything else
would be running it outside its validated input distribution (see
docs/ARCHITECTURE.md #4-#5). What two angles buys instead is robustness:
if one camera's view of the subject is partially occluded, side-on, or out
of frame, the other angle can still produce a usable sequence, and the
better one is used per identification pass.

Running full MediaPipe pose extraction on two streams roughly doubles CPU
cost versus the single-camera pipeline - that's the real price of this
robustness, not something this module tries to avoid. See
docs/DUAL_CAMERA.md for the Jetson Nano budget.
"""

import time
from typing import Callable, Optional, Tuple

from src.camera.dual_camera_manager import DualCameraManager
from src.database.gait_database import GaitDatabase
from src.events.identity_event import build_event, emit_event
from src.features.gait_features import CapturedSequence, LiveSequenceBuffer
from src.model.gait_model import GaitModel, IdentificationResult
from src.pose.pose_extraction import GaitFeatureExtractor
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DualGaitIdentifier:
    def __init__(
        self,
        config: Optional[dict] = None,
        device_id: str = "jetson-dual",
        pair_grace_period_s: float = 0.5,
    ):
        self.config = config or load_config()
        self.device_id = device_id
        # How long to wait for the second angle's sequence to complete once
        # the first one is ready, before identifying on that one alone.
        self.pair_grace_period_s = pair_grace_period_s

        self.model = GaitModel(self.config)
        self.db = GaitDatabase(self.config)

        cam_cfg = self.config["camera"]
        seq_cfg = self.config["sequence"]

        self.primary_extractor = GaitFeatureExtractor(
            min_detection_confidence=cam_cfg["min_detection_confidence"],
            min_tracking_confidence=cam_cfg["min_tracking_confidence"],
            sequence_length=seq_cfg["length"],
        )
        self.secondary_extractor = GaitFeatureExtractor(
            min_detection_confidence=cam_cfg["min_detection_confidence"],
            min_tracking_confidence=cam_cfg["min_tracking_confidence"],
            sequence_length=seq_cfg["length"],
        )
        self.primary_buffer = LiveSequenceBuffer(
            self.primary_extractor,
            sequence_length=seq_cfg["length"],
            min_valid_frames=seq_cfg["min_valid_frames"],
        )
        self.secondary_buffer = LiveSequenceBuffer(
            self.secondary_extractor,
            sequence_length=seq_cfg["length"],
            min_valid_frames=seq_cfg["min_valid_frames"],
        )

    def close(self) -> None:
        self.primary_extractor.close()
        self.secondary_extractor.close()
        self.db.close()

    def __enter__(self) -> "DualGaitIdentifier":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def identify_sequence(self, captured: CapturedSequence, source: str) -> IdentificationResult:
        gallery = self.db.get_gallery()
        start = time.perf_counter()
        result = self.model.identify(captured.model_input, gallery)
        processing_time_ms = (time.perf_counter() - start) * 1000

        camera_device_id = (
            self.config["camera"]["device_id"] if source == "primary"
            else self.config["camera"]["secondary_device_id"]
        )

        self.db.log_identification_event(
            status="IDENTIFIED" if result.is_identified else "UNKNOWN",
            identified_person_name=result.identified_name,
            top_k=result.top_k,
            threshold_used=result.threshold,
            processing_time_ms=processing_time_ms,
            pose_quality_score=captured.pose_quality,
            num_enrolled_identities=len(gallery),
            device_id=f"{self.device_id}:{source}",
            camera_device_id=camera_device_id,
        )

        event = build_event(result, device_id=self.device_id)
        emit_event(event, self.config)

        return result

    def run(
        self,
        on_result: Optional[Callable[[IdentificationResult, str], None]] = None,
        max_iterations: Optional[int] = None,
    ) -> None:
        """Continuously capture from both cameras and identify each
        completed gait sequence, taking the better-quality angle whenever
        both complete close together in time. Blocks until the cameras
        stop or max_iterations is hit."""
        pending: Tuple[Optional[CapturedSequence], Optional[CapturedSequence]] = (None, None)
        pending_since: Optional[float] = None
        iterations = 0

        with DualCameraManager(self.config) as cameras:
            for primary_frame, secondary_frame in cameras.frame_pairs():
                primary_result = self.primary_buffer.add_frame(primary_frame)
                secondary_result = self.secondary_buffer.add_frame(secondary_frame)

                if primary_result is not None or secondary_result is not None:
                    pending = (
                        primary_result or pending[0],
                        secondary_result or pending[1],
                    )
                    if pending_since is None:
                        pending_since = time.monotonic()

                both_ready = pending[0] is not None and pending[1] is not None
                grace_elapsed = (
                    pending_since is not None
                    and time.monotonic() - pending_since >= self.pair_grace_period_s
                )

                if pending_since is not None and (both_ready or grace_elapsed):
                    candidates = [
                        (seq, name) for seq, name in
                        [(pending[0], "primary"), (pending[1], "secondary")]
                        if seq is not None
                    ]
                    best_sequence, source = max(candidates, key=lambda c: c[0].pose_quality)

                    result = self.identify_sequence(best_sequence, source)
                    if on_result:
                        on_result(result, source)

                    pending = (None, None)
                    pending_since = None

                    iterations += 1
                    if max_iterations is not None and iterations >= max_iterations:
                        return
