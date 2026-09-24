"""Pretrained GaitGraph2 skeleton-gait embeddings + cosine matching.

The model is GaitGraph2's ResGCN-N51-R4 trained on OUMVLP-Pose (~10k people),
used as-is (no training on our data). See docs/ARCHITECTURE.md #4 for why it
replaced the old deepfake-detection checkpoint, and how it was validated on
the 13-subject research videos.

Pipeline for one captured walk:
    MediaPipe (T, 33, 3) landmarks + timestamps
    -> resample to 25 fps (the rate OUMVLP was recorded at)
    -> 30-frame sliding windows, each mapped to OpenPose-18 joints in
       OUMVLP pixel space -> GaitGraph2 multi-input -> ResGCN
    -> per-window 384-d embedding (3 TTA views x 128) -> mean -> L2 normalize
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.model import resgcn
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

# OpenPose-18 order the checkpoint was trained on: nose, neck, Rsho, Relb,
# Rwri, Lsho, Lelb, Lwri, Rhip, Rkne, Rank, Lhip, Lkne, Lank, Reye, Leye,
# Rear, Lear. MediaPipe has no neck landmark (-1): it's the shoulder midpoint.
MP_TO_OPENPOSE18 = [0, -1, 12, 14, 16, 11, 13, 15, 24, 26, 28, 23, 25, 27, 5, 2, 8, 7]
CONNECT_JOINT = np.array([1, 1, 1, 2, 3, 1, 5, 6, 2, 8, 9, 5, 11, 12, 0, 0, 14, 15])
FLIP_IDX = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16]
CENTER_JOINT = 1  # neck

# Where the checkpoint's input BatchNorm expects joints to land (its running
# stats): OUMVLP's 1280x980 pixel frame, body size set by the spread of y
# relative to the neck. MediaPipe gives no per-joint confidence, so every
# joint gets the training mean confidence (normalizes to 0 in the BN).
OUMVLP_CENTER_XY = np.array([677.1, 460.4], dtype=np.float32)
OUMVLP_REL_Y_STD = 81.2
OUMVLP_MEAN_CONF = 0.624

FPS = 25
WINDOW = 30
STRIDE = 8


@dataclass
class IdentificationResult:
    identified_name: Optional[str]   # None if no candidate cleared the threshold
    similarity: float                # best candidate's score, whether or not it cleared threshold
    threshold: float
    all_scores: Dict[str, float]
    top_k: List[Tuple[str, float]]

    @property
    def is_identified(self) -> bool:
        return self.identified_name is not None


def resample(poses: np.ndarray, timestamps: np.ndarray, fps: int = FPS) -> np.ndarray:
    """(T, J, C) frames at arbitrary times -> evenly spaced frames at `fps`."""
    t = np.arange(timestamps[0], timestamps[-1], 1.0 / fps)
    flat = poses.reshape(len(poses), -1)
    out = np.stack([np.interp(t, timestamps, flat[:, i]) for i in range(flat.shape[1])], axis=1)
    return out.reshape((len(t),) + poses.shape[1:]).astype(np.float32)


def to_oumvlp(poses: np.ndarray, frame_size: Tuple[int, int]) -> np.ndarray:
    """(T, 33, 3) MediaPipe normalized landmarks -> (T, 18, 3) OUMVLP-like x, y, conf."""
    joints = poses[:, [max(i, 0) for i in MP_TO_OPENPOSE18], :2].copy()
    joints[:, 1] = (poses[:, 11, :2] + poses[:, 12, :2]) / 2
    xy = joints * np.asarray(frame_size, dtype=np.float32)       # to pixels, keeps aspect ratio
    rel_y = xy[..., 1] - xy[:, CENTER_JOINT:CENTER_JOINT + 1, 1]
    scale = OUMVLP_REL_Y_STD / max(float(rel_y.std()), 1e-6)
    xy = (xy - xy.reshape(-1, 2).mean(axis=0)) * scale + OUMVLP_CENTER_XY
    conf = np.full(xy.shape[:2] + (1,), OUMVLP_MEAN_CONF, dtype=np.float32)
    return np.concatenate([xy, conf], axis=-1).astype(np.float32)


def multi_input(x: np.ndarray) -> np.ndarray:
    """(T, V, 3) -> (T, V, 3 branches, 5) joints/velocity/bones input.

    Mirrors GaitGraph2's transforms/multi_input.py exactly, quirks included
    (e.g. the confidence overwriting channel 3 of velocity and bones) -
    the checkpoint was trained on this layout.
    """
    T, V, C = x.shape
    out = np.zeros((T, V, 3, C + 2), dtype=np.float32)
    # Joints: x, y, conf, position relative to the neck
    out[:, :, 0, :C] = x
    out[:, :, 0, C:] = x[:, :, :2] - x[:, CENTER_JOINT:CENTER_JOINT + 1, :2]
    # Velocity: 1- and 2-frame displacements (last two frames stay zero)
    out[:T - 2, :, 1, :2] = x[1:T - 1, :, :2] - x[:T - 2, :, :2]
    out[:T - 2, :, 1, 3:] = x[2:, :, :2] - x[:T - 2, :, :2]
    out[:, :, 1, 3] = x[:, :, 2]
    # Bones: vector to the parent joint and its angles
    bone = x[:, :, :2] - x[:, CONNECT_JOINT, :2]
    out[:, :, 2, :2] = bone
    length = np.sqrt((bone ** 2).sum(axis=-1)) + 0.0001
    out[:, :, 2, C:] = np.arccos(np.clip(bone / length[..., None], -1, 1))
    out[:, :, 2, 3] = x[:, :, 2]
    return out


class GaitModel:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        cfg = self.config["model"]
        self.device = self._resolve_device(cfg.get("device", "auto"))
        self.threshold = self.config["identification"]["similarity_threshold"]
        self.top_k = self.config["identification"].get("top_k_candidates", 3)
        self.net = self._load_net(resolve_path(cfg["checkpoint_path"]))
        self.center = np.load(str(resolve_path(cfg["center_path"]))).astype(np.float32)

    @staticmethod
    def _resolve_device(device_setting: str) -> torch.device:
        if device_setting == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_setting)

    def _load_net(self, path) -> torch.nn.Module:
        if not path.exists():
            raise FileNotFoundError(
                "GaitGraph2 weights not found: {}\n"
                "Download model_weights.zip from "
                "https://github.com/tteepe/GaitGraph2/releases/tag/v0.1 and run\n"
                "  python scripts/convert_gaitgraph2.py <path/to/model_weights.zip>\n"
                "(see docs/SETUP.md).".format(path)
            )
        try:
            state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:  # Jetson's torch 1.10 predates weights_only
            state_dict = torch.load(str(path), map_location="cpu")
        # A (the skeleton adjacency) is a buffer in the state dict, so a
        # placeholder of the right shape is enough to build the network.
        net = resgcn.create(
            "resgcn-n51-r4",
            A=torch.zeros_like(state_dict["A"]),
            num_class=128, num_input=3, num_channel=5, parts=None,
        )
        net.load_state_dict(state_dict)
        logger.info("Loaded GaitGraph2 weights %s", path.name)
        return net.to(self.device).eval()

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """GaitGraph2's ResGCN forward: (N, T, V, 3, 5) -> (N, 128), un-normalized."""
        net = self.net
        x = x.permute(0, 3, 4, 1, 2)
        x = torch.cat([branch(x[:, i]) for i, branch in enumerate(net.input_branches)], dim=1)
        for layer in net.main_stream:
            x = layer(x, net.A)
        return net.fcn(net.global_pooling(x).flatten(1))

    def embed(self, poses: np.ndarray, timestamps: np.ndarray,
              frame_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """One walk -> L2-normalized 384-d embedding, or None if the walk is
        shorter than one 30-frame window (1.2 s) after resampling to 25 fps."""
        seq = resample(poses, timestamps)
        if len(seq) < WINDOW:
            return None
        # Normalize each window on its own: walking toward the camera the
        # body grows in the frame, so one scale for the whole walk is wrong.
        windows = [to_oumvlp(seq[s:s + WINDOW], frame_size)
                   for s in range(0, len(seq) - WINDOW + 1, STRIDE)]

        # Test-time augmentation as in GaitGraph2: original, time-reversed,
        # left/right-swapped; the three embeddings are concatenated.
        views = ([multi_input(w) for w in windows]
                 + [multi_input(w[::-1].copy()) for w in windows]
                 + [multi_input(w[:, FLIP_IDX]) for w in windows])
        with torch.no_grad():
            out = self._forward(torch.from_numpy(np.stack(views)).to(self.device)).cpu().numpy()
        n = len(windows)
        per_window = np.concatenate([out[:n], out[n:2 * n], out[2 * n:]], axis=1)
        per_window /= np.linalg.norm(per_window, axis=1, keepdims=True)
        return _normalize(per_window.mean(axis=0))

    @staticmethod
    def make_signature(embeddings: List[np.ndarray]) -> np.ndarray:
        """Several walk-pass embeddings -> one enrolled signature."""
        return _normalize(np.mean(np.stack(embeddings), axis=0))

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine after subtracting the mean embedding of the research videos.
        Raw GaitGraph2 embeddings from any one camera share a large common
        component (every pair scores ~0.99); centering spreads them out so a
        threshold is meaningful. See docs/ARCHITECTURE.md #4."""
        return float(_normalize(a - self.center) @ _normalize(b - self.center))

    def identify(self, query: np.ndarray, gallery: Dict[str, np.ndarray],
                 threshold: Optional[float] = None) -> IdentificationResult:
        """Compare one walk embedding against every enrolled signature."""
        threshold = self.threshold if threshold is None else threshold
        if not gallery:
            return IdentificationResult(None, 0.0, threshold, {}, [])

        for name, signature in gallery.items():
            if signature.shape != query.shape:
                raise ValueError(
                    "Enrolled signature for {!r} has {} values, expected {}: it was "
                    "enrolled with the old model. Delete database/gait.db and "
                    "re-enroll everyone.".format(name, signature.size, query.size)
                )

        all_scores = {name: self.similarity(query, sig) for name, sig in gallery.items()}
        ranked = sorted(all_scores.items(), key=lambda kv: kv[1], reverse=True)
        best_name, best_score = ranked[0]

        return IdentificationResult(
            identified_name=best_name if best_score > threshold else None,
            similarity=best_score,
            threshold=threshold,
            all_scores=all_scores,
            top_k=ranked[:self.top_k],
        )


def _normalize(v: np.ndarray) -> np.ndarray:
    return (v / max(float(np.linalg.norm(v)), 1e-12)).astype(np.float32)
