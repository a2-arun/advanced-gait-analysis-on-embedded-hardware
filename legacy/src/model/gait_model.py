"""Loads the trained gait checkpoint and runs 1:N identification.

See docs/ARCHITECTURE.md #4 for why this is NOT embedding+cosine-similarity:
the checkpoint's only validated decision mechanism is the difference-based
verification head, which takes the RAW 78-dim feature sequences of a query
and ONE candidate and outputs P(same person). Identification against a
gallery of N enrolled people = running that N times (batched) and taking the
best match above threshold - exactly what the research repo's
check_all_identities() does for 1:1 claims, generalized to open-set search.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from models.full_pipeline import GaitDeepfakeDetector
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


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


def load_checkpoint(path, map_location="cpu") -> dict:
    """torch.load that works on both new torch (needs weights_only=False for a
    checkpoint holding a metrics dict) and Jetson's torch 1.10, which predates
    the weights_only argument."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class GaitModel:
    def __init__(self, config: Optional[dict] = None):
        self.config = (config or load_config())["model"]
        self.device = self._resolve_device(self.config.get("device", "auto"))
        self.model = self._build_and_load_model()
        self.model.eval()

    @staticmethod
    def _resolve_device(device_setting: str) -> torch.device:
        if device_setting == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_setting)

    def _build_and_load_model(self) -> GaitDeepfakeDetector:
        cfg = self.config
        model = GaitDeepfakeDetector(
            input_dim=cfg["input_dim"],
            encoder_hidden_dims=tuple(cfg["encoder_hidden_dims"]),
            encoder_output_dim=cfg["encoder_output_dim"],
            lstm_hidden=cfg["lstm_hidden"],
            lstm_layers=cfg["lstm_layers"],
            transformer_d_model=cfg["transformer_d_model"],
            transformer_heads=cfg["transformer_heads"],
            transformer_layers=cfg["transformer_layers"],
            embedding_dim=cfg["embedding_dim"],
            verification_hidden=cfg["verification_hidden"],
            dropout=cfg["dropout"],
        )

        checkpoint_path = resolve_path(cfg["checkpoint_path"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                "See docs/SETUP.md to copy it from the research repository."
            )

        checkpoint = load_checkpoint(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)

        logger.info(
            "Loaded checkpoint %s (epoch %s, AUC %.2f%%)",
            checkpoint_path.name,
            checkpoint.get("epoch"),
            checkpoint.get("metrics", {}).get("auc", float("nan")),
        )
        return model

    def _to_tensor(self, features: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
        if self.config.get("normalize_features", False):
            raise NotImplementedError(
                "normalize_features=true but no feature_stats mechanism is "
                "implemented - the checkpoint has none and was never "
                "validated with normalized inputs. See docs/ARCHITECTURE.md."
            )
        return tensor

    def identify(
        self,
        query_features: np.ndarray,
        gallery: Dict[str, np.ndarray],
        threshold: Optional[float] = None,
        top_k: int = 3,
    ) -> IdentificationResult:
        """Compare one query gait sequence against every enrolled identity.

        Args:
            query_features: (sequence_length, 78) raw feature sequence
            gallery: name -> (sequence_length, 78) enrolled signature
            threshold: similarity cutoff; defaults to config's operating point
        """
        if not gallery:
            return IdentificationResult(None, 0.0, threshold or 0.0, {}, [])

        threshold = threshold if threshold is not None else self.config.get(
            "similarity_threshold", 0.7737
        )

        names = list(gallery.keys())
        query_tensor = self._to_tensor(query_features).unsqueeze(0)          # (1, T, 78)
        query_batch = query_tensor.repeat(len(names), 1, 1).to(self.device)  # (N, T, 78)
        candidate_batch = torch.stack(
            [self._to_tensor(gallery[name]) for name in names]
        ).to(self.device)  # (N, T, 78)

        with torch.no_grad():
            output = self.model(query_batch, candidate_batch, mode="verification")
            scores = output["similarity"].cpu().numpy()  # (N,) P(same person)

        all_scores = {name: float(score) for name, score in zip(names, scores)}
        ranked = sorted(all_scores.items(), key=lambda kv: kv[1], reverse=True)

        best_name, best_score = ranked[0]
        identified_name = best_name if best_score > threshold else None

        return IdentificationResult(
            identified_name=identified_name,
            similarity=best_score,
            threshold=threshold,
            all_scores=all_scores,
            top_k=ranked[:top_k],
        )

    def get_embedding(self, features: np.ndarray) -> np.ndarray:
        """Gait embedding for diagnostics only - NOT used for matching
        (see module docstring). Useful for clustering/visualization."""
        tensor = self._to_tensor(features).unsqueeze(0).to(self.device)
        embedding = self.model.get_embedding(tensor)
        return embedding.cpu().numpy()[0]
