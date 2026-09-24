"""Loads config.yaml once and hands back a plain dict.

No pydantic/dataclass schema on purpose - the config is small, every module
reads only the section it needs, and a schema layer would just be another
place for the real checkpoint hyperparameters to drift out of sync (see
docs/ARCHITECTURE.md for why that already happened once).
"""

from pathlib import Path
from typing import Any, Optional
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

_cached_config: Optional[dict] = None


def load_config(path: Optional[Path] = None, force_reload: bool = False) -> dict:
    """Load config.yaml. Cached after first call unless force_reload=True."""
    global _cached_config

    if _cached_config is not None and not force_reload and path is None:
        return _cached_config

    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if path is None:
        _cached_config = config

    return config


def resolve_path(relative_path: str) -> Path:
    """Resolve a config path (e.g. 'models/checkpoint/gaitgraph2_oumvlp.pth')
    against the project root, so scripts work regardless of cwd."""
    p = Path(relative_path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def get(config: dict, dotted_key: str, default: Any = None) -> Any:
    """Fetch a nested key via dotted path, e.g. get(cfg, 'model.embedding_dim')."""
    node = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
