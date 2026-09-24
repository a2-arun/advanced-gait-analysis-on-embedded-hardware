"""One-time setup: turn GaitGraph2's released Lightning checkpoint into the
plain state dict src/model/gait_model.py loads.

Download model_weights.zip from
https://github.com/tteepe/GaitGraph2/releases/tag/v0.1 (not in this repo: the
weights carry no license, so we don't redistribute them), then:

    python scripts/convert_gaitgraph2.py path/to/model_weights.zip
"""

import io
import pickle
import sys
import types
import zipfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.config import load_config, resolve_path

# OUMVLP-Pose, AlphaPose keypoints (stored in OpenPose-18 layout)
MEMBER = "oumvlp_alphapose/checkpoints/last.ckpt"


class _Placeholder:
    """Stands in for pytorch_lightning classes pickled in the checkpoint's
    hyperparameters, so we don't need Lightning installed to read it."""
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


_SAFE_BUILTINS = {"set", "frozenset", "slice", "tuple", "list", "dict", "complex"}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        root = module.split(".")[0]
        if root in ("torch", "collections", "numpy", "_codecs") or (root == "builtins" and name in _SAFE_BUILTINS):
            return super().find_class(module, name)
        return _Placeholder


def main(path: str) -> None:
    if Path(path).is_dir():  # already extracted
        name = str(next(Path(path).glob("**/" + MEMBER)))
        raw = Path(name).read_bytes()
    else:
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith(MEMBER))
            raw = z.read(name)

    pickle_module = types.SimpleNamespace(Unpickler=_Unpickler, load=pickle.load, __name__="pickle")
    # Trusted: GaitGraph2's official release asset, fetched by the user.
    # Classes outside torch/numpy become inert placeholders (see _Unpickler).
    try:
        checkpoint = torch.load(io.BytesIO(raw), map_location="cpu",
                                pickle_module=pickle_module, weights_only=False)
    except TypeError:  # Jetson's torch 1.10 predates weights_only
        checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", pickle_module=pickle_module)

    prefix = "backbone."
    state_dict = {k[len(prefix):]: v for k, v in checkpoint["state_dict"].items() if k.startswith(prefix)}

    out = resolve_path(load_config()["model"]["checkpoint_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, str(out))
    print("Saved {} tensors from {} -> {}".format(len(state_dict), name, out))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
