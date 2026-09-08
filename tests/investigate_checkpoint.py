"""
Investigate checkpoint structure and normalization approach
"""

import torch
from pathlib import Path

research_dir = Path("../Deepfake-Updated/DeepFake-Detection")
checkpoint_path = research_dir / "outputs/ablation/full_hybrid_best.pth"

print("Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location="cpu")

print("\nCheckpoint keys:")
for key in checkpoint.keys():
    if isinstance(checkpoint[key], dict):
        print(f"  {key}: dict with {len(checkpoint[key])} items")
        if len(checkpoint[key]) < 20:
            for k in checkpoint[key].keys():
                print(f"    - {k}")
    elif isinstance(checkpoint[key], torch.Tensor):
        print(f"  {key}: tensor {checkpoint[key].shape}")
    else:
        print(f"  {key}: {type(checkpoint[key])}")

print("\nModel state dict keys (first 10):")
state_dict = checkpoint['model_state_dict']
for i, key in enumerate(list(state_dict.keys())[:10]):
    print(f"  {key}: {state_dict[key].shape}")
