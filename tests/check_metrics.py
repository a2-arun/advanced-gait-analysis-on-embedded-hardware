"""Check checkpoint in detail"""
import torch
from pathlib import Path

research_dir = Path("../Deepfake-Updated/DeepFake-Detection")
checkpoint_path = research_dir / "outputs/ablation/full_hybrid_best.pth"

checkpoint = torch.load(checkpoint_path, map_location="cpu")

print("Metrics dict:")
metrics = checkpoint['metrics']
for k, v in metrics.items():
    print(f"  {k}: {type(v)} = {v}")

print("\nChecking if params dict has feature info...")
if 'params' in metrics:
    params = metrics['params']
    print(f"Params type: {type(params)}")
    if isinstance(params, dict):
        for k, v in params.items():
            print(f"  {k}: {v}")
