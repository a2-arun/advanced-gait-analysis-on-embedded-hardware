"""
Model checkpoint sanity checks.

Verifies the checkpoint in models/checkpoint/ loads with the architecture
hyperparameters declared in config.yaml (they must match exactly - the
checkpoint was NOT trained with the create_model() defaults in
models/full_pipeline.py) and that inference runs end to end.

NOTE on normalization: this checkpoint has no saved feature_stats, so
inference runs on raw (non z-score-normalized) features - see
docs/ARCHITECTURE.md for why. There is deliberately no normalization test
here; introducing one would validate a code path the model was never
trained or evaluated with.
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.full_pipeline import GaitDeepfakeDetector
from src.utils.config import load_config, resolve_path

MODEL_CONFIG = load_config()["model"]


def build_model() -> GaitDeepfakeDetector:
    return GaitDeepfakeDetector(
        input_dim=MODEL_CONFIG["input_dim"],
        encoder_hidden_dims=tuple(MODEL_CONFIG["encoder_hidden_dims"]),
        encoder_output_dim=MODEL_CONFIG["encoder_output_dim"],
        lstm_hidden=MODEL_CONFIG["lstm_hidden"],
        lstm_layers=MODEL_CONFIG["lstm_layers"],
        transformer_d_model=MODEL_CONFIG["transformer_d_model"],
        transformer_heads=MODEL_CONFIG["transformer_heads"],
        transformer_layers=MODEL_CONFIG["transformer_layers"],
        embedding_dim=MODEL_CONFIG["embedding_dim"],
        verification_hidden=MODEL_CONFIG["verification_hidden"],
        dropout=MODEL_CONFIG["dropout"],
    )


def test_checkpoint_loading():
    print("=" * 70)
    print("TEST 1: Checkpoint Loading")
    print("=" * 70)

    checkpoint_path = resolve_path(MODEL_CONFIG["checkpoint_path"])
    if not checkpoint_path.exists():
        print(f"[FAIL] Checkpoint not found at {checkpoint_path}")
        print("       See docs/SETUP.md to copy it from the research repository.")
        return False

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        print(f"[OK] Checkpoint loaded: {checkpoint_path}")
        print(f"  File size: {checkpoint_path.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"  Checkpoint keys: {', '.join(checkpoint.keys())}")

        if "feature_stats" not in checkpoint:
            print("  [OK] No feature_stats in checkpoint (expected - see module docstring)")
        return True
    except Exception as e:
        print(f"[FAIL] Could not load checkpoint: {e}")
        return False


def test_model_initialization():
    print("\n" + "=" * 70)
    print("TEST 2: Model Architecture Initialization")
    print("=" * 70)

    try:
        model = build_model()
        print("[OK] Model initialized successfully")
        print(f"  Input: (batch, {MODEL_CONFIG['sequence_length']}, {MODEL_CONFIG['input_dim']})")
        print(f"  Embedding dim: {MODEL_CONFIG['embedding_dim']}")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
        return model
    except Exception as e:
        print(f"[FAIL] Could not initialize model: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_inference(model):
    print("\n" + "=" * 70)
    print("TEST 3: Inference on Dummy Input")
    print("=" * 70)

    if model is None:
        print("[FAIL] SKIPPED: model not initialized")
        return False

    try:
        seq_len = MODEL_CONFIG["sequence_length"]
        input_dim = MODEL_CONFIG["input_dim"]
        embedding_dim = MODEL_CONFIG["embedding_dim"]

        dummy_input = torch.randn(1, seq_len, input_dim)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input, mode="embedding")

        embedding = output["video_embedding"]
        print(f"[OK] Inference successful, embedding shape: {tuple(embedding.shape)}")

        if embedding.shape != torch.Size([1, embedding_dim]):
            print(f"[FAIL] Expected embedding shape [1, {embedding_dim}]")
            return False
        return True
    except Exception as e:
        print(f"[FAIL] Inference error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_verification_mode(model):
    print("\n" + "=" * 70)
    print("TEST 4: Verification Mode (the actual identification mechanism)")
    print("=" * 70)

    if model is None:
        print("[FAIL] SKIPPED: model not initialized")
        return False

    try:
        seq_len = MODEL_CONFIG["sequence_length"]
        input_dim = MODEL_CONFIG["input_dim"]

        query = torch.randn(2, seq_len, input_dim)
        candidate = torch.randn(2, seq_len, input_dim)

        model.eval()
        with torch.no_grad():
            output = model(query, candidate, mode="verification")

        similarity = output["similarity"]
        print(f"[OK] Verification mode ran, similarity shape: {tuple(similarity.shape)}, "
              f"values in [0,1]: {bool(((similarity >= 0) & (similarity <= 1)).all())}")
        return similarity.shape == torch.Size([2])
    except Exception as e:
        print(f"[FAIL] Verification mode error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_device_compatibility():
    print("\n" + "=" * 70)
    print("TEST 5: Device Compatibility")
    print("=" * 70)

    print("[OK] CPU available")
    if torch.cuda.is_available():
        print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("  CUDA not available (fine for laptop/CPU-only development)")
    return True


def test_model_state_loading():
    print("\n" + "=" * 70)
    print("TEST 6: Loading Checkpoint Weights")
    print("=" * 70)

    try:
        model = build_model()
        checkpoint_path = resolve_path(MODEL_CONFIG["checkpoint_path"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        print("[OK] Weights loaded from checkpoint")
        print(f"  Epoch: {checkpoint['epoch']}, AUC: {checkpoint['metrics']['auc']:.2f}%, "
              f"Accuracy: {checkpoint['metrics']['accuracy']:.2f}%")

        model.eval()
        dummy_input = torch.randn(1, MODEL_CONFIG["sequence_length"], MODEL_CONFIG["input_dim"])
        with torch.no_grad():
            output = model(dummy_input, mode="embedding")

        print(f"[OK] Inference with loaded weights successful, "
              f"embedding shape: {tuple(output['video_embedding'].shape)}")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\nGAIT MODEL CHECKPOINT VERIFICATION\n")

    results = {}
    results["Checkpoint Loading"] = test_checkpoint_loading()
    if not results["Checkpoint Loading"]:
        print("\n[FAIL] CRITICAL: cannot proceed without a checkpoint. Aborting.")
        return False

    model = test_model_initialization()
    results["Model Initialization"] = model is not None
    results["Model Inference"] = test_model_inference(model)
    results["Verification Mode"] = test_verification_mode(model)
    results["Device Compatibility"] = test_device_compatibility()
    results["Model State Loading"] = test_model_state_loading()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_passed = True
    for name, passed in results.items():
        print(f"{'[OK] PASS' if passed else '[FAIL] FAIL'}: {name}")
        all_passed = all_passed and passed
    print("=" * 70)

    if all_passed:
        print("\n[OK] All checks passed - the checkpoint is usable via src/model/gait_model.py")
    else:
        print("\n[FAIL] Some checks failed - fix before running scripts/enroll.py or identify.py")

    return all_passed


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
