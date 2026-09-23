#!/usr/bin/env python
"""Run live gait identification using two cameras at different angles.

Each identification pass uses whichever camera produced the higher-quality
gait sequence (see src/identification/dual_identifier.py for why this isn't
full multi-view fusion). Requires camera.secondary_device_id to be set in
config.yaml - run scripts/test_dual_camera.py first to confirm both cameras
work.

Usage:
    python scripts/identify_dual.py
    python scripts/identify_dual.py --device-id jetson-nano-01
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.identification.dual_identifier import DualGaitIdentifier
from src.model.gait_model import IdentificationResult
from src.utils.config import load_config


def print_result(result: IdentificationResult, source: str) -> None:
    if result.is_identified:
        print(f"IDENTIFIED: {result.identified_name}  "
              f"(similarity: {result.similarity:.4f}, camera: {source})")
    else:
        print(f"UNKNOWN PERSON  (best candidate similarity: {result.similarity:.4f}, "
              f"threshold: {result.threshold:.4f}, camera: {source})")
    if len(result.top_k) > 1:
        others = ", ".join(f"{name}={score:.4f}" for name, score in result.top_k[1:])
        print(f"  other candidates: {others}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Live dual-camera gait identification")
    parser.add_argument("--device-id", default="jetson-dual", help="Label for this device in logs/events")
    args = parser.parse_args()

    config = load_config()

    if config["camera"].get("secondary_device_id") is None:
        print("config.yaml's camera.secondary_device_id is not set - "
              "run scripts/identify.py for single-camera identification, "
              "or set secondary_device_id and run scripts/test_dual_camera.py first.")
        return 1

    with DualGaitIdentifier(config, device_id=args.device_id) as identifier:
        gallery_size = len(identifier.db.get_gallery())
        if gallery_size == 0:
            print("No identities enrolled yet - run scripts/enroll.py first.")
            return 1

        print(f"{gallery_size} identities enrolled. Watching for a walking subject "
              "on both cameras (Ctrl+C to stop)...\n")
        try:
            identifier.run(on_result=print_result)
        except KeyboardInterrupt:
            print("\nStopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
