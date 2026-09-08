#!/usr/bin/env python
"""Run live gait identification against the enrolled gallery.

Usage:
    python scripts/identify.py
    python scripts/identify.py --device-id jetson-01   # tag events by device
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.identification.identifier import GaitIdentifier
from src.model.gait_model import IdentificationResult
from src.utils.config import load_config


def print_result(result: IdentificationResult) -> None:
    if result.is_identified:
        print(f"IDENTIFIED: {result.identified_name}  (similarity: {result.similarity:.4f})")
    else:
        print(f"UNKNOWN PERSON  (best candidate similarity: {result.similarity:.4f}, "
              f"threshold: {result.threshold:.4f})")
    if len(result.top_k) > 1:
        others = ", ".join(f"{name}={score:.4f}" for name, score in result.top_k[1:])
        print(f"  other candidates: {others}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Live gait identification")
    parser.add_argument("--device-id", default="laptop", help="Label for this device in logs/events")
    args = parser.parse_args()

    config = load_config()

    with GaitIdentifier(config, device_id=args.device_id) as identifier:
        gallery_size = len(identifier.db.get_gallery())
        if gallery_size == 0:
            print("No identities enrolled yet - run scripts/enroll.py first.")
            return 1

        print(f"{gallery_size} identities enrolled. Watching for a walking subject "
              "(Ctrl+C to stop)...\n")
        try:
            identifier.run(on_result=print_result)
        except KeyboardInterrupt:
            print("\nStopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
