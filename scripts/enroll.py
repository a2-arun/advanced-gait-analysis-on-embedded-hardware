#!/usr/bin/env python
"""Enroll a new identity from the live camera.

Usage:
    python scripts/enroll.py --name "Alice"
    python scripts/enroll.py --name "Alice" --no-display   # over SSH / no monitor
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.enrollment.enrollment_manager import enroll_from_camera
from src.utils.config import load_config
from src.utils.preview import display_available


def main() -> int:
    parser = argparse.ArgumentParser(description="Enroll a gait identity from the webcam")
    parser.add_argument("--name", required=True, help="Name of the person to enroll")
    parser.add_argument("--no-display", action="store_true",
                        help="Don't open the live preview window")
    args = parser.parse_args()

    show_preview = not args.no_display
    if show_preview and not display_available():
        print("No display found (SSH?) - running without preview. "
              "Use `export DISPLAY=:0` to show it on the board's monitor.")
        show_preview = False

    config = load_config()
    enrollment_cfg = config["enrollment"]

    print(f"Enrolling '{args.name}'.")
    print(f"Walk across the camera's view {enrollment_cfg['min_sequences']}-"
          f"{enrollment_cfg['max_sequences']} times. Each pass needs a clear, "
          "unobstructed full-body view.\n")

    def on_captured(index, result):
        print(f"  Pass {index} captured (pose quality: {result.pose_quality:.0%})")

    try:
        outcome = enroll_from_camera(args.name, config, on_sequence_captured=on_captured,
                                     show_preview=show_preview)
    except RuntimeError as exc:
        print(f"\nEnrollment failed: {exc}")
        return 1

    print(f"\nEnrolled '{outcome.person_name}' "
          f"({outcome.num_sequences} sequences, avg quality {outcome.average_quality:.0%})")
    print(f"person_id: {outcome.person_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
