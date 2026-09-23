#!/usr/bin/env python
"""Sanity-check that both cameras in config.yaml (camera.device_id and
camera.secondary_device_id) open and deliver frames at the configured
resolution (480p by default - see docs/DUAL_CAMERA.md). Run this before
scripts/identify_dual.py, and check the reported fps against what's needed
for smooth capture on your hardware.

Usage: python scripts/test_dual_camera.py
       python scripts/test_dual_camera.py --no-display   # over SSH / no monitor
Press 'q' to quit the preview window, or Ctrl+C in the terminal.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from src.camera.dual_camera_manager import DualCameraManager
from src.utils.config import load_config


def main() -> int:
    config = load_config()
    headless = "--no-display" in sys.argv

    if config["camera"].get("secondary_device_id") is None:
        print("FAILED: config.yaml's camera.secondary_device_id is not set. "
              "Set it to your second camera's device index (check `ls /dev/video*` "
              "or, on Windows, try 1) before running this.")
        return 1

    try:
        with DualCameraManager(config) as cameras:
            print("Both cameras opened." if headless else
                  "Both cameras opened. Showing preview - press 'q' to quit.")
            frame_count = 0
            start = time.time()
            last_primary, last_secondary = None, None

            for primary_frame, secondary_frame in cameras.frame_pairs():
                frame_count += 1
                last_primary, last_secondary = primary_frame, secondary_frame

                if headless:
                    if frame_count >= 100:
                        break
                    continue

                side_by_side = np.hstack([primary_frame, secondary_frame])
                cv2.imshow("Dual Camera Test - press q to quit", side_by_side)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            elapsed = time.time() - start
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"\nCaptured {frame_count} synced pairs in {elapsed:.1f}s (~{fps:.1f} pairs/s)")

            if headless:
                if not frame_count:
                    print("FAILED: cameras opened but delivered no frames")
                    return 1
                cv2.imwrite("outputs/dual_camera_test_primary.jpg", last_primary)
                cv2.imwrite("outputs/dual_camera_test_secondary.jpg", last_secondary)
                print("Saved last frames to outputs/dual_camera_test_primary.jpg "
                      "and outputs/dual_camera_test_secondary.jpg")
            else:
                cv2.destroyAllWindows()
            return 0

    except RuntimeError as exc:
        print(f"FAILED: {exc}")
        print(f"Check config.yaml's camera.device_id "
              f"(currently {config['camera']['device_id']}) and "
              f"camera.secondary_device_id (currently "
              f"{config['camera']['secondary_device_id']}), and that no other "
              "application is using either camera.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
