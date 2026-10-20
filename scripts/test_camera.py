#!/usr/bin/env python
"""Sanity-check that the configured camera opens and delivers frames.
Run this before capture/enroll/verify - a working camera is the one thing
this project cannot substitute for.

Usage: python scripts/test_camera.py
       python scripts/test_camera.py --no-display   # over SSH / no monitor: grabs 100 frames
Press 'q' to quit the preview window, or Ctrl+C in the terminal.
"""

import sys
import time
import os
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2

from src.camera.camera_manager import CameraManager
from src.utils.config import load_config


def main() -> int:
    config = load_config()
    headless = "--no-display" in sys.argv

    try:
        with CameraManager(config) as camera:
            print("Camera opened." if headless else "Camera opened. Showing preview - press 'q' to quit.")
            frame_count = 0
            start = time.time()

            for frame in camera.frames():
                frame_count += 1
                if headless:
                    if frame_count >= 100:
                        break
                    continue
                cv2.imshow("Camera Test - press q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            elapsed = time.time() - start
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"\nCaptured {frame_count} frames in {elapsed:.1f}s (~{fps:.1f} fps)")
            if headless:
                if not frame_count:
                    print("FAILED: camera opened but delivered no frames")
                    return 1
                output_dir = Path('outputs')
                output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                snapshot = output_dir / ('camera_test-' + uuid.uuid4().hex + '.jpg')
                fd = os.open(str(snapshot), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(fd)
                if not cv2.imwrite(str(snapshot), frame):
                    snapshot.unlink()
                    print('FAILED: could not save camera snapshot')
                    return 1
                print('Saved last frame to ' + str(snapshot))
            else:
                cv2.destroyAllWindows()
            return 0

    except RuntimeError as exc:
        print(f"FAILED: {exc}")
        print(f"Check config.yaml's camera.device_id (currently {config['camera']['device_id']}) "
              "and that no other application is using the camera.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
