#!/usr/bin/env python3
"""Check installed Nano packages/tools before recording; does not modify the system."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.verification.environment import collect_nano_environment, nano_problems

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='/dev/video0')
    parser.add_argument('--skip-camera', action='store_true')
    parser.add_argument('--capture-test', action='store_true',
                        help='Also record/decode a private temporary 2-second USB clip, then delete it')
    parser.add_argument('--config')
    args = parser.parse_args()
    info = collect_nano_environment(args.device)
    errors = nano_problems(info, check_camera=not args.skip_camera)
    if args.capture_test and not errors:
        try:
            from src.verification.cli import configuration
            from src.verification.environment import camera_decode_smoke
            info['camera_decode_smoke'] = camera_decode_smoke(configuration(args.config), args.device)
        except Exception as exc:
            errors.append('Camera recording/decoding check failed: ' + str(exc))
    print(json.dumps(dict(status='FAIL' if errors else 'PASS', errors=errors, environment=info,
                          scope='Installed environment checks; not biometric or performance certification'), indent=2))
    sys.exit(2 if errors else 0)
