#!/usr/bin/env python3
"""Record and enroll one USB-camera walk, or use --video for an original file."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.verification.cli import main
if __name__ == '__main__':
    os.umask(0o077)
    sys.exit(main(['enroll'] + sys.argv[1:]))
