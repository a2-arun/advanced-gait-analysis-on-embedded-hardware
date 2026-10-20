#!/usr/bin/env python3
"""Verify the explicitly supplied identity claim; never search for another person."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.verification.cli import main
if __name__ == '__main__':
    os.umask(0o077)
    sys.exit(main(['verify'] + sys.argv[1:]))
