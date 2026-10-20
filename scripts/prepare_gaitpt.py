#!/usr/bin/env python3
"""One-time restricted checkpoint conversion on Python >=3.10 / modern PyTorch."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.verification.artifacts import prepare_bundle

if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_bundle(args.checkpoint, args.output), indent=2))
