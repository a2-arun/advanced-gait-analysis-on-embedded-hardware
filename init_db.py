#!/usr/bin/env python3
"""Create the V2 reference database without touching the historical database."""
import os
from src.verification.cli import configuration, protect_originals
from src.verification.registry import RecordingRegistry
from src.verification.store import EnrollmentStore
if __name__ == '__main__':
    os.umask(0o077)
    config = configuration(None)
    protect_originals(config, RecordingRegistry(config['recordings']))
    EnrollmentStore(config['database'])
    print('Verification V2 database initialized.')
