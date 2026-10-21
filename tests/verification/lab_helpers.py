"""Synthetic fixtures shared by the lab-readiness tests. Nothing here is a real biometric."""
import copy
import json
from pathlib import Path

import numpy as np

from src.verification.backend import POSE_SETTINGS
from src.verification.contract import sha256_file
from src.verification.deployment import gaitpt_grew_contract, semantic_id
from src.verification.environment import pinned_packages

APPROVAL = 'nano-model-approval-v1'


def nano_facts(**changes):
    """Pretend-Nano environment facts (as collect_nano_environment returns them)."""
    facts = dict(machine='aarch64', python='3.6.9', ubuntu='18.04',
                 hardware='NVIDIA Jetson Nano Developer Kit', l4t='# R32 (release), REVISION: 7.1,',
                 packages=dict(pinned_packages(), torch='1.10.0+git36449ea', mediapipe='0.8.5',
                               opencv='4.1.1', matplotlib='2.1.1'),
                 import_errors={}, cuda='10.2', cuda_available=True,
                 tools=dict(ffmpeg=True, ffprobe=True, libx264=True), camera_accessible=True)
    facts.update(changes)
    return facts


def host_facts():
    return nano_facts(machine='x86_64', hardware='', python='3.11.1', ubuntu='24.04', l4t='')


def make_package(root, name='final-1', approve=True, parity=True, edit=None, parity_hardware=None):
    """Write a structurally complete synthetic deployment package under root/name."""
    directory = Path(root) / name
    directory.mkdir(parents=True)
    weights = directory / 'weights.npz'
    np.savez(str(weights), w=np.arange(6, dtype=np.float32))
    digest = sha256_file(weights)
    contract = gaitpt_grew_contract(dict(weights_sha256=digest), dict(
        version='0.8.5', settings=POSE_SETTINGS, asset_sha256=['a' * 64, 'b' * 64]))
    contract = copy.deepcopy(contract)
    if edit:
        edit(contract)
    (directory / 'contract.json').write_text(json.dumps(contract))
    (directory / 'manifest.json').write_text(json.dumps(dict(schema='gaitpt-npz-v1',
                                                             weights_sha256=digest)))
    sid = semantic_id(contract)
    if approve:
        (directory / 'approval.json').write_text(json.dumps(dict(
            schema=APPROVAL, semantic_id=sid, weights_sha256=digest, approved_by='research-owner',
            approved_at='2026-10-22T09:00:00+05:30', research_reference='repo@commit')))
    if parity:
        hardware = parity_hardware or 'NVIDIA Jetson Nano Developer Kit'
        (directory / 'parity_report.json').write_text(json.dumps(dict(
            schema='nano-model-parity-v1', semantic_id=sid, weights_sha256=digest,
            environment=dict(hardware=hardware, python='3.6.9'),
            report=dict(schema='gait-parity-report-v1', passed=True, not_run=[], results=[]))))
    return directory


def allow_development_model(monkeypatch):
    """Existing CLI fixtures exercise registry/store/policy with a fake encoder; this lets them
    bypass the activated-model gate exactly as `--development-baseline` would."""
    from src.verification import cli
    monkeypatch.setattr(cli, 'resolve_model', lambda args, cfg: (cfg['bundle_dir'], None))
