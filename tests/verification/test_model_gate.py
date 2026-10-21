import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from lab_helpers import make_package, nano_facts

from src.verification import backend, capture, cli, environment, model_slot
from src.verification.backend import LocalGaitPT
from src.verification.cli import main
from src.verification.contract import sha256_file
from src.verification.model_slot import ModelIncompatible, ModelNotDeployed
from src.verification.registry import RecordingRegistry
from src.verification.service import MESSAGES, Verifier, response
from src.verification.store import EnrollmentStore

REAL_RESOLVE = cli.resolve_model


def config(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(dict(schema_version=2, source_dir='source', bundle_dir='bundle',
        device='cuda', database='data/enroll.db', recordings='data/recordings.json', policy=None,
        events_dir='events', log_file='logs/operator.log', capture_dir='data/raw/camera',
        model_slot_dir='slot', camera=dict(device_id=0, fps_target=30, resolution=[640, 480]))))
    return path


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """Real model gate; camera, pose and model execution are replaced by fixtures."""
    recorded = []
    monkeypatch.setattr(cli, 'resolve_model', REAL_RESOLVE)
    monkeypatch.setattr(environment, 'collect_nano_environment', lambda *a, **k: nano_facts())
    monkeypatch.setattr(LocalGaitPT, 'validate_artifacts', staticmethod(lambda s, b: {}))
    monkeypatch.setattr(capture, 'capture_video', lambda *a, **k: recorded.append(a))
    monkeypatch.setattr(backend, 'pose_runtime_smoke', lambda: None)
    return config(tmp_path), recorded


def run(cfg, command, capsys, *extra):
    argv = ([command, '--config', str(cfg), '--consent', '--single-walker-reviewed']
            + (['--name', 'S001'] if command == 'enroll' else ['--identity', 'S001']) + list(extra))
    code = main(argv)
    captured = capsys.readouterr()
    run.err = captured.err
    return code, json.loads(captured.out)


def test_model_absent_and_incompatible_are_structured_non_decisions():
    for status in ('MODEL_NOT_DEPLOYED', 'MODEL_INCOMPATIBLE'):
        assert status in MESSAGES
        result = response('S001', status)
        assert result['verdict'] is None and result['status'] == status
        assert set(result) == {'schema_version', 'request_id', 'claimed_subject_id', 'status',
                               'verdict', 'message'}


@pytest.mark.parametrize('error,status', [(ModelNotDeployed('none'), 'MODEL_NOT_DEPLOYED'),
                                          (ModelIncompatible('bad'), 'MODEL_INCOMPATIBLE')])
def test_verifier_maps_model_problems_after_the_claim_is_found(tmp_path, error, status):
    registry = RecordingRegistry(tmp_path / 'recordings.json')
    for name, take in (('ref', 'T1'), ('probe', 'T2')):
        (tmp_path / name).write_bytes(name.encode())
        registry.register(tmp_path / name, take, 'S001', 's1', 'toward', 'development',
                          'consent001', '2026-10-20T12:00:00+05:30')
    store = EnrollmentStore(tmp_path / 'e.db')
    store.add('S001', 'T1', sha256_file(tmp_path / 'ref'), 'r', np.array([1., 0.]), 'development')

    def factory():
        raise error
    verifier = Verifier(store, registry, 'r', factory)
    assert verifier.verify(tmp_path / 'probe', 'S001')['status'] == status
    # No enrollment still wins: the claim is reported before any model is consulted.
    assert Verifier(store, registry, 'r', factory).verify(tmp_path / 'probe', 'S002')[
        'status'] == 'CLAIM_NOT_ENROLLED'


def test_verifier_flags_encoder_recipe_drift_as_model_incompatible(tmp_path):
    registry = RecordingRegistry(tmp_path / 'recordings.json')
    for name, take in (('ref', 'T1'), ('probe', 'T2')):
        (tmp_path / name).write_bytes(name.encode())
        registry.register(tmp_path / name, take, 'S001', 's1', 'toward', 'development',
                          'consent001', '2026-10-20T12:00:00+05:30')
    store = EnrollmentStore(tmp_path / 'e.db')
    store.add('S001', 'T1', sha256_file(tmp_path / 'ref'), 'r', np.array([1., 0.]), 'development')

    class Drifted:
        recipe = 'other-recipe'
    assert Verifier(store, registry, 'r', Drifted).verify(tmp_path / 'probe', 'S001')[
        'status'] == 'MODEL_INCOMPATIBLE'


def test_enroll_refuses_without_active_model_and_touches_nothing(lab, tmp_path, capsys):
    cfg, recorded = lab
    code, result = run(cfg, 'enroll', capsys)
    assert code == 2 and result['status'] == 'MODEL_NOT_DEPLOYED' and result['verdict'] is None
    assert recorded == []
    assert not (tmp_path / 'data').exists()  # no database, registry or recording created


def test_verify_without_model_is_a_non_decision_but_unenrolled_claim_is_reported_first(
        lab, tmp_path, capsys):
    cfg, recorded = lab
    code, result = run(cfg, 'verify', capsys)
    assert result['status'] == 'CLAIM_NOT_ENROLLED'
    EnrollmentStore(tmp_path / 'data/enroll.db').add(
        'S001', 'T1', 'a' * 64, 'some-recipe', np.array([1., 0.]), 'operational')
    code, result = run(cfg, 'verify', capsys)
    assert code == 2 and result['status'] == 'MODEL_NOT_DEPLOYED' and result['verdict'] is None
    assert recorded == []


def test_installed_but_unactivated_or_unapproved_package_is_not_a_model(lab, tmp_path, capsys):
    cfg, recorded = lab
    make_package(tmp_path / 'slot', 'final-1', approve=False)
    code, result = run(cfg, 'enroll', capsys)
    assert result['status'] == 'MODEL_NOT_DEPLOYED' and recorded == []


def activated(tmp_path):
    package = make_package(tmp_path / 'slot', 'final-1')
    model_slot.activate(tmp_path / 'slot', 'final-1', nano_facts(), 'source')
    return package


def test_active_model_enrolls_with_exactly_that_package(lab, tmp_path, monkeypatch, capsys):
    cfg, recorded = lab
    package = activated(tmp_path)
    contract = json.loads((package / 'contract.json').read_text())
    seen = []

    class Encoder:
        recipe = 'active-recipe'

        def __init__(self, source, bundle, recipe, device):
            seen.append(bundle)

        def encode(self, video):
            return np.array([1., 0.])

    def fake_capture(path, *a, **k):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b'synthetic capture')
    monkeypatch.setattr(capture, 'capture_video', fake_capture)
    monkeypatch.setattr(backend, 'runtime_recipe',
                        lambda source, bundle, device: ('active-recipe', dict(contract=contract)))
    monkeypatch.setattr(backend, 'LocalGaitPT', Encoder)
    code, result = run(cfg, 'enroll', capsys)
    assert code == 0 and result['status'] == 'ENROLLED'
    assert seen == [str(package)]


def test_active_model_whose_runtime_contract_differs_is_incompatible(lab, tmp_path, monkeypatch,
                                                                     capsys):
    cfg, recorded = lab
    package = activated(tmp_path)
    other = json.loads((package / 'contract.json').read_text())
    other['pose']['asset_sha256'] = ['c' * 64, 'd' * 64]  # different pose assets on the device
    monkeypatch.setattr(backend, 'runtime_recipe',
                        lambda source, bundle, device: ('r', dict(contract=other)))
    code, result = run(cfg, 'enroll', capsys)
    assert code == 2 and result['status'] == 'MODEL_INCOMPATIBLE' and result['verdict'] is None
    assert recorded == []


def test_development_baseline_requires_explicit_flag_and_warns(lab, tmp_path, monkeypatch, capsys):
    cfg, recorded = lab

    class Encoder:
        recipe = 'baseline-recipe'

        def __init__(self, *args):
            pass

        def encode(self, video):
            return np.array([1., 0.])

    def fake_capture(path, *a, **k):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b'synthetic capture')
    monkeypatch.setattr(capture, 'capture_video', fake_capture)
    monkeypatch.setattr(backend, 'runtime_recipe', lambda *a: ('baseline-recipe', {}))
    monkeypatch.setattr(backend, 'LocalGaitPT', Encoder)
    code, result = run(cfg, 'enroll', capsys)
    assert result['status'] == 'MODEL_NOT_DEPLOYED'
    code, result = run(cfg, 'enroll', capsys, '--development-baseline')
    assert code == 0 and result['status'] == 'ENROLLED'
    assert 'DEVELOPMENT BASELINE' in run.err
