import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.verification.deployment import gaitpt_grew_contract
from src.verification.parity import ParityError, check_fixture, compare, run_parity

FIXTURE = Path(__file__).parent / 'fixtures/research_gaitpt_grew_parity_v1.json'
ROOT = Path(__file__).resolve().parents[2]
CONTRACT = gaitpt_grew_contract({'weights_sha256': 'd' * 64},
                                dict(version='0.8.5', settings={}, asset_sha256=[]))
# Contract identity must equal the audited research pins; tests refuse to fabricate them.
CONTRACT['model']['checkpoint_sha256'] = \
    'e6b287e2c2ae27a9344b8009077c9d4f70811d2d6f3ff20198b99bfa4f824637'


@pytest.fixture
def fixture():
    return json.loads(FIXTURE.read_text())


def replayed(fixture):
    cases = iter(c.get('walk_embedding') for c in fixture['cases'])
    return lambda crops: np.asarray(next(cases))


def levels(report):
    return dict(((r['case'], r['level']), r['status']) for r in report['results'])


def test_research_fixture_identity_matches_deployed_contract(fixture):
    check_fixture(fixture, CONTRACT)
    assert fixture['producer']['commit'] == 'eabfb08409d47517ab44e2cbbd46beef79e42612'


def test_tensor_and_scoring_levels_pass_and_model_levels_are_not_assumed(fixture):
    report = run_parity(fixture, CONTRACT)
    found = levels(report)
    assert found[('whole_sequence_91f', 'model_input')] == 'PASS'
    assert found[('whole_sequence_91f', 'reference_prototype')] == 'PASS'
    assert found[('whole_sequence_91f', 'similarity')] == 'PASS'
    assert found[('whole_sequence_91f', 'walk_embedding')] == 'NOT_RUN'
    assert report['not_run'] == ['walk_embedding'] and report['passed']


def test_replayed_research_embeddings_pass_and_windowed_difference_is_locked(fixture):
    # Same embedding as research: whole-sequence case passes; the windowed case must NOT.
    report = run_parity(fixture, CONTRACT, encode=replayed(fixture))
    found = levels(report)
    assert found[('whole_sequence_91f', 'walk_embedding')] == 'PASS'
    assert found[('windowed_operational_181f', 'walk_embedding')] == 'FAIL'  # unexpected match
    assert not report['passed']
    # A hardware embedding that differs by the documented amount is accepted as documented.
    outputs = iter([fixture['cases'][0]['walk_embedding'],
                    (np.asarray(fixture['cases'][1]['walk_embedding']) + 3e-3).tolist()])
    report = run_parity(fixture, CONTRACT, encode=lambda crops: np.asarray(next(outputs)))
    assert levels(report)[('windowed_operational_181f', 'walk_embedding')] == 'DIFFERS_AS_DOCUMENTED'
    assert report['passed']


def test_numerical_failure_reports_the_measured_difference(fixture):
    fixture['cases'][0]['model_input'][0][0][0][0] += 1e-3
    result = [r for r in run_parity(fixture, CONTRACT)['results'] if r['level'] == 'model_input'][0]
    assert result['status'] == 'FAIL'
    assert result['max_abs_diff'] == pytest.approx(1e-3, rel=1e-2)
    assert result['atol'] == 1e-6


def test_prototype_and_similarity_failures_are_detected(fixture):
    fixture['cases'][0]['similarity'] -= 0.01
    fixture['cases'][0]['reference_prototype'][0] += 0.01
    found = levels(run_parity(fixture, CONTRACT))
    assert found[('whole_sequence_91f', 'similarity')] == 'FAIL'
    assert found[('whole_sequence_91f', 'reference_prototype')] == 'FAIL'


@pytest.mark.parametrize('edit', [
    lambda f: f['identity'].update(checkpoint_sha256='0' * 64),
    lambda f: f['identity'].update(architecture_sha256='0' * 64),
    lambda f: f['identity'].update(source_revision='x'),
    lambda f: f['identity'].update(joint_order=list(range(18))),
    lambda f: f['identity'].update(input_shape=[1, 30, 18, 2]),
    lambda f: f['producer'].pop('commit'),
    lambda f: f.update(schema='gait-parity-fixture-v0'),
    lambda f: f.update(cases=[]),
])
def test_fixture_with_wrong_fingerprint_or_provenance_is_refused(fixture, edit):
    edit(fixture)
    with pytest.raises(ParityError):
        run_parity(fixture, CONTRACT)


def test_compare_flags_shape_and_nonfinite_mismatch():
    assert compare('x', [1, 2], [1, 2, 3], 1e-6)['status'] == 'FAIL'
    assert compare('x', [np.nan], [1.0], 1e-6)['status'] == 'FAIL'
    assert compare('x', [1.0], [1.0 + 1e-7], 1e-6)['status'] == 'PASS'


def test_model_only_parity_against_research_embeddings(fixture):
    """Runs only where torch, the pinned GaitPT clone and the exported NPZ bundle exist."""
    source = os.environ.get('GAIT_PARITY_GAITPT_SOURCE')
    bundle = ROOT / '.cache/gaitpt-grew-npz'
    torch = pytest.importorskip('torch')
    if not source or not (bundle / 'weights.npz').is_file():
        pytest.skip('Set GAIT_PARITY_GAITPT_SOURCE and prepare .cache/gaitpt-grew-npz')
    from src.verification.backend import LocalGaitPT
    device = os.environ.get('GAIT_PARITY_DEVICE', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    manifest = json.loads((bundle / 'manifest.json').read_text())
    contract = gaitpt_grew_contract(manifest, dict(version='0.8.5', settings={}, asset_sha256=[]))
    report = run_parity(fixture, contract,
                        encode=LocalGaitPT(source, str(bundle), 'parity', device).encode_batch,
                        tolerances=dict(walk_embedding=1e-4))
    assert report['not_run'] == [] and report['passed'], report
