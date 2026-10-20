import json
import numpy as np
import pytest

from src.verification.contract import InsufficientGait, native_crops, normalize, fingerprint
from src.verification.store import EnrollmentStore
from src.verification.policy import decide


def track(n=61):
    xy = np.zeros((n, 33, 2), dtype=np.float32)
    xy[:, :, 0] = np.arange(33) + 100
    xy[:, :, 1] = 100
    xy[:, 27:33, 1] = 400
    xy[:, 27:33, 0] += np.sin(np.arange(n)[:, None]) * 10
    return xy, np.ones((n, 33)), np.arange(n) / 30.


def test_native_height_scale_and_official_repeated_nose():
    xy, confidence, pts = track()
    batch = native_crops(xy, confidence, pts, 640, 480)
    assert batch.shape == (2, 30, 18, 2)
    np.testing.assert_allclose(batch[0, 0, 0], xy[0, 0] / 480)
    np.testing.assert_allclose(batch[:, :, 0], batch[:, :, 3])
    np.testing.assert_allclose(batch[1, 0, 0], xy[30, 0] / 480)


@pytest.mark.parametrize('kind', ['short', 'missing', 'cropped', 'timestamp', 'static', 'jump'])
def test_quality_never_fabricates_evidence(kind):
    xy, confidence, pts = track()
    if kind == 'short':
        xy, confidence, pts = track(30)
    elif kind == 'missing':
        confidence[::20] = 0
    elif kind == 'cropped':
        xy[:, 31, 1] = 500
    elif kind == 'timestamp':
        pts[3] = pts[2]
    elif kind == 'static':
        xy[:] = xy[0]
    else:
        xy[30:] += 400
    with pytest.raises(InsufficientGait):
        native_crops(xy, confidence, pts, 640, 480)


@pytest.mark.parametrize('value', [[0., 0.], [float('nan'), 1.], [[1., 2.]]])
def test_invalid_vectors_refused(value):
    with pytest.raises(ValueError):
        normalize(value)


def test_store_claim_isolation_original_uniqueness_and_private_file(tmp_path):
    store = EnrollmentStore(tmp_path / 'new.db')
    store.add('S001', 'take1', 'a'*64, 'recipe', [1., 0.], 'development')
    store.add('S002', 'take2', 'b'*64, 'recipe', [0., 1.], 'development')
    assert len(store.claim('S001', 'recipe')) == 1
    assert store.claim('S003', 'recipe') == []
    assert store.claim('S001', 'different-recipe') == []
    np.testing.assert_array_equal(store.claim('S001', 'recipe')[0]['embedding'], [1., 0.])
    with pytest.raises(ValueError):
        store.add('S002', 'take3', 'a'*64, 'recipe', [1., 0.], 'development')
    with pytest.raises(ValueError):
        store.add('S001', 'take1', 'c'*64, 'recipe', [1., 0.], 'development')
    assert (tmp_path / 'new.db').stat().st_mode & 0o777 == 0o600


def test_policy_absence_never_becomes_false(tmp_path):
    assert decide(.1, 'recipe', None) == ('POLICY_NOT_READY', None)
    policy = tmp_path / 'policy.json'
    policy.write_text(json.dumps({'threshold': .5, 'ready': True}))
    assert decide(.1, 'recipe', policy) == ('POLICY_NOT_READY', None)


def test_only_bound_calibration_evidence_can_enable_boolean(tmp_path):
    evidence = {'schema': 'nano-calibration-evidence-v1', 'recipe': 'recipe',
                'split_role': 'calibration', 'threshold': .5, 'reference_count': 1,
                'validation_status': 'APPROVED', 'trial_manifest_sha256': 'a'*64,
                'split_sha256': 'b'*64, 'report_sha256': 'c'*64,
                'approved_by': 'operator', 'approved_at': '2026-10-20T12:00:00+00:00'}
    (tmp_path / 'evidence.json').write_text(json.dumps(evidence))
    policy = tmp_path / 'policy.json'
    policy.write_text(json.dumps({'schema': 'nano-policy-v1', 'recipe': 'recipe',
                                 'threshold': .5, 'reference_count': 1, 'evidence': 'evidence.json',
                                 'evidence_sha256': fingerprint(evidence)}))
    assert decide(.5, 'recipe', policy, reference_count=1) == ('OK', True)
    assert decide(.49, 'recipe', policy, reference_count=1) == ('OK', False)
    assert decide(.9, 'recipe', policy, reference_count=2) == ('POLICY_NOT_READY', None)
    assert decide(.49, 'other-pose', policy) == ('POLICY_NOT_READY', None)
    evidence['split_role'] = 'locked_final'
    (tmp_path / 'evidence.json').write_text(json.dumps(evidence))
    assert decide(.1, 'recipe', policy) == ('POLICY_NOT_READY', None)
