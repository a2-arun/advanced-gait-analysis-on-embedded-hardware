import json
import io
import zipfile
import numpy as np
import pytest

from src.verification.artifacts import load_arrays, prepare_bundle
from src.verification.contract import sha256_file, fingerprint, InsufficientGait
from src.verification.registry import RecordingRegistry
from src.verification.service import Verifier
from src.verification.store import EnrollmentStore


def register(registry, path, take, subject='S001', role='development'):
    registry.register(path, take, subject, 'session1', 'left45', role,
                      'consent-private-001', '2026-10-20T12:00:00+00:00')


def test_npz_pickle_arrays_are_refused(tmp_path):
    path = tmp_path / 'bad.npz'
    np.savez(path, a=np.array([{'code': 'never execute'}], dtype=object))
    with pytest.raises(ValueError):
        load_arrays(path)


def test_npz_nonfinite_values_refused(tmp_path):
    path = tmp_path / 'bad.npz'
    np.savez(path, a=np.array([float('nan')], dtype=np.float32))
    with pytest.raises(ValueError):
        load_arrays(path)


def test_untrusted_array_shape_cannot_allocate_beyond_the_archive(tmp_path, monkeypatch):
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, dict(shape=(100000000,),
        fortran_order=False, descr='<f4'))
    path = tmp_path / 'shape-bomb.npz'
    with zipfile.ZipFile(str(path), 'w') as archive:
        archive.writestr('tensor.npy', header.getvalue())
    def refuse_allocation(*args, **kwargs):
        pytest.fail('Malformed tensor header reached array allocation')
    monkeypatch.setattr(np, 'ndarray', refuse_allocation)
    with pytest.raises(ValueError):
        load_arrays(path)


def test_export_refuses_unpinned_checkpoint_before_deserializing(tmp_path):
    path = tmp_path / 'fake.pth'
    path.write_bytes(b'not a checkpoint')
    with pytest.raises(ValueError):
        prepare_bundle(path, tmp_path / 'bundle')
    assert not (tmp_path / 'bundle').exists()


def test_registry_checks_hashes_roles_ownership_and_unmanaged_copy(tmp_path):
    path = tmp_path / 'raw.mp4'
    path.write_bytes(b'original bytes')
    registry = RecordingRegistry(tmp_path / 'registry.json')
    register(registry, path, 'take1')
    assert registry.resolve(path)['original_take_id'] == 'take1'
    copy = tmp_path / 'other.mp4'
    copy.write_bytes(b'changed source')
    with pytest.raises(ValueError):
        registry.resolve(copy)
    with pytest.raises(ValueError):
        register(registry, copy, 'take2', role='calibration')
    path.write_bytes(b'tampered')
    with pytest.raises(ValueError):
        registry.resolve(path)


class Encoder:
    recipe = 'test-recipe'
    def encode(self, video):
        return np.array([1., 0.], dtype=np.float32)


def test_verifier_claim_lookup_precedes_model_and_never_discloses_others(tmp_path):
    registry = RecordingRegistry(tmp_path / 'registry.json')
    probe, ref = tmp_path / 'probe', tmp_path / 'ref'
    probe.write_bytes(b'probe')
    ref.write_bytes(b'reference')
    register(registry, probe, 'query1')
    register(registry, ref, 'ref1', subject='S002')
    store = EnrollmentStore(tmp_path / 'enroll.db')
    store.add('S002', 'ref1', sha256_file(ref), Encoder.recipe, [0., 1.], 'development')
    def must_not_load():
        raise AssertionError('Model should not load for missing claim')
    verifier = Verifier(store, registry, Encoder.recipe, must_not_load)
    result = verifier.verify(probe, 'S003')
    assert result['status'] == 'CLAIM_NOT_ENROLLED'
    assert result['verdict'] is None
    verifier = Verifier(store, registry, Encoder.recipe, Encoder)
    result = verifier.verify(probe, 'S002')
    assert result['status'] == 'POLICY_NOT_READY'
    assert set(result) == {'schema_version', 'request_id', 'claimed_subject_id', 'status', 'verdict', 'message'}
    assert 'similarity' not in json.dumps(result)
    assert verifier.verify(ref, 'S002')['status'] == 'INVALID_INPUT'
    assert verifier.verify(probe, '../S002')['status'] == 'INVALID_INPUT'


@pytest.mark.parametrize('error,status', [(InsufficientGait('bad feet'), 'INSUFFICIENT_GAIT'),
                                        (RuntimeError('CUDA broken'), 'PROCESSING_ERROR')])
def test_failed_extraction_never_rejects_identity(tmp_path, error, status):
    registry = RecordingRegistry(tmp_path / 'registry.json')
    probe = tmp_path / 'probe'
    probe.write_bytes(b'probe')
    register(registry, probe, 'query1')
    store = EnrollmentStore(tmp_path / 'enroll.db')
    ref = tmp_path / 'ref'
    ref.write_bytes(b'reference')
    register(registry, ref, 'ref1')
    store.add('S001', 'ref1', sha256_file(ref), Encoder.recipe, [1., 0.], 'development')
    class Broken(Encoder):
        def encode(self, video):
            raise error
    result = Verifier(store, registry, Encoder.recipe, Broken).verify(probe, 'S001')
    assert result['status'] == status
    assert result['verdict'] is None


@pytest.mark.parametrize('query_vector,verdict', [([1., 0.], True), ([0., 1.], False)])
def test_calibrated_fixture_boolean_and_private_score_are_separate(tmp_path, query_vector, verdict):
    # This is an explicitly synthetic policy, never installed in the repository.
    evidence = dict(schema='nano-calibration-evidence-v1', recipe=Encoder.recipe,
                    split_role='calibration', threshold=.5, reference_count=1, validation_status='APPROVED',
                    trial_manifest_sha256='a'*64, split_sha256='b'*64, report_sha256='c'*64,
                    approved_by='synthetic-fixture', approved_at='2026-10-20T12:00:00+00:00')
    (tmp_path / 'evidence.json').write_text(json.dumps(evidence))
    policy = tmp_path / 'policy.json'
    policy.write_text(json.dumps(dict(schema='nano-policy-v1', recipe=Encoder.recipe,
        threshold=.5, reference_count=1, evidence='evidence.json', evidence_sha256=fingerprint(evidence))))
    registry = RecordingRegistry(tmp_path / 'registry.json')
    store = EnrollmentStore(tmp_path / 'enroll.db')
    for name in ('probe', 'ref'):
        path = tmp_path / name
        path.write_bytes(name.encode())
        register(registry, path, name, role='calibration')
    store.add('S001', 'ref', sha256_file(tmp_path / 'ref'), Encoder.recipe, [1., 0.], 'calibration')
    class Query(Encoder):
        def encode(self, video):
            return np.asarray(query_vector)
    private = []
    verifier = Verifier(store, registry, Encoder.recipe, Query, policy,
                        diagnostic_sink=private.append)
    result = verifier.verify(tmp_path / 'probe', 'S001')
    assert result['status'] == 'OK'
    assert result['verdict'] is verdict
    assert 'raw_similarity' not in result
    assert len(private) == 1
    assert private[0]['raw_similarity'] == (1. if verdict else 0.)
    assert private[0]['reference_take_ids'] == ['ref']
    assert private[0]['recipe'] == Encoder.recipe
    extra = tmp_path / 'extra'
    extra.write_bytes(b'another independent original')
    register(registry, extra, 'extra', role='calibration')
    store.add('S001', 'extra', sha256_file(extra), Encoder.recipe, [1., 0.], 'calibration')
    assert verifier.verify(tmp_path / 'probe', 'S001')['status'] == 'POLICY_NOT_READY'
