import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from lab_helpers import allow_development_model

from src.verification.cli import main
from src.verification.contract import InsufficientGait
from src.verification.contract import fingerprint
from src.verification.media import inspect_video, video_frames
from src.verification.registry import RecordingRegistry
from src.verification.store import EnrollmentStore


def config(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(dict(schema_version=2, source_dir='missing-source',
        bundle_dir='missing-bundle', database='data/enroll.db', recordings='data/recordings.json',
        events_dir='events', log_file='logs/operator.log', device='cuda', policy=None)))
    return path


def test_invalid_claim_returns_invalid_input_before_model_setup(tmp_path, capsys):
    cfg = config(tmp_path)
    code = main(['verify', '--config', str(cfg), '--video', 'missing.mp4',
                 '--claimed-id', '../bad', '--single-walker-reviewed'])
    result = json.loads(capsys.readouterr().out)
    assert code == 2
    assert result['status'] == 'INVALID_INPUT'
    assert result['verdict'] is None


def test_missing_claim_returns_unenrolled_without_weights_or_pose_installed(tmp_path, capsys):
    cfg = config(tmp_path)
    video = tmp_path / 'original.mp4'
    video.write_bytes(b'registered original')
    registry = RecordingRegistry(tmp_path / 'data/recordings.json')
    registry.register(video, 'query1', 'S001', 'session1', 'toward', 'development',
                      'consent001', '2026-10-20T12:00:00+05:30')
    code = main(['verify', '--config', str(cfg), '--video', str(video),
                 '--claimed-id', 'S999', '--single-walker-reviewed'])
    result = json.loads(capsys.readouterr().out)
    assert code == 2
    assert result['status'] == 'CLAIM_NOT_ENROLLED'
    events = list((tmp_path / 'events').glob('*.json'))
    assert len(events) == 1
    assert json.loads(events[0].read_text()) == result
    assert events[0].stat().st_mode & 0o777 == 0o600


def test_operator_log_cannot_mutate_a_registered_original(tmp_path, capsys):
    cfg = config(tmp_path)
    video = tmp_path / 'original.mp4'
    video.write_bytes(b'original must stay immutable')
    registry = RecordingRegistry(tmp_path / 'data/recordings.json')
    registry.register(video, 'query1', 'S001', 'session1', 'toward', 'development',
                      'consent001', '2026-10-20T12:00:00+05:30')
    values = yaml.safe_load(cfg.read_text())
    values['log_file'] = str(video)
    cfg.write_text(yaml.safe_dump(values))
    store = EnrollmentStore(tmp_path / 'data/enroll.db')
    store.add('S001', 'ref1', 'a'*64, 'old-recipe', [1., 0.], 'development')
    before = video.read_bytes()
    assert main(['verify', '--config', str(cfg), '--video', str(video),
                 '--claimed-id', 'S001', '--single-walker-reviewed']) == 2
    assert video.read_bytes() == before
    assert json.loads(capsys.readouterr().out)['verdict'] is None


def make_video(path, duration='1.1'):
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi', '-i',
                    'testsrc2=size=64x48:rate=30', '-t', duration, '-c:v', 'mpeg4',
                    '-threads', '1', str(path)], check=True)


def test_actual_decode_keeps_every_source_pts_and_handles_rotation(tmp_path):
    raw = tmp_path / 'raw.mp4'
    make_video(raw)
    width, height, pts = inspect_video(raw)
    frames = list(video_frames(raw, width, height, pts))
    assert len(frames) == len(pts) == 33
    assert frames[0][1].shape == (48, 64, 3)
    np.testing.assert_allclose([frame[0] for frame in frames], pts)
    rotated = tmp_path / 'rotated.mp4'
    help_text = subprocess.check_output(['ffmpeg', '-hide_banner', '-h', 'full'],
                                        stderr=subprocess.DEVNULL)
    if b'-display_rotation' in help_text:
        command = ['ffmpeg', '-nostdin', '-v', 'error', '-display_rotation', '90',
                   '-i', str(raw), '-c', 'copy', str(rotated)]
    else:
        command = ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(raw), '-c', 'copy',
                   '-metadata:s:v:0', 'rotate=90', str(rotated)]
    subprocess.run(command, check=True)
    width, height, rotated_pts = inspect_video(rotated)
    frames = list(video_frames(rotated, width, height, rotated_pts))
    assert frames[0][1].shape == (64, 48, 3)
    np.testing.assert_allclose(rotated_pts, pts)


def test_short_video_is_insufficient_gait(tmp_path):
    raw = tmp_path / 'short.mp4'
    make_video(raw, duration='0.5')
    with pytest.raises(InsufficientGait):
        inspect_video(raw)


def test_identify_entry_point_requires_explicit_claim_and_exposes_verification_help():
    root = Path(__file__).resolve().parents[2]
    help_run = subprocess.run([sys.executable, str(root / 'scripts/identify.py'), '--help'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert help_run.returncode == 0
    assert b'--identity' in help_run.stdout and b'--claimed-id' in help_run.stdout
    run = subprocess.run([sys.executable, str(root / 'scripts/identify.py')],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert run.returncode == 2
    assert b'--claimed-id' in run.stderr
    assert not run.stdout


@pytest.mark.parametrize('existing_output', [False, True])
def test_headless_camera_preview_creates_private_snapshot_without_overwriting(tmp_path, monkeypatch, existing_output):
    root = Path(__file__).resolve().parents[2]
    namespace = runpy.run_path(str(root / 'scripts/test_camera.py'))
    class Camera:
        def __init__(self, cfg):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def frames(self):
            yield np.zeros((24, 32, 3), dtype=np.uint8)
    monkeypatch.setitem(namespace['main'].__globals__, 'CameraManager', Camera)
    monkeypatch.setattr(sys, 'argv', ['test_camera.py', '--no-display'])
    monkeypatch.chdir(tmp_path)
    if existing_output:
        (tmp_path / 'outputs').mkdir()
        (tmp_path / 'outputs/camera_test.jpg').write_bytes(b'preserve user snapshot')
    assert namespace['main']() == 0
    images = list((tmp_path / 'outputs').glob('camera_test-*.jpg'))
    assert len(images) == 1
    assert images[0].stat().st_mode & 0o777 == 0o600
    if existing_output:
        assert (tmp_path / 'outputs/camera_test.jpg').read_bytes() == b'preserve user snapshot'


def test_cli_enrollment_query_and_private_diagnostics_with_synthetic_backend(tmp_path, monkeypatch, capsys):
    # Camera/model boundary is a fixture; all registry, store, policy and event code is real.
    from src.verification import backend
    recipe = 'synthetic-cli-recipe'
    class Encoder:
        def __init__(self, source, bundle, value, device):
            self.recipe = value
        def encode(self, video):
            return np.array([0., 1.]) if Path(video).name == 'ref2.mp4' else np.array([1., 0.])
    allow_development_model(monkeypatch)
    monkeypatch.setattr(backend, 'runtime_recipe', lambda *args: (recipe, {}))
    monkeypatch.setattr(backend, 'LocalGaitPT', Encoder)
    cfg = config(tmp_path)
    raw = tmp_path / 'raw'
    raw.mkdir()
    for name, subject in [('ref1', 'S001'), ('ref2', 'S002'), ('probe', 'S001')]:
        video = raw / (name + '.mp4')
        video.write_bytes(name.encode())
        assert main(['register', '--config', str(cfg), '--video', str(video),
                     '--subject-id', subject, '--take-id', name, '--session-id', 'day1',
                     '--view', 'toward', '--role', 'development', '--consent-ref', 'consent001',
                     '--captured-at', '2026-10-20T12:00:00+05:30']) == 0
        capsys.readouterr()
        if name != 'probe':
            assert main(['enroll', '--config', str(cfg), '--video', str(video),
                         '--subject-id', subject, '--single-walker-reviewed']) == 0
            assert json.loads(capsys.readouterr().out)['status'] == 'ENROLLED'
    for claim in ('S001', 'S002'):
        assert main(['verify', '--config', str(cfg), '--video', str(raw / 'probe.mp4'),
                     '--claimed-id', claim, '--single-walker-reviewed',
                     '--private-diagnostics-dir', str(tmp_path / 'private')]) == 2
        result = json.loads(capsys.readouterr().out)
        assert result['status'] == 'POLICY_NOT_READY' and result['verdict'] is None
        assert 'raw_similarity' not in result
    diagnostics = [json.loads(p.read_text()) for p in (tmp_path / 'private').glob('*.json')]
    assert sorted(record['raw_similarity'] for record in diagnostics) == [0., 1.]
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in (tmp_path / 'private').glob('*.json'))
    evidence = dict(schema='nano-calibration-evidence-v1', recipe=recipe, threshold=.5,
                    reference_count=1, split_role='calibration', validation_status='APPROVED',
                    trial_manifest_sha256='a'*64, split_sha256='b'*64, report_sha256='c'*64,
                    approved_by='synthetic-fixture', approved_at='2026-10-20T12:00:00+00:00')
    (tmp_path / 'evidence.json').write_text(json.dumps(evidence))
    (tmp_path / 'policy.json').write_text(json.dumps(dict(schema='nano-policy-v1', recipe=recipe,
        threshold=.5, reference_count=1, evidence='evidence.json', evidence_sha256=fingerprint(evidence))))
    values = yaml.safe_load(cfg.read_text())
    values['policy'] = 'policy.json'
    cfg.write_text(yaml.safe_dump(values))
    for claim, expected in [('S001', True), ('S002', False)]:
        assert main(['verify', '--config', str(cfg), '--video', str(raw / 'probe.mp4'),
                     '--claimed-id', claim, '--single-walker-reviewed']) == 0
        result = json.loads(capsys.readouterr().out)
        assert result['status'] == 'OK' and result['verdict'] is expected
        assert 'raw_similarity' not in result
    assert len(list((tmp_path / 'events').glob('*.json'))) == 4
