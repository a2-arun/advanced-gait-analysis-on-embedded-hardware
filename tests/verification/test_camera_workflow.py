import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.verification.cli import main
from src.verification.registry import RecordingRegistry
from src.verification.store import EnrollmentStore


def setup_workflow(tmp_path, monkeypatch):
    from src.verification import backend, capture
    cfg = tmp_path / 'config.yaml'
    cfg.write_text(yaml.safe_dump(dict(schema_version=2, source_dir='source', bundle_dir='bundle',
        device='cuda', database='data/enroll.db', recordings='data/recordings.json', policy=None,
        events_dir='events', log_file='logs/operator.log', capture_dir='data/raw/camera',
        camera=dict(device_id=0, fps_target=30, resolution=[640, 480]))))
    recordings = []
    def fake_camera(path, camera, seconds, device=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('independent capture ' + str(len(recordings))).encode())
        recordings.append(path)
    class Encoder:
        recipe = 'fixture-camera-recipe'
        def __init__(self, *args):
            pass
        def encode(self, video):
            return np.array([1., 0.])
    monkeypatch.setattr(capture, 'capture_video', fake_camera)
    monkeypatch.setattr(backend, 'runtime_recipe', lambda *args: (Encoder.recipe, {}))
    monkeypatch.setattr(backend, 'LocalGaitPT', Encoder)
    monkeypatch.setattr(backend, 'pose_runtime_smoke', lambda: None)
    return cfg, recordings


def test_camera_enrollment_persists_and_next_command_verifies_claim(tmp_path, monkeypatch, capsys):
    cfg, captures = setup_workflow(tmp_path, monkeypatch)
    assert main(['enroll', '--config', str(cfg), '--name', 'Abhishek', '--consent',
                 '--single-walker-reviewed']) == 0
    enrolled = json.loads(capsys.readouterr().out)
    assert enrolled['status'] == 'ENROLLED'
    assert len(captures) == 1
    # Reopen the file-backed store as the subsequent script invocation will.
    references = EnrollmentStore(tmp_path / 'data/enroll.db').claim('Abhishek', 'fixture-camera-recipe')
    assert len(references) == 1
    np.testing.assert_array_equal(references[0]['embedding'], [1., 0.])
    assert references[0]['role'] == 'operational'
    assert main(['verify', '--config', str(cfg), '--identity', 'Abhishek', '--consent',
                 '--single-walker-reviewed']) == 2
    verified = json.loads(capsys.readouterr().out)
    assert verified['status'] == 'POLICY_NOT_READY' and verified['verdict'] is None
    assert len(captures) == 2 and captures[0] != captures[1]
    rows = RecordingRegistry(tmp_path / 'data/recordings.json').rows()
    assert rows[0]['subject_id'] == 'Abhishek'
    assert rows[1]['subject_id'] is None
    assert rows[1]['purpose'] == 'query'
    assert len(EnrollmentStore(tmp_path / 'data/enroll.db').claim('Abhishek', 'fixture-camera-recipe')) == 1


def test_unknown_claim_does_not_start_camera(tmp_path, monkeypatch, capsys):
    cfg, captures = setup_workflow(tmp_path, monkeypatch)
    assert main(['verify', '--config', str(cfg), '--identity', 'Unenrolled', '--consent',
                 '--single-walker-reviewed']) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'CLAIM_NOT_ENROLLED'
    assert captures == []


def test_model_startup_failure_precedes_camera_recording(tmp_path, monkeypatch, capsys):
    from src.verification import backend
    cfg, captures = setup_workflow(tmp_path, monkeypatch)
    def broken_model(*args):
        raise RuntimeError('weights cannot load')
    monkeypatch.setattr(backend, 'LocalGaitPT', broken_model)
    assert main(['enroll', '--config', str(cfg), '--name', 'Abhishek', '--consent',
                 '--single-walker-reviewed']) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'PROCESSING_ERROR'
    assert captures == []


def test_missing_consent_and_bad_gait_do_not_create_enrollments(tmp_path, monkeypatch, capsys):
    from src.verification import backend
    from src.verification.contract import InsufficientGait
    cfg, captures = setup_workflow(tmp_path, monkeypatch)
    assert main(['enroll', '--config', str(cfg), '--name', 'Abhishek',
                 '--single-walker-reviewed']) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'INVALID_INPUT'
    assert captures == []
    def bad_gait(self, video):
        raise InsufficientGait('missing feet')
    monkeypatch.setattr(backend.LocalGaitPT, 'encode', bad_gait)
    assert main(['enroll', '--config', str(cfg), '--name', 'Abhishek', '--consent',
                 '--single-walker-reviewed']) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'INSUFFICIENT_GAIT'
    assert not EnrollmentStore(tmp_path / 'data/enroll.db').has_claim('Abhishek')


def test_usb_capture_command_respects_settings_and_refuses_overwrite(tmp_path, monkeypatch):
    from src.verification.capture import capture_video
    calls = []
    def camera_process(command, **kwargs):
        calls.append((command, kwargs))
        Path(command[-1]).write_bytes(b'camera output')
    monkeypatch.setattr('subprocess.run', camera_process)
    path = tmp_path / 'take.mp4'
    capture_video(path, dict(device_id=0, fps_target=30, resolution=[640, 480]), 30)
    assert '/dev/video0' in calls[0][0]
    assert '640x480' in calls[0][0] and '-n' in calls[0][0]
    assert calls[0][1]['timeout'] <= 60
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError):
        capture_video(path, dict(device_id=0), 30)
    assert len(calls) == 1


def test_camera_preflight_decodes_real_frames_and_removes_its_temporary_clip(monkeypatch):
    from src.verification import capture
    from src.verification.environment import camera_decode_smoke
    recordings = []
    def synthetic_camera(path, *args):
        recordings.append(path)
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                        'testsrc2=size=64x48:rate=30', '-t', '2', '-c:v', 'mpeg4',
                        '-threads', '1', str(path)], check=True)
    monkeypatch.setattr(capture, 'capture_video', synthetic_camera)
    result = camera_decode_smoke({})
    assert result['decoded_frames'] == 60
    assert result['temporary_recording_deleted'] is True
    assert not recordings[0].exists()


def test_optional_original_video_enrollment_and_reference_reuse_guard(tmp_path, monkeypatch, capsys):
    cfg, captures = setup_workflow(tmp_path, monkeypatch)
    video = tmp_path / 'original.mp4'
    video.write_bytes(b'fixture original')
    flags = ['--config', str(cfg), '--consent', '--register-original', '--single-walker-reviewed']
    assert main(['enroll', '--name', 'Abhishek', '--video', str(video)] + flags) == 0
    assert json.loads(capsys.readouterr().out)['reference_count'] == 1
    assert main(['verify', '--identity', 'Abhishek', '--video', str(video)] + flags) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'INVALID_INPUT'
    probe = tmp_path / 'probe.mp4'
    probe.write_bytes(b'different original fixture')
    assert main(['verify', '--identity', 'Abhishek', '--video', str(probe)] + flags) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'POLICY_NOT_READY'
    assert captures == []
    rows = RecordingRegistry(tmp_path / 'data/recordings.json').rows()
    assert len(rows) == 2 and rows[1]['subject_id'] is None
    assert rows[0]['captured_at'] is None  # Unknown original capture date is not invented.
