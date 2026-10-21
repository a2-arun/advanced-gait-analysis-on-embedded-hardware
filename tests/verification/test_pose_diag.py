import ast
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scripts.make_parity_fixture import synthetic_coco17

from src.verification import pose_diag
from src.verification.camera_diag import analyze_video
from src.verification.parity import coco17_to_mediapipe33
from src.verification.pose_diag import (POSE_BACKENDS, check_backend, check_named_backend,
                                        extract_pose_report, load_backend)

pytestmark = pytest.mark.skipif(not shutil.which('ffmpeg'), reason='ffmpeg required')


class FakePose:
    """Scripted landmarks: a deterministic synthetic walker, not a pose estimator."""
    name = 'fake-synthetic-pose'
    layout = 'mediapipe33'

    def __init__(self, missing=()):
        coco, _ = synthetic_coco17(90, 1)
        self.xy = coco17_to_mediapipe33(coco)
        self.missing, self.index, self.closed = set(missing), 0, False

    def describe(self):
        return dict(name=self.name, version='0-test', layout=self.layout, settings={}, asset_sha256=[])

    def process(self, bgr):
        index, self.index = self.index, self.index + 1
        if index in self.missing:
            return None
        return self.xy[index], np.full(33, .95, np.float32)

    def close(self):
        self.closed = True


def clip(path, seconds=3):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc2=size=640x480:rate=30',
                    '-t', str(seconds), '-c:v', 'mpeg4', '-threads', '1', str(path)], check=True)
    return Path(path)


def test_complete_detection_reports_the_exact_backend_and_the_provisional_gate(tmp_path):
    backend = FakePose()
    report = extract_pose_report(clip(tmp_path / 'walk.mp4'), backend)
    assert report['backend']['name'] == 'fake-synthetic-pose' and report['backend']['version'] == '0-test'
    assert (report['frames_total'], report['frames_detected']) == (90, 90)
    assert report['detection_rate'] == 1.0 and report['longest_full_body_run'] == 90
    assert report['provisional_gate']['result'] == 'PASS'
    assert report['joint_visibility']['left_ankle']['mean_visibility'] == pytest.approx(.95)
    assert report['joint_visibility']['left_ankle']['in_frame_fraction'] == 1.0
    assert backend.closed is True


def test_missing_detections_are_reported_not_filled(tmp_path):
    backend = FakePose(missing=range(0, 90, 2))
    report = extract_pose_report(clip(tmp_path / 'walk.mp4'), backend)
    assert report['frames_detected'] == 45 and report['detection_rate'] == 0.5
    assert report['longest_full_body_run'] == 1
    assert report['provisional_gate']['result'] == 'INSUFFICIENT_GAIT'
    assert report['provisional_gate']['reason']


def test_report_never_claims_anatomical_accuracy_or_identity(tmp_path):
    report = extract_pose_report(clip(tmp_path / 'walk.mp4'), FakePose())
    assert report['anatomical_accuracy'] == 'NOT_ASSESSED'
    assert 'not' in report['interpretation'].lower() and 'confidence' in report['interpretation'].lower()
    text = json.dumps(report).lower()
    for forbidden in ('embedding', 'subject', 'identity', 'verdict'):
        assert forbidden not in text
    assert report['video']['sha256'] and report['video']['path'].endswith('walk.mp4')


def test_overlay_is_a_separate_decodable_video_and_never_the_source(tmp_path):
    video = clip(tmp_path / 'walk.mp4')
    overlay = tmp_path / 'out' / 'overlay.mp4'
    report = extract_pose_report(video, FakePose(), overlay=overlay)
    assert report['overlay']['status'] == 'PASS'
    analysis = analyze_video(overlay)
    assert analysis['frames_decoded'] == 90 and analysis['playable'] is True
    before = video.read_bytes()
    with pytest.raises(ValueError):
        extract_pose_report(video, FakePose(), overlay=video)
    assert video.read_bytes() == before
    with pytest.raises(ValueError):
        extract_pose_report(video, FakePose(), overlay=overlay)  # never overwrite evidence


def test_unreadable_video_is_an_error_not_a_pose_report(tmp_path):
    bad = tmp_path / 'bad.mp4'
    bad.write_bytes(b'junk' * 100)
    with pytest.raises(ValueError):
        extract_pose_report(bad, FakePose())


def test_backend_check_loads_and_reports_failures(monkeypatch):
    assert check_backend(FakePose())['status'] == 'PASS'

    class Broken(FakePose):
        def describe(self):
            raise RuntimeError('asset missing')
    result = check_backend(Broken())
    assert result['status'] == 'FAIL' and 'asset missing' in result['detail']


def test_unknown_backend_names_are_refused_and_the_registry_is_explicit():
    assert 'mediapipe-solutions-single-person-lite' in POSE_BACKENDS
    with pytest.raises(ValueError) as caught:
        load_backend('rtmpose-m')
    assert 'mediapipe-solutions-single-person-lite' in str(caught.value)


def test_installed_mediapipe_is_only_accepted_when_it_is_the_nano_recipe_version():
    pytest.importorskip('mediapipe')
    import mediapipe
    result = check_named_backend('mediapipe-solutions-single-person-lite')
    if mediapipe.__version__ != '0.8.5':
        assert result['status'] == 'FAIL' and '0.8.5' in result['detail']


def test_capture_storage_and_diagnostics_do_not_depend_on_the_pose_backend():
    root = Path(pose_diag.__file__).parent
    for name in ('capture', 'store', 'dbcheck', 'registry', 'camera_diag', 'model_slot', 'policy'):
        imported = set()
        for node in ast.walk(ast.parse((root / (name + '.py')).read_text())):
            if isinstance(node, ast.ImportFrom):
                imported.add((node.module or '') + ' ' + ' '.join(a.name for a in node.names))
            elif isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
        assert not any('pose' in i or 'mediapipe' in i for i in imported), (name, imported)
