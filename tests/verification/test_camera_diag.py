import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.verification import camera_diag, capture
from src.verification.camera_diag import (analyze_video, list_cameras, parse_formats_ext,
                                          record_diagnostic, validate_record_args)

pytestmark = pytest.mark.skipif(not shutil.which('ffmpeg'), reason='ffmpeg required')

SAMPLE = """ioctl: VIDIOC_ENUM_FMT
\tType: Video Capture

\t[0]: 'MJPG' (Motion-JPEG, compressed)
\t\tSize: Discrete 1280x720
\t\t\tInterval: Discrete 0.033s (30.000 fps)
\t\t\tInterval: Discrete 0.050s (20.000 fps)
\t\tSize: Discrete 640x480
\t\t\tInterval: Discrete 0.033s (30.000 fps)
\t[1]: 'YUYV' (YUYV 4:2:2)
\t\tSize: Discrete 640x480
\t\t\tInterval: Discrete 0.033s (30.000 fps)
"""


def synth(path, seconds=3, source='testsrc2', extra=()):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    source + (':' if '=' in source else '=') + 'size=160x120:rate=30', '-t', str(seconds)] + list(extra)
                   + ['-c:v', 'mpeg4', '-threads', '1', str(path)], check=True)
    return Path(path)


def codes(report):
    return [f['code'] for f in report['findings']]


def test_parse_formats_ext_lists_pixel_formats_sizes_and_rates():
    modes = parse_formats_ext(SAMPLE)
    assert modes[0] == dict(format='MJPG', width=1280, height=720, fps=[30.0, 20.0])
    assert dict(format='YUYV', width=640, height=480, fps=[30.0]) in modes
    assert len(modes) == 3
    assert parse_formats_ext('') == []


def test_list_cameras_without_v4l2_ctl_reports_modes_not_tested(monkeypatch, tmp_path):
    node = tmp_path / 'video0'
    node.write_bytes(b'')
    monkeypatch.setattr(camera_diag.shutil, 'which', lambda name: None)
    result = list_cameras(nodes=[str(node)])
    assert result['nodes'][0]['path'] == str(node)
    assert result['modes_status'] == 'NOT_TESTED' and result['nodes'][0]['modes'] == []


def test_list_cameras_uses_v4l2_ctl_for_modes(monkeypatch, tmp_path):
    node = tmp_path / 'video0'
    node.write_bytes(b'')
    monkeypatch.setattr(camera_diag.shutil, 'which', lambda name: '/usr/bin/v4l2-ctl')
    monkeypatch.setattr(camera_diag, 'run_text', lambda command, timeout=10: SAMPLE)
    result = list_cameras(nodes=[str(node)])
    assert result['modes_status'] == 'PASS' and len(result['nodes'][0]['modes']) == 3


def test_good_clip_is_playable_decodable_and_measured(tmp_path):
    report = analyze_video(synth(tmp_path / 'good.mp4'), expected=dict(width=160, height=120, fps=30))
    assert report['status'] == 'PASS' and report['playable'] is True
    assert (report['width'], report['height']) == (160, 120)
    assert report['frames_probed'] == report['frames_decoded'] == 90
    assert report['measured_fps'] == pytest.approx(30, abs=1)
    assert report['max_gap_s'] == pytest.approx(1 / 30, abs=0.01)
    assert report['findings'] == []


def test_resolution_and_rate_mismatch_are_warnings(tmp_path):
    report = analyze_video(synth(tmp_path / 'good.mp4'), expected=dict(width=640, height=480, fps=60))
    assert report['status'] == 'WARNING'
    assert 'RESOLUTION_MISMATCH' in codes(report) and 'LOW_FPS' in codes(report)


def test_short_black_and_frozen_clips_are_flagged(tmp_path):
    short = analyze_video(synth(tmp_path / 'short.mp4', seconds=.3))
    assert 'SHORT_RECORDING' in codes(short)
    black = analyze_video(synth(tmp_path / 'black.mp4', source='color=c=black'))
    assert 'NEAR_BLACK' in codes(black) and black['status'] == 'WARNING'
    frozen = analyze_video(synth(tmp_path / 'frozen.mp4', source='color=c=gray'))
    assert 'FROZEN_IMAGE' in codes(frozen)


def test_timestamp_gaps_are_reported(tmp_path):
    clip = tmp_path / 'gap.mkv'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=160x120:rate=30',
                    '-t', '4', '-vf', "select='lt(n,30)+gt(n,60)'", '-vsync', 'vfr', '-c:v', 'mpeg4',
                    '-threads', '1', str(clip)], check=True)
    report = analyze_video(clip)
    assert 'TIMESTAMP_GAPS' in codes(report) and report['max_gap_s'] > 0.9


def test_unplayable_and_empty_files_fail_without_raising(tmp_path):
    garbage = tmp_path / 'garbage.mp4'
    garbage.write_bytes(b'not a video' * 100)
    empty = tmp_path / 'empty.mp4'
    empty.write_bytes(b'')
    for path in (garbage, empty, tmp_path / 'missing.mp4'):
        report = analyze_video(path)
        assert report['status'] == 'FAIL' and report['playable'] is False
        assert 'UNPLAYABLE' in codes(report)


@pytest.mark.parametrize('kwargs', [
    dict(seconds=0), dict(seconds=31), dict(seconds='5'), dict(seconds=True),
    dict(device='/etc/passwd'), dict(device='/dev/video0;reboot'), dict(device='video0'),
    dict(resolution=[0, 480]), dict(resolution=[4000, 480]), dict(fps=0), dict(fps=500)])
def test_record_arguments_are_validated_before_the_camera_is_touched(tmp_path, monkeypatch, kwargs):
    monkeypatch.setattr(capture, 'capture_video', lambda *a, **k: pytest.fail('camera touched'))
    cfg = dict(diagnostics_dir=str(tmp_path / 'diag'), capture_dir=str(tmp_path / 'raw'),
               camera=dict(device_id=0, fps_target=30, resolution=[640, 480]))
    arguments = dict(seconds=2, device='/dev/video0', resolution=None, fps=None)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        record_diagnostic(cfg, **arguments)


def test_valid_arguments_normalize():
    assert validate_record_args(5, '/dev/video1', [640, 480], 30) == (5, '/dev/video1', (640, 480), 30)
    assert validate_record_args(5, 2, None, None, default_camera=dict(device_id=0))[1] == '/dev/video2'


def test_diagnostic_is_separate_from_originals_and_registries(tmp_path, monkeypatch):
    calls = []

    def fake_camera(path, camera, seconds, device=None):
        calls.append((Path(path), camera, seconds, device))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        synth(path, seconds=2)
    monkeypatch.setattr(capture, 'capture_video', fake_camera)
    cfg = dict(diagnostics_dir=str(tmp_path / 'data/diagnostics'), capture_dir=str(tmp_path / 'data/raw'),
               recordings=str(tmp_path / 'data/recordings.json'),
               camera=dict(device_id=0, fps_target=30, resolution=[160, 120]))
    report = record_diagnostic(cfg, seconds=2, device='/dev/video0')
    video = Path(report['video'])
    assert video.parent == tmp_path / 'data/diagnostics/camera' and video.is_file()
    assert report['purpose'] == 'engineering-diagnostic-not-a-biometric-record'
    assert report['analysis']['status'] == 'PASS' and report['camera_released'] is True
    sidecar = json.loads(Path(report['report']).read_text())
    assert sidecar['analysis']['frames_decoded'] == 60
    assert Path(report['report']).stat().st_mode & 0o777 == 0o600
    assert not (tmp_path / 'data/raw').exists() and not (tmp_path / 'data/recordings.json').exists()
    assert len(calls) == 1 and calls[0][2] == 2


def test_output_inside_the_originals_folder_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, 'capture_video', lambda *a, **k: pytest.fail('camera touched'))
    cfg = dict(diagnostics_dir=str(tmp_path / 'raw' / 'camera'), capture_dir=str(tmp_path / 'raw'),
               camera=dict(device_id=0, fps_target=30, resolution=[160, 120]))
    with pytest.raises(ValueError):
        record_diagnostic(cfg, seconds=2, device='/dev/video0')


@pytest.mark.parametrize('error', [subprocess.CalledProcessError(1, 'ffmpeg'),
                                   subprocess.TimeoutExpired('ffmpeg', 30)])
def test_capture_errors_become_a_failed_report_and_leave_no_empty_file(tmp_path, monkeypatch, error):
    def broken(path, *a, **k):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b'')
        raise error
    monkeypatch.setattr(capture, 'capture_video', broken)
    cfg = dict(diagnostics_dir=str(tmp_path / 'diag'), capture_dir=str(tmp_path / 'raw'),
               camera=dict(device_id=0, fps_target=30, resolution=[160, 120]))
    report = record_diagnostic(cfg, seconds=2, device='/dev/video0')
    assert report['analysis']['status'] == 'FAIL' and 'CAPTURE_FAILED' in codes(report['analysis'])
    assert not list((tmp_path / 'diag' / 'camera').glob('*.mp4'))
