"""Model-independent camera diagnostics. Recording happens only when the operator runs
`record`; clips stay under diagnostics_dir/camera and are never registered as originals."""
import json
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import capture
from .media import video_frames
from .readiness import run_text

SCHEMA = 'nano-camera-diagnostic-v1'
PURPOSE = 'engineering-diagnostic-not-a-biometric-record'
MAX_SECONDS = 30
GATE_FRAMES, GATE_GAP_S = 31, 0.1  # provisional Nano quality-gate numbers (contract.py)
DEVICE = re.compile(r'/dev/video\d{1,3}')


def parse_formats_ext(text):
    """Flatten `v4l2-ctl --list-formats-ext` into (format, size) rows with their frame rates."""
    modes, fmt = [], None
    for line in text.splitlines():
        match = re.search(r"\[\d+\]: '(\w+)'", line)
        if match:
            fmt = match.group(1)
        size = re.search(r'Size: Discrete (\d+)x(\d+)', line)
        if size and fmt:
            modes.append(dict(format=fmt, width=int(size.group(1)), height=int(size.group(2)), fps=[]))
        rate = re.search(r'\((\d+(?:\.\d+)?) fps\)', line)
        if rate and modes:
            modes[-1]['fps'].append(float(rate.group(1)))
    return modes


def list_cameras(nodes=None):
    """Device nodes and, when v4l2-ctl exists, their pixel formats/modes. No frames are read."""
    import glob
    nodes = sorted(glob.glob('/dev/video*')) if nodes is None else list(nodes)
    have_tool = bool(shutil.which('v4l2-ctl'))
    found = []
    for node in nodes:
        modes = parse_formats_ext(run_text(['v4l2-ctl', '-d', node, '--list-formats-ext'])) \
            if have_tool else []
        found.append(dict(path=str(node), accessible=os.access(str(node), os.R_OK | os.W_OK),
                          modes=modes))
    return dict(schema=SCHEMA, nodes=found, modes_status='PASS' if have_tool else 'NOT_TESTED',
                note=None if have_tool else 'v4l2-ctl missing (apt package v4l-utils): modes not listed')


def validate_record_args(seconds, device, resolution, fps, default_camera=None):
    default_camera = default_camera or {}
    if type(seconds) is not int or not 1 <= seconds <= MAX_SECONDS:
        raise ValueError('Diagnostic duration must be an integer 1-{} seconds'.format(MAX_SECONDS))
    device = default_camera.get('device_id', 0) if device is None else device
    if isinstance(device, int) and not isinstance(device, bool) or str(device).isdigit():
        device = '/dev/video' + str(device)
    if not isinstance(device, str) or not DEVICE.fullmatch(device):
        raise ValueError('Camera must be a /dev/videoN device node')
    width, height = resolution or default_camera.get('resolution', [640, 480])
    fps = default_camera.get('fps_target', 30) if fps is None else fps
    if (type(width) is not int or type(height) is not int or type(fps) is not int
            or not (1 <= width <= 1920 and 1 <= height <= 1080 and 1 <= fps <= 60)):
        raise ValueError('Unsupported camera resolution or frame rate')
    return seconds, device, (width, height), fps


def finding(level, code, message):
    return dict(level=level, code=code, message=message)


def unplayable(path, message, extra=None):
    report = dict(path=str(path), status='FAIL', playable=False, findings=[
        finding('ERROR', 'UNPLAYABLE', message)])
    report.update(extra or {})
    return report


def probe(path):
    done = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                           'stream=codec_name,width,height,r_frame_rate:frame=best_effort_timestamp_time',
                           '-of', 'json', str(path)], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=60, check=True)
    return json.loads(done.stdout.decode('utf8'))


def sample_luma(path, width, height, pts):
    """(frames decoded, mean luma of every 5th frame, whether all sampled frames are identical)."""
    count, means, first, identical = 0, [], None, True
    for index, (_, frame) in enumerate(video_frames(path, width, height, pts)):
        count += 1
        if index % 5 == 0:
            small = frame[::8, ::8]
            means.append(float(small.mean()))
            first = small.copy() if first is None else first
            identical = identical and np.array_equal(first, small)
    return count, means, identical and len(means) >= 3


def analyze_video(path, expected=None):
    """Playability, decodability, geometry, timestamps and obvious capture faults."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return unplayable(path, 'file is missing or empty')
    try:
        value = probe(path)
        stream = value['streams'][0]
        width, height = int(stream['width']), int(stream['height'])
        pts = np.array([float(f['best_effort_timestamp_time']) for f in value['frames']])
    except (subprocess.SubprocessError, OSError, ValueError, KeyError, IndexError, TypeError):
        return unplayable(path, 'ffprobe cannot read a video stream with frame timestamps')
    findings = []
    report = dict(path=str(path), bytes=path.stat().st_size, codec=stream.get('codec_name'),
                  width=width, height=height, nominal_fps=stream.get('r_frame_rate'),
                  frames_probed=len(pts), frames_decoded=0, playable=True)
    if not len(pts):
        return unplayable(path, 'no video frames', report)
    gaps = np.diff(pts)
    span = float(pts[-1] - pts[0])
    report.update(first_pts_s=float(pts[0]), last_pts_s=float(pts[-1]), duration_s=span,
                  measured_fps=(len(pts) - 1) / span if span > 0 else None,
                  median_gap_s=float(np.median(gaps)) if len(gaps) else None,
                  max_gap_s=float(gaps.max()) if len(gaps) else None)
    if len(gaps) and np.any(gaps <= 0):
        findings.append(finding('ERROR', 'TIMESTAMPS_NOT_INCREASING',
                                'presentation timestamps repeat or go backwards'))
    try:
        decoded, means, frozen = sample_luma(path, width, height, pts)
        report['frames_decoded'], report['mean_luma'] = decoded, round(float(np.mean(means)), 1)
        if decoded != len(pts):
            findings.append(finding('ERROR', 'DECODE_MISMATCH',
                                    '{} frames probed but {} decoded'.format(len(pts), decoded)))
        if means and max(means) < 8:
            findings.append(finding('WARNING', 'NEAR_BLACK', 'sampled frames are almost black (lens cap?)'))
        elif frozen:
            findings.append(finding('WARNING', 'FROZEN_IMAGE', 'sampled frames are identical (no motion/stuck feed)'))
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        report['playable'] = False
        findings.append(finding('ERROR', 'DECODE_ERROR', str(exc)))
    if len(pts) < GATE_FRAMES:
        findings.append(finding('WARNING', 'SHORT_RECORDING',
                                '{} frames; the pose gate needs at least {}'.format(len(pts), GATE_FRAMES)))
    over = int(np.sum(gaps > GATE_GAP_S)) if len(gaps) else 0
    if over:
        findings.append(finding('WARNING', 'TIMESTAMP_GAPS',
                                '{} adjacent gaps exceed {} s (max {:.3f} s)'.format(
                                    over, GATE_GAP_S, report['max_gap_s'])))
    expected = expected or {}
    if expected.get('width') and (width, height) != (expected['width'], expected['height']):
        findings.append(finding('WARNING', 'RESOLUTION_MISMATCH', '{}x{} recorded, {}x{} requested'.format(
            width, height, expected['width'], expected['height'])))
    wanted = expected.get('fps') or 20
    if report['measured_fps'] is not None and report['measured_fps'] < 0.8 * wanted:
        findings.append(finding('WARNING', 'LOW_FPS', 'measured {:.1f} fps vs {} expected'.format(
            report['measured_fps'], wanted)))
    report['findings'] = findings
    levels = [f['level'] for f in findings]
    report['status'] = 'FAIL' if 'ERROR' in levels else 'WARNING' if levels else 'PASS'
    report['playable'] = report['playable'] and 'ERROR' not in levels
    return report


def write_private_json(path, value):
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def record_diagnostic(cfg, seconds, device, resolution=None, fps=None):
    """Operator-triggered short recording, then analysis. Returns the report (status FAIL on faults)."""
    seconds, device, (width, height), fps = validate_record_args(
        seconds, device, resolution, fps, cfg.get('camera'))
    folder = Path(cfg['diagnostics_dir']) / 'camera'
    originals = Path(cfg['capture_dir']).resolve()
    if folder.resolve() == originals or originals in folder.resolve().parents:
        raise ValueError('Diagnostics must be kept apart from enrollment/verification originals')
    stem = 'camera-diag-{}-{}'.format(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'), uuid.uuid4().hex[:6])
    video = folder / (stem + '.mp4')
    camera = dict(cfg.get('camera', {}), resolution=[width, height], fps_target=fps)
    try:
        capture.capture_video(video, camera, seconds, device)
        analysis = analyze_video(video, dict(width=width, height=height, fps=fps))
    except (subprocess.SubprocessError, RuntimeError, OSError) as exc:
        if video.is_file() and video.stat().st_size == 0:
            video.unlink()
        analysis = unplayable(video, 'capture failed: {}'.format(exc))
        analysis['findings'] = [finding('ERROR', 'CAPTURE_FAILED', 'ffmpeg/V4L2 capture failed: {}'.format(exc))]
    # subprocess.run has waited for (or killed) ffmpeg here, so the device node is released.
    report = dict(schema=SCHEMA, purpose=PURPOSE, created_at=datetime.now(timezone.utc).isoformat(
        timespec='seconds'), device=device, requested=dict(seconds=seconds, width=width, height=height, fps=fps),
        video=str(video), report=str(folder / (stem + '.diag.json')), analysis=analysis,
        camera_released=True)
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_private_json(Path(report['report']), report)
    return report
