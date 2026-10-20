"""Diagnose the confirmed Nano profile without installing or upgrading anything."""
import importlib
import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

MODULES = {'numpy': 'numpy', 'protobuf': 'google.protobuf', 'PyYAML': 'yaml',
           'einops': 'einops', 'absl-py': 'absl', 'attrs': 'attr',
           'dataclasses': 'dataclasses', 'six': 'six', 'typing_extensions': 'typing_extensions'}


def pinned_packages():
    path = Path(__file__).resolve().parents[2] / 'requirements-jetson.txt'
    pins = {}
    for line in path.read_text().splitlines():
        match = re.match(r'^([A-Za-z0-9_-]+)==([^;\s]+)', line)
        if match:
            pins[match.group(1)] = match.group(2)
    return pins


def read_optional(path):
    try:
        return Path(path).read_text().strip('\x00\n')
    except OSError:
        return ''


def distribution_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except ImportError:
        # Python 3.6: provided by the explicitly pinned bootstrap setuptools.
        from pkg_resources import get_distribution
        return get_distribution(name).version


def collect_nano_environment(device='/dev/video0'):
    info = dict(machine=platform.machine(), python=platform.python_version(),
                hardware=read_optional('/proc/device-tree/model'),
                l4t=read_optional('/etc/nv_tegra_release'), packages={}, import_errors={})
    release = dict(line.split('=', 1) for line in read_optional('/etc/os-release').splitlines()
                   if '=' in line and not line.startswith('#'))
    info['ubuntu'] = (release.get('VERSION_ID', '').strip(chr(34))
                      if release.get('ID', '').strip(chr(34)) == 'ubuntu' else '')
    modules = dict(MODULES, torch='torch', mediapipe='mediapipe', opencv='cv2', matplotlib='matplotlib')
    for name, module_name in modules.items():
        try:
            module = importlib.import_module(module_name)
            info['packages'][name] = getattr(module, '__version__', None) or distribution_version(name)
        except Exception as exc:
            info['import_errors'][name] = str(exc)
    try:
        torch = importlib.import_module('torch')
        info.update(cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())
    except Exception:
        info.update(cuda=None, cuda_available=False)
    tools = dict(ffmpeg=bool(shutil.which('ffmpeg')), ffprobe=bool(shutil.which('ffprobe')), libx264=False)
    if tools['ffmpeg']:
        result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=10)
        tools['libx264'] = result.returncode == 0 and b'libx264' in result.stdout
    info['tools'] = tools
    info['camera_accessible'] = Path(device).exists() and os.access(device, os.R_OK | os.W_OK)
    return info


def nano_problems(info, check_camera=True):
    problems = []
    for field, expected in [('machine', 'aarch64'), ('ubuntu', '18.04')]:
        if info.get(field) != expected:
            problems.append('{} must be {} (found {})'.format(field, expected, info.get(field)))
    if not str(info.get('python', '')).startswith('3.6.'):
        problems.append('Python 3.6 is required by the Nano wheel profile')
    if 'Jetson Nano' not in info.get('hardware', '') or 'Orin' in info.get('hardware', ''):
        problems.append('Original Jetson Nano hardware is required')
    if not re.search(r'R32.*REVISION:\s*[67]\.', info.get('l4t', '')):
        problems.append('JetPack 4.6.x / L4T R32.6.x or R32.7.x is required')
    expected = dict(pinned_packages(), torch='1.10.0', mediapipe='0.8.5', opencv='4.1.1')
    for name, target in expected.items():
        actual = info['packages'].get(name)
        match = str(actual).startswith(target) if name == 'torch' else actual == target
        if not match:
            problems.append('{} must be {} (found {})'.format(name, target, actual))
    for name, error in info.get('import_errors', {}).items():
        problems.append('{} import failed: {}'.format(name, error))
    if info.get('cuda') != '10.2' or not info.get('cuda_available'):
        problems.append('NVIDIA PyTorch must expose working CUDA 10.2; no CPU fallback')
    for name in ('ffmpeg', 'ffprobe', 'libx264'):
        if not info['tools'].get(name):
            problems.append('Missing required media tool/encoder: ' + name)
    if check_camera and not info.get('camera_accessible'):
        problems.append('USB camera is missing or not readable/writable by this user')
    return problems


def require_nano_environment():
    problems = nano_problems(collect_nano_environment(), check_camera=False)
    if problems:
        raise RuntimeError('Nano environment conflicts: ' + '; '.join(problems))


def camera_decode_smoke(cfg, device='/dev/video0'):
    from .capture import capture_video
    from .media import inspect_video, video_frames
    with tempfile.TemporaryDirectory(prefix='gait-camera-check-') as folder:
        path = Path(folder) / 'camera.mp4'
        capture_video(path, cfg.get('camera', {}), 2, device)
        width, height, pts = inspect_video(path)
        count = sum(1 for _ in video_frames(path, width, height, pts))
        if count != len(pts):
            raise RuntimeError('Camera decoding did not preserve every frame')
        return dict(decoded_frames=count, width=width, height=height,
                    duration_seconds=float(pts[-1]-pts[0]), temporary_recording_deleted=True)
