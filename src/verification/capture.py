"""Local USB recording for one independent walk; never overwrites an original."""
import os
import subprocess
import sys
from pathlib import Path


def capture_video(path, camera, seconds, device=None):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError('Original output already exists')
    if not 1 <= seconds <= 120:
        raise ValueError('Capture duration must be 1-120 seconds')
    device = camera.get('device_id', 0) if device is None else device
    if isinstance(device, int) or str(device).isdigit():
        device = '/dev/video' + str(device)
    if not str(device).startswith('/dev/video'):
        raise ValueError('USB capture requires a local /dev/video device')
    width, height = camera.get('resolution', [640, 480])
    fps = camera.get('fps_target', 30)
    if not (1 <= width <= 1920 and 1 <= height <= 1080 and 1 <= fps <= 60):
        raise ValueError('Unsupported camera resolution or frame rate')
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    print('Recording {} seconds from {}. Walk naturally with your whole body and feet visible.'
          .format(seconds, device), file=sys.stderr)
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-n', '-f', 'v4l2',
                    '-framerate', str(fps), '-video_size', '{}x{}'.format(width, height),
                    '-i', str(device), '-t', str(seconds), '-an', '-vsync', '0',
                    '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
                    '-pix_fmt', 'yuv420p', str(path)], check=True, timeout=seconds+30)
    if not path.is_file() or not path.stat().st_size:
        raise RuntimeError('Camera produced no recording')
    os.chmod(str(path), 0o600)
