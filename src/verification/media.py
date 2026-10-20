"""Decode every frame with FFmpeg and preserve presentation timestamps from ffprobe."""
import json
import subprocess
from pathlib import Path

import numpy as np
from .contract import InsufficientGait


def inspect_video(path):
    if not Path(path).is_file():
        raise ValueError('Video does not exist')
    run = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v',
                          '-show_entries', 'stream=width,height:stream_tags=rotate:'
                          'stream_side_data=rotation:frame=best_effort_timestamp_time',
                          '-of', 'json', str(path)], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, timeout=60, check=True)
    value = json.loads(run.stdout.decode('utf8'))
    streams = value['streams']
    if len(streams) != 1:
        raise ValueError('Exactly one video stream is required')
    stream = streams[0]
    width, height = stream['width'], stream['height']
    rotation = int(stream.get('tags', {}).get('rotate', 0))
    for side in stream.get('side_data_list', []):
        rotation = int(side.get('rotation', rotation))
    if rotation % 90:
        raise ValueError('Unsupported arbitrary display rotation')
    if rotation % 180:
        width, height = height, width
    pts = np.array([float(f['best_effort_timestamp_time']) for f in value['frames']])
    if len(pts) < 31:
        raise InsufficientGait('Too few video frames for a gait observation')
    if (len(pts) > 7200 or not np.isfinite(pts).all()
            or np.any(np.diff(pts) <= 0) or pts[-1]-pts[0] > 120
            or max(width, height) > 1920 or min(width, height) > 1080):
        raise ValueError('Need 31-7200 ordered frames, <=120 seconds, <=1920x1080')
    return width, height, pts


def video_frames(path, width, height, pts):
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path), '-map', '0:v:0',
               '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        frame_bytes = width*height*3
        for timestamp in pts:
            data = process.stdout.read(frame_bytes)
            if len(data) != frame_bytes:
                raise ValueError('Truncated or mismatched decoded frame stream')
            yield timestamp, np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        if process.stdout.read(1) or process.wait(timeout=30) != 0:
            raise ValueError('Unexpected frame count or FFmpeg decoder error')
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.kill()
        process.wait()
