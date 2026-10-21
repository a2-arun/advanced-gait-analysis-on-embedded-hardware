"""Pose estimation as an independent diagnostic: load a backend, run it on an operator-chosen
recording, and report coverage/quality numbers plus an optional skeleton overlay. No model is
loaded and nothing is stored; confidence scores are not proof of anatomical correctness.

A backend implements describe()/process()/close(). Replacing MediaPipe later means adding a
class here (and a deployment contract naming it); capture and SQLite storage never import it.
"""
import subprocess
from pathlib import Path

import numpy as np

from .backend import POSE_SETTINGS, pose_assets
from .camera_diag import probe
from .contract import InsufficientGait, MP_COCO, native_crops, sha256_file
from .media import video_frames

MEDIAPIPE_LITE = 'mediapipe-solutions-single-person-lite'
JOINTS = {'nose': 0, 'left_eye': 2, 'right_eye': 5, 'left_ear': 7, 'right_ear': 8,
          'left_shoulder': 11, 'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14,
          'left_wrist': 15, 'right_wrist': 16, 'left_hip': 23, 'right_hip': 24, 'left_knee': 25,
          'right_knee': 26, 'left_ankle': 27, 'right_ankle': 28, 'left_heel': 29, 'right_heel': 30,
          'left_foot_index': 31, 'right_foot_index': 32}
BONES = ((11, 12), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
         (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32), (11, 13), (13, 15),
         (12, 14), (14, 16))
INTERPRETATION = ('Visibility/confidence and gate results describe detector output only; they do '
                  'not establish that landmarks are anatomically correct. Inspect the overlay manually.')


class MediaPipeLitePose:
    """The pose graph of the current Nano recipe (MediaPipe 0.8.5 solutions Pose, lite)."""
    name, layout = MEDIAPIPE_LITE, 'mediapipe33'

    def __init__(self):
        self.assets = pose_assets()  # raises unless MediaPipe is exactly 0.8.5 with both assets
        import mediapipe as mp
        self.version = mp.__version__
        self.graph = mp.solutions.pose.Pose(**POSE_SETTINGS)

    def describe(self):
        return dict(name=self.name, version=self.version, layout=self.layout,
                    settings=dict(POSE_SETTINGS), asset_sha256=[sha256_file(p) for p in self.assets])

    def process(self, bgr):
        result = self.graph.process(np.ascontiguousarray(bgr[:, :, ::-1]))
        if result.pose_landmarks is None:
            return None
        height, width = bgr.shape[:2]
        points = result.pose_landmarks.landmark
        return (np.array([[p.x * width, p.y * height] for p in points], dtype=np.float32),
                np.array([p.visibility for p in points], dtype=np.float32))

    def close(self):
        self.graph.close()


POSE_BACKENDS = {MEDIAPIPE_LITE: MediaPipeLitePose}


def load_backend(name):
    if name not in POSE_BACKENDS:
        raise ValueError('Unknown pose backend {!r}; available: {}'.format(
            name, ', '.join(sorted(POSE_BACKENDS))))
    return POSE_BACKENDS[name]()


def check_backend(backend):
    """Does the backend load and describe itself? Closes it; never reads a recording."""
    try:
        description = backend.describe()
        return dict(status='PASS', detail='backend loaded', backend=description)
    except Exception as exc:
        return dict(status='FAIL', detail='backend failed to load/describe: {}'.format(exc), backend=None)
    finally:
        close = getattr(backend, 'close', None)
        if close:
            close()


def check_named_backend(name):
    """Load a registered backend by name (unknown names raise) and report PASS/FAIL."""
    if name not in POSE_BACKENDS:
        load_backend(name)  # raises ValueError listing the available backends
    try:
        backend = load_backend(name)
    except Exception as exc:
        return dict(status='FAIL', detail='backend failed to load: {}'.format(exc), backend=None)
    return check_backend(backend)


def draw_overlay_writer(path, width, height, fps):
    """ffmpeg stdin pipe that encodes BGR frames; separate from the source recording."""
    from .readiness import ffmpeg_probe
    codec = 'libx264' if ffmpeg_probe()['libx264'] else 'mpeg4'
    return subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt',
                             'bgr24', '-s', '{}x{}'.format(width, height), '-r', '{:.3f}'.format(fps),
                             '-i', 'pipe:0', '-c:v', codec, '-pix_fmt', 'yuv420p', str(path)],
                            stdin=subprocess.PIPE)


def draw_skeleton(cv2, frame, xy):
    canvas = frame.copy()
    if xy is not None:
        for a, b in BONES:
            cv2.line(canvas, tuple(int(v) for v in xy[a]), tuple(int(v) for v in xy[b]), (0, 220, 0), 2)
        for point in xy[sorted(JOINTS.values())]:
            cv2.circle(canvas, (int(point[0]), int(point[1])), 3, (255, 255, 255), -1)
    return canvas


def extract_pose_report(video, backend, overlay=None):
    video = Path(video)
    if overlay is not None:
        overlay = Path(overlay)
        if overlay.resolve() == video.resolve() or overlay.exists():
            raise ValueError('Overlay must be a new file different from the source recording')
    try:
        value = probe(video)
        stream = value['streams'][0]
        width, height = int(stream['width']), int(stream['height'])
        pts = np.array([float(f['best_effort_timestamp_time']) for f in value['frames']])
    except (subprocess.SubprocessError, OSError, ValueError, KeyError, IndexError):
        raise ValueError('Video cannot be read by ffprobe')
    count = len(pts)
    xy = np.full((count, 33, 2), np.nan, dtype=np.float32)
    visibility = np.zeros((count, 33), dtype=np.float32)
    detected = np.zeros(count, dtype=bool)
    writer, cv2, status = None, None, None
    if overlay is not None:
        try:
            import cv2
            overlay.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fps = (count - 1) / (pts[-1] - pts[0]) if count > 1 and pts[-1] > pts[0] else 30.0
            writer = draw_overlay_writer(overlay, width, height, fps)
        except ImportError:
            status = 'NOT_TESTED: OpenCV missing'
    frames = video_frames(video, width, height, pts)
    try:
        for index, (_, bgr) in enumerate(frames):
            found = backend.process(bgr)
            if found is not None:
                xy[index], visibility[index] = found
                detected[index] = True
            if writer is not None:
                writer.stdin.write(draw_skeleton(cv2, bgr, xy[index] if detected[index] else None).tobytes())
    finally:
        frames.close()
        if writer is not None:
            writer.stdin.close()
            writer.wait()
        description = backend.describe()
        if hasattr(backend, 'close'):
            backend.close()
    if overlay is not None and status is None:
        status = 'PASS' if writer.returncode == 0 and overlay.is_file() else 'FAIL'
    report = dict(schema='nano-pose-diagnostic-v1', backend=description,
                  video=dict(path=str(video), sha256=sha256_file(video), width=width, height=height,
                             frames=count),
                  frames_total=count, frames_detected=int(detected.sum()),
                  detection_rate=round(float(detected.mean()), 4) if count else 0.0,
                  anatomical_accuracy='NOT_ASSESSED', interpretation=INTERPRETATION)
    report['joint_visibility'] = {}
    for name, joint in JOINTS.items():
        inside = detected & (xy[:, joint, 0] > 0) & (xy[:, joint, 0] < width) \
            & (xy[:, joint, 1] > 0) & (xy[:, joint, 1] < height)
        report['joint_visibility'][name] = dict(
            mean_visibility=round(float(visibility[detected, joint].mean()), 4) if detected.any() else None,
            in_frame_fraction=round(float(inside.sum() / max(int(detected.sum()), 1)), 4))
    report['longest_full_body_run'] = longest_run(xy, visibility, width, height)
    ankles = xy[detected][:, [27, 28]]
    steps = np.linalg.norm(np.diff(ankles, axis=0), axis=2) / height if len(ankles) > 1 else np.zeros(1)
    report['ankle_step_median_height_fraction'] = round(float(np.median(steps)), 5)
    report['ankle_step_max_height_fraction'] = round(float(steps.max()), 5)
    try:
        native_crops(xy, visibility, pts, width, height)
        report['provisional_gate'] = dict(result='PASS', reason=None)
    except InsufficientGait as exc:
        report['provisional_gate'] = dict(result='INSUFFICIENT_GAIT', reason=str(exc))
    report['provisional_gate']['note'] = ('Current MediaPipe recipe gate (provisional, not validated); '
                                          'no model is run.')
    if overlay is not None:
        report['overlay'] = dict(status=status, path=str(overlay))
    return report


def longest_run(xy, visibility, width, height):
    joints = sorted(set(MP_COCO) | {29, 30, 31, 32})
    good = (np.isfinite(xy[:, joints]).all(axis=(1, 2)) & (visibility[:, joints] >= .5).all(axis=1)
            & (xy[:, joints, 0] > 0).all(axis=1) & (xy[:, joints, 0] < width).all(axis=1)
            & (xy[:, joints, 1] > 0).all(axis=1) & (xy[:, joints, 1] < height).all(axis=1))
    best = run = 0
    for ok in good:
        run = run + 1 if ok else 0
        best = max(best, run)
    return best
