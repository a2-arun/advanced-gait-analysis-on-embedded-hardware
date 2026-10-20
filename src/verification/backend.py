"""Frozen official GaitPT, loaded locally from checked numeric arrays on the Nano."""
import importlib.util
import platform
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .artifacts import check_bundle, load_arrays
from .contract import QUALITY_RECIPE, fingerprint, native_crops, normalize, sha256_file
from .media import inspect_video, video_frames

POSE_SETTINGS = dict(static_image_mode=False, model_complexity=0, smooth_landmarks=False,
                     min_detection_confidence=.5, min_tracking_confidence=.5)


def pose_assets():
    import mediapipe as mp
    if mp.__version__ != '0.8.5':
        raise RuntimeError('This Nano recipe requires MediaPipe 0.8.5; no silent substitution')
    root = Path(mp.__file__).parent
    assets = [root / 'modules/pose_detection/pose_detection.tflite',
              root / 'modules/pose_landmark/pose_landmark_lite.tflite']
    if not all(p.is_file() for p in assets):
        raise RuntimeError('Install the two pose .tflite assets before running; see README')
    return assets


def pose_runtime_smoke():
    import mediapipe as mp
    pose_assets()
    with mp.solutions.pose.Pose(**POSE_SETTINGS) as pose:
        pose.process(np.zeros((480, 640, 3), dtype=np.uint8))


def runtime_recipe(source, bundle, device):
    if platform.machine() == 'aarch64':
        from .environment import require_nano_environment
        require_nano_environment()
    import torch
    import mediapipe as mp
    import einops
    manifest = check_bundle(source, bundle)
    ffmpeg = subprocess.check_output(['ffmpeg', '-version']).decode('utf8').splitlines()[0]
    details = dict(model=manifest, pose='mediapipe-solutions-single-person-lite',
                   mediapipe=mp.__version__, pose_settings=POSE_SETTINGS,
                   pose_assets=[sha256_file(p) for p in pose_assets()],
                   quality=QUALITY_RECIPE, native='grew-two-center30-height-v1',
                   aggregation='normalize-crops-mean-normalize-walk-centroid-v1',
                   variant='raw', torch=torch.__version__, numpy=np.__version__,
                   einops=einops.__version__, cuda=torch.version.cuda,
                   cudnn=torch.backends.cudnn.version(), device=device, ffmpeg=ffmpeg,
                   machine=platform.machine(), python=platform.python_version(),
                   implementation={name: sha256_file(Path(__file__).parent / (name + '.py'))
                                   for name in ('backend', 'media', 'contract', 'store', 'service')},
                   input_assumption='operator-checked-one-walker-no-bystanders')
    return fingerprint(details), details


class LocalGaitPT:
    def __init__(self, source, bundle, recipe, device='cuda'):
        import torch
        import yaml
        check_bundle(source, bundle)
        if device not in ('cuda', 'cpu'):
            raise ValueError('Device must be cuda or explicit development cpu')
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no CPU fallback')
        config = yaml.safe_load((Path(source) / 'configs/gaitpt_config.yaml').read_text())
        config['sequence_length'] = 30
        spec = importlib.util.spec_from_file_location('_pinned_official_gaitpt',
                                                       str(Path(source) / 'gaitpt.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.GaitPT(SimpleNamespace(**config))
        arrays = load_arrays(Path(bundle) / 'weights.npz')
        expected = model.state_dict()
        if set(arrays) != set(expected):
            raise ValueError('Model state keys differ')
        for key, tensor in expected.items():
            if arrays[key].shape != tuple(tensor.shape):
                raise ValueError('Model state shape differs: ' + key)
        model.load_state_dict({k: torch.from_numpy(v) for k, v in arrays.items()}, strict=True)
        self.model = model.to(device).eval().requires_grad_(False)
        self.device, self.recipe, self.embedding_dim = device, recipe, None

    def encode_batch(self, batch):
        import torch
        if (batch.shape != (2, 30, 18, 2) or batch.dtype != np.float32
                or not np.isfinite(batch).all()):
            raise ValueError('Invalid native GaitPT crop batch')
        # Process the two native crops sequentially to bound Nano activation memory.
        self.model.eval()
        with torch.no_grad():
            outputs = [self.model(torch.from_numpy(crop[None]).to(self.device))
                       .cpu().numpy()[0] for crop in batch]
        vector = normalize(np.stack([normalize(v) for v in outputs]).mean(axis=0))
        if self.embedding_dim is not None and len(vector) != self.embedding_dim:
            raise ValueError('Embedding dimension changed')
        self.embedding_dim = len(vector)
        return vector

    def encode(self, video):
        import mediapipe as mp
        pose_assets()  # Refuse implicit network downloads from inside MediaPipe.
        width, height, pts = inspect_video(video)
        xy = np.full((len(pts), 33, 2), np.nan, dtype=np.float32)
        confidence = np.zeros((len(pts), 33), dtype=np.float32)
        frames = video_frames(video, width, height, pts)
        try:
            with mp.solutions.pose.Pose(**POSE_SETTINGS) as pose:
                for index, (_, bgr) in enumerate(frames):
                    result = pose.process(np.ascontiguousarray(bgr[:, :, ::-1]))
                    if result.pose_landmarks is None:
                        continue
                    for joint, landmark in enumerate(result.pose_landmarks.landmark):
                        xy[index, joint] = landmark.x*width, landmark.y*height
                        confidence[index, joint] = landmark.visibility
        finally:
            frames.close()
        return self.encode_batch(native_crops(xy, confidence, pts, width, height))
