"""Versioned Nano input contract. Quality constants are provisional pilot gates."""
import hashlib
import json
import re

import numpy as np

MP_COCO = (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
GAITPT_ORDER = (0, 1, 3, 0, 2, 4, 5, 7, 9, 6, 8, 10, 11, 13, 15, 12, 14, 16)
QUALITY_RECIPE = 'nano-raw-longest-contiguous31-vis0.5-gap0.1-motion0.003-jump0.2-v1'


class InsufficientGait(ValueError):
    pass


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', value):
        raise ValueError('Use a pseudonymous ID of 1-64 letters, digits, _ or -')
    return value


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf8')).hexdigest()


def normalize(value):
    x = np.asarray(value, dtype=np.float32)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError('Expected a finite nonempty embedding vector')
    norm = np.linalg.norm(x)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError('Zero or invalid embedding norm')
    return x / norm


def native_crops(xy, confidence, pts, width, height):
    xy, confidence, pts = np.asarray(xy), np.asarray(confidence), np.asarray(pts)
    n = len(xy)
    if (n < 31 or xy.shape != (n, 33, 2) or confidence.shape != (n, 33)
            or pts.shape != (n,) or width <= 0 or height <= 0
            or not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0)):
        raise InsufficientGait('Invalid/missing frame timeline or too few frames')
    # Retain missing frames in the timeline; never compact detected poses or interpolate.
    joints = sorted(set(MP_COCO) | {29, 30, 31, 32})
    observed = (np.isfinite(xy[:, joints]).all(axis=(1, 2))
                & np.isfinite(confidence[:, joints]).all(axis=1)
                & (confidence[:, joints] >= .5).all(axis=1)
                & (xy[:, joints, 0] > 0).all(axis=1)
                & (xy[:, joints, 0] < width).all(axis=1)
                & (xy[:, joints, 1] > 0).all(axis=1)
                & (xy[:, joints, 1] < height).all(axis=1))
    # Refuse large discontinuities anywhere; do not pool different uncertain tracks.
    visible_pairs = observed[1:] & observed[:-1]
    jumps = np.linalg.norm(np.diff(xy[:, MP_COCO], axis=0), axis=2).mean(axis=1)
    if np.any(jumps[visible_pairs] > height * .2):
        raise InsufficientGait('Uncertain track/camera discontinuity')
    runs, start = [], None
    for i, good in enumerate(observed):
        if i and pts[i] - pts[i-1] > .1:
            if start is not None:
                runs.append((start, i))
            start = None
        if good and start is None:
            start = i
        if not good and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, n))
    if not runs:
        raise InsufficientGait('No complete full-body pose')
    a, b = max(runs, key=lambda run: run[1] - run[0])
    if b-a < 31 or pts[b-1] - pts[a] < .8:
        raise InsufficientGait('Need at least 31 contiguous observed frames over 0.8 seconds')
    segment = xy[a:b]
    hips = segment[:, [23, 24]].mean(axis=1)
    feet = segment[:, [27, 28, 29, 30, 31, 32]]
    if (np.median(feet[:, :, 1].mean(axis=1) - hips[:, 1]) < height*.15
            or np.std(feet - hips[:, None], axis=0).max() < height*.003):
        raise InsufficientGait('Insufficient leg resolution or walking-motion evidence')
    coco = segment[:, MP_COCO]
    n = len(coco)
    halves = (coco[:max(n//2, 30)], coco[min(n//2, n-1-30):n-1])
    crops = []
    for half in halves:
        start = int(len(half)/2 - 15)
        crops.append(half[start:start+30, GAITPT_ORDER] / float(height))
    return np.stack(crops).astype(np.float32)
