"""GaitModel sanity checks - run: python tests/test_gait_model.py

Needs the converted GaitGraph2 weights (docs/SETUP.md), no camera and no
research data: walks are synthesized as MediaPipe-style stick figures.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.gait_model import GaitModel


def synthetic_walk(stride_hz, arm_swing, leg_len, seconds=3.0, fps=30, phase=0.0, seed=0):
    """(T, 33, 3) MediaPipe-normalized landmarks of a figure walking left to right."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1.0 / fps)
    poses = np.zeros((len(t), 33, 3), np.float32)
    for i, ti in enumerate(t):
        hip_x, hip_y = 0.2 + 0.2 * ti, 0.55
        swing = np.sin(2 * np.pi * stride_hz * ti + phase)
        p = poses[i]
        p[0, :2] = hip_x, hip_y - 0.33                      # nose
        p[[2, 5, 7, 8], 0], p[[2, 5, 7, 8], 1] = hip_x, hip_y - 0.35  # eyes, ears
        for side, s in ((11, 1), (12, -1)):                 # shoulders, arms
            p[side, :2] = hip_x + 0.02 * s, hip_y - 0.25
            p[side + 2, :2] = hip_x + 0.02 * s + arm_swing * s * swing * 0.5, hip_y - 0.13
            p[side + 4, :2] = hip_x + 0.02 * s + arm_swing * s * swing, hip_y - 0.02
        for side, s in ((23, 1), (24, -1)):                 # hips, legs
            p[side, :2] = hip_x + 0.015 * s, hip_y
            p[side + 2, :2] = hip_x - 0.08 * s * swing, hip_y + leg_len / 2
            p[side + 4, :2] = hip_x - 0.15 * s * swing, hip_y + leg_len
        p[:, :2] += rng.normal(0, 0.002, (33, 2))          # pose-estimator jitter
    return poses, t


def main():
    model = GaitModel()
    size = (640, 480)
    embed = lambda walk: model.embed(walk[0], walk[1], size)

    a1 = embed(synthetic_walk(1.0, 0.06, 0.40, phase=0.0, seed=1))
    a2 = embed(synthetic_walk(1.0, 0.06, 0.40, phase=1.3, seed=2))   # same gait, another pass
    b = embed(synthetic_walk(1.6, 0.01, 0.30, phase=0.4, seed=3))    # different gait

    assert a1.shape == (384,) and abs(np.linalg.norm(a1) - 1) < 1e-4, "embedding must be 384-d unit vector"
    same, diff = model.similarity(a1, a2), model.similarity(a1, b)
    print("same gait: {:.3f}   different gait: {:.3f}".format(same, diff))
    assert same > diff, "the same gait should score higher than a different one"

    short = synthetic_walk(1.0, 0.06, 0.40, seconds=1.0)
    assert model.embed(short[0], short[1], size) is None, "walks under 1.2 s can't be embedded"

    result = model.identify(a2, {"A": model.make_signature([a1]), "B": b})
    assert result.top_k[0][0] == "A", result.top_k

    try:
        model.identify(a1, {"old": np.zeros(60 * 78, np.float32)})
        raise AssertionError("old-model signatures must be rejected")
    except ValueError:
        pass

    print("[OK] all GaitModel checks passed")


if __name__ == "__main__":
    main()
