#!/usr/bin/env python3
"""Workstation-only. Derive a parity fixture from the research implementation, read-only.

Nothing is written except --output. Extract the research package from Git without
touching its checkout, e.g. (run from any directory):
  git -C <research> archive origin/feat/gait-verification-v2 gait_verification | tar -x -C <tmp>
Then pass that directory as --research-src. The research GaitPT clone and checkpoint are
only read; bytecode writing is disabled. Torch >= 1.13 (weights_only loading) is required.
"""
import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

RESEARCH_FILES = ('contracts.py', 'skeleton_contracts.py', 'backends/gaitpt.py',
                  'pose_quality.py', 'scoring.py')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synthetic_coco17(n, seed, width=640, height=480):
    """Deterministic walking-like COCO17 pixel track (not a real person)."""
    t = np.arange(n, dtype=np.float64)[:, None]
    base = np.array([[320, 80], [330, 75], [310, 75], [340, 78], [300, 78], [350, 140],
                     [290, 140], [360, 200], [280, 200], [365, 250], [275, 250], [335, 260],
                     [305, 260], [338, 340], [302, 340], [340, 420], [300, 420]], np.float64)
    xy = np.repeat(base[None], n, axis=0)
    phase = 2 * np.pi * (t / 28.0 + .004 * t ** 2 / n * 28) + seed  # chirp: windows differ
    xy[:, :, 0] += 100 - 220 + 2.0 * t
    for joint, side in ((15, 1), (13, .5), (7, -.5), (9, -1)):
        swing = 25 * (1 + .8 * t[:, 0] / n) * side * np.sin(phase[:, 0])
        xy[:, joint, 0] += swing
        xy[:, joint + 1, 0] -= swing
    xy += np.random.RandomState(seed).normal(0, .4, xy.shape)
    return np.round(xy, 3).astype(np.float32), np.arange(n) / 30.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--research-src', required=True, help='dir containing gait_verification/')
    ap.add_argument('--research-commit', required=True)
    ap.add_argument('--gaitpt-source', required=True, help='clean audited GaitPT clone (read)')
    ap.add_argument('--checkpoint', required=True, help='official GREW checkpoint (read)')
    ap.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    import torch
    import yaml
    sys.path.insert(0, str(Path(args.research_src).resolve()))
    contracts = importlib.import_module('gait_verification.contracts')
    skeleton = importlib.import_module('gait_verification.skeleton_contracts')
    gaitpt = importlib.import_module('gait_verification.backends.gaitpt')
    pose_quality = importlib.import_module('gait_verification.pose_quality')
    scoring = importlib.import_module('gait_verification.scoring')

    source = Path(args.gaitpt_source)
    config = yaml.safe_load((source / 'configs/gaitpt_config.yaml').read_text())
    config['sequence_length'] = 30
    official = gaitpt.load_source(source / 'gaitpt.py', '_parity_official_gaitpt')
    model = official.GaitPT(SimpleNamespace(**config))
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict({k.replace('module.', '', 1): v for k, v in state.items()}, strict=True)
    # The research encoder's own encode(); its __init__ only adds git/hash gates and CUDA.
    encoder = gaitpt.GaitPTEncoder.__new__(gaitpt.GaitPTEncoder)
    encoder.model, encoder.device, encoder.embedding_dim = (
        model.to(args.device).eval().requires_grad_(False), args.device, None)
    encoder.metadata = SimpleNamespace(model_id='parity')

    def sequence(xy, pts, width=640, height=480):
        n = len(xy)
        return contracts.PoseSequence(
            xy=xy, confidence=np.full((n, 17), .9, np.float32), observed=np.ones((n, 17), bool),
            timestamps=pts, layout='coco17', width=width, height=height,
            original_take_id='synthetic', source_sha256='0' * 64, backend_id='synthetic-coco17',
            frame_indices=np.arange(n), track_ids=np.zeros(n, int), bboxes=np.zeros((n, 4)))

    def case(case_id, seed, n, **extra):
        xy, pts = synthetic_coco17(n, seed)
        seq = sequence(xy, pts)
        return seq, dict(id=case_id, coco17_xy=xy.tolist(), timestamps=pts.tolist(), width=640,
                         height=480, **extra)

    cases = []
    seq, entry = case('whole_sequence_91f', 1, 91)
    entry['model_input'] = skeleton.gaitpt_grew_inputs(seq).tolist()
    probe = encoder.encode(seq)
    entry['walk_embedding'] = probe.normalized().tolist()
    templates = []
    for seed in (2, 3, 4):
        reference = encoder.encode(case('ref', seed, 91)[0])
        templates.append(SimpleNamespace(embedding=reference, quality_weight=1.0))
    entry['reference_embeddings'] = [t.embedding.normalized().tolist() for t in templates]
    entry['reference_prototype'] = scoring.prototype(templates).tolist()
    entry['similarity'] = scoring.score_claim(probe, templates).score
    cases.append(entry)

    # Research operational path (2 s windows, per-window encode, mean) vs hardware single run.
    seq, entry = case('windowed_operational_181f', 5, 181, expect='differs')
    windowed = pose_quality.WindowedEncoder(encoder)
    entry['accepted_segments'] = [list(map(int, s)) for s in
                                  pose_quality.analyze_pose(seq)['accepted_segments']]
    entry['walk_embedding'] = windowed.encode(seq).normalized().tolist()
    cases.append(entry)

    root = Path(args.research_src) / 'gait_verification'
    fixture = dict(
        schema='gait-parity-fixture-v1',
        producer=dict(repository='deepfake-detection', commit=args.research_commit,
                      tool='scripts/make_parity_fixture.py', device=args.device,
                      torch=torch.__version__, numpy=np.__version__,
                      research_files={f: sha256(root / f) for f in RESEARCH_FILES},
                      note='Synthetic COCO17 skeletons; official GaitPT-GREW weights; '
                           'no participant data.'),
        identity=dict(checkpoint_sha256=sha256(args.checkpoint),
                      source_revision=gaitpt.REVISION,
                      architecture_sha256=sha256(source / 'configs/gaitpt_config.yaml'),
                      source_sha256=sha256(source / 'gaitpt.py'),
                      joint_order=list(skeleton.GAITPT_ORDER), input_shape=[2, 30, 18, 2]),
        cases=cases)
    Path(args.output).write_text(json.dumps(fixture, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    main()
