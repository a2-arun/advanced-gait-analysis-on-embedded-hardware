"""Check the gait model against the 13-subject research videos (read-only).

Runs the same MediaPipe extractor the live camera uses over every video in
deepfake-detection/data/videos, embeds each walk with GaitModel, and reports:

  1. Leave-one-video-out identification: each video vs. galleries enrolled
     from the OTHER videos (same camera view: F=front, S=side).
  2. A 2-people-enrolled simulation per view: can it tell the two apart, and
     where's the threshold that balances false accepts of strangers against
     false rejects of enrolled people (EER)?

Usage:
    python scripts/eval_research_videos.py [--videos DIR] [--write-center]

--write-center saves the mean embedding over all videos to model.center_path
(the vector GaitModel subtracts before cosine). Pose extraction (~12 min on a
laptop CPU) is cached in outputs/research_poses.pkl.
"""

import argparse
import itertools
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.gait_model import FPS, GaitModel
from src.pose.pose_extraction import GaitFeatureExtractor
from src.utils.config import load_config, resolve_path

DEFAULT_VIDEOS = Path(__file__).parent.parent.parent / "deepfake-detection" / "data" / "videos"
CACHE = resolve_path("outputs/research_poses.pkl")


def extract_poses(videos_dir: Path) -> dict:
    """video stem -> (poses (T,33,3), timestamps (T,), (w, h)), sampled at 25 fps."""
    if CACHE.exists():  # written by this script below, so trusted
        return pickle.load(open(str(CACHE), "rb"))
    extractor = GaitFeatureExtractor()
    out = {}
    videos = sorted(videos_dir.glob("*.mp4"))
    for n, video in enumerate(videos, 1):
        cap = cv2.VideoCapture(str(video))
        fps, size = cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4)))
        poses, stamps, i, last_slot = [], [], 0, -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            slot, i = int(i / fps * FPS), i + 1
            if slot == last_slot:
                continue
            last_slot = slot
            landmarks = extractor.extract_pose_from_frame(frame)
            if landmarks is not None:
                poses.append(landmarks)
                stamps.append(slot / FPS)
        cap.release()
        out[video.stem] = (np.array(poses, np.float32), np.array(stamps), size)
        print("  {}/{} {}: {} pose frames".format(n, len(videos), video.stem, len(poses)), flush=True)
    extractor.close()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(str(CACHE), "wb"))
    return out


def eer(genuine, impostor):
    ths = np.linspace(-1, 1, 4001)
    far = np.array([(impostor > t).mean() for t in ths])
    frr = np.array([(genuine <= t).mean() for t in ths])
    i = np.argmin(np.abs(far - frr))
    return 50 * (far[i] + frr[i]), ths[i]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS)
    ap.add_argument("--write-center", action="store_true")
    args = ap.parse_args()

    config = load_config()
    center_path = resolve_path(config["model"]["center_path"])
    if args.write_center and not center_path.exists():
        np.save(str(center_path), np.zeros(384, np.float32))  # placeholder so GaitModel loads; embeddings don't use it
    model = GaitModel(config)

    E = {}
    for stem, (poses, stamps, size) in extract_poses(args.videos).items():
        e = model.embed(poses, stamps, size) if len(poses) else None
        if e is None:
            print("  skipped {}: walk too short".format(stem))
        else:
            E[stem] = e

    if args.write_center:
        model.center = np.mean(list(E.values()), axis=0).astype(np.float32)
        np.save(str(center_path), model.center)
        print("Wrote center from {} videos -> {}".format(len(E), center_path))

    person = lambda k: k.rsplit("_", 1)[0]
    view = lambda k: k.rsplit("_", 1)[1][0]
    people = sorted({person(k) for k in E})

    print("\n1) Leave-one-video-out, same-view gallery, {} people (chance ~{:.0%})".format(len(people), 1 / len(people)))
    for v in ("F", "S"):
        ok = n = 0
        gen, imp = [], []
        for q in [k for k in E if view(k) == v]:
            scores = {}
            for p in people:
                ks = [k for k in E if person(k) == p and view(k) == v and k != q]
                if ks:
                    scores[p] = np.mean([model.similarity(E[q], E[k]) for k in ks])
            if person(q) not in scores:
                continue
            n += 1
            ok += max(scores, key=scores.get) == person(q)
            gen.append(scores[person(q)])
            imp += [s for p, s in scores.items() if p != person(q)]
        e, t = eer(np.array(gen), np.array(imp))
        print("   view {}: rank-1 {}/{} ({:.0%})   EER {:.1f}% at {:.3f}".format(v, ok, n, ok / n, e, t))

    print("\n2) Two people enrolled, everyone else walks by as a stranger")
    for v in ("F", "S"):
        keys = [k for k in E if view(k) == v]
        told_apart = n_pairs = 0
        gen, imp = [], []
        for pair in itertools.combinations(people, 2):
            enrolled = {p: [k for k in keys if person(k) == p] for p in pair}
            if any(len(ks) < 2 for ks in enrolled.values()):
                continue
            for p in pair:
                for q in enrolled[p]:  # enrolled person walks: enroll from their other videos
                    sigs = {pp: model.make_signature([E[k] for k in ks if k != q]) for pp, ks in enrolled.items()}
                    result = model.identify(E[q], sigs)
                    n_pairs += 1
                    told_apart += result.top_k[0][0] == p
                    gen.append(result.all_scores[p])
            sigs = {pp: model.make_signature([E[k] for k in ks]) for pp, ks in enrolled.items()}
            for q in keys:
                if person(q) not in pair:
                    imp.append(model.identify(E[q], sigs).similarity)
        e, t = eer(np.array(gen), np.array(imp))
        print("   view {}: tells the 2 apart {}/{} ({:.0%})   stranger EER {:.1f}% at threshold {:.3f}".format(
            v, told_apart, n_pairs, told_apart / n_pairs, e, t))
    print("\nconfig.yaml identification.similarity_threshold is currently {}".format(model.threshold))


if __name__ == "__main__":
    main()
