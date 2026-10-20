# Architecture

This document records what was actually verified by inspecting the research
repository and the checkpoint itself, as opposed to what earlier planning
documents assumed. Where the two disagree, this file and the code follow the
verified behavior. See `docs/phase-reports/` for the superseded plans.

## 1. Relationship to the research project

The research project (`Deepfake-Detection`, sibling repo) asks: *"does this
video's walk match the person it claims to be?"* — a 1:1 **verification**
task used to catch face-swapped video (the face says A, but gait says B).

This project asks a different question: *"who is walking in front of this
camera right now?"* — an open-set **identification** task. A live camera
feed can't be a deepfake (there's no video file to swap a face onto), so the
authenticity-verification framing doesn't apply here. What carries over is
the gait representation and the trained model that scores "does this gait
match that enrolled signature" — identification is just that scoring
function run against every enrolled person instead of one claimed identity.

Reused as-is: the 78-dim feature engineering (`src/pose/pose_extraction.py`,
ported from the research repo's `utils/pose_extraction.py`) and the trained
checkpoint (`models/checkpoint/full_hybrid_best.pth`, copied from
`outputs/ablation/full_hybrid_best.pth`).
New in this project: everything that turns those into a live pipeline -
camera capture, sequence buffering, the identification/enrollment
orchestration, the SQLite gallery, and event emission.

## 2. Pipeline

```
Camera frame
    -> Pose extraction (MediaPipe, 33 landmarks -> 12 gait-relevant)
    -> Rolling sequence buffer (src/features/gait_features.py)
    -> Resample to 60 frames, compute 78-dim/frame features
    -> GaitModel.identify() against the enrolled gallery (src/model/gait_model.py)
    -> IdentificationResult -> IdentityEvent (src/events/identity_event.py)
```

## 3. The 78-dim feature vector

Per frame, concatenated in this exact order (order matters - it must match
what the checkpoint was trained on):

| Component | Dims | Source |
|---|---|---|
| Hip-centered 3D coordinates | 36 (12 x 3) | 12 gait landmarks: shoulders, hips, knees, ankles, heels, foot-tips |
| Joint flexion angles | 6 | knee/hip/ankle, left+right |
| Frame-to-frame velocities | 36 (12 x 3) | first derivative of the coordinates above |

Sequences are resampled to 60 frames via linear interpolation on the raw
landmarks *before* angles/velocities are computed (not after) -
`GaitFeatureExtractor.sequence_to_model_input()` is the single place this
happens, used by both offline video processing and the live camera buffer,
specifically so the two paths can't drift apart.

## 4. Why identification uses the verification head, not embeddings+cosine

The model produces a 128-dim embedding via `GaitEncoder -> BiLSTM/Transformer
fusion`, and there's a separate `IdentityVerifier` module built around
cosine similarity between two projected embeddings. **Neither of those is
what the checkpoint's `forward(mode='verification')` actually uses.**

Reading `models/full_pipeline.py`'s verification branch: it computes
`diff`, `abs_diff`, and `product` directly from the two RAW 78-dim feature
sequences (query vs. candidate), feeds that through a small 1D CNN
(`diff_conv` + `diff_classifier`), and outputs `P(same person)`. The 128-dim
embedding is computed alongside but never enters this calculation - it's
there for diagnostics and the standalone classifier mode.

This isn't an oversight to fix; per the research team's own technical
notes, the embedding-comparison approach was tried first and **collapsed
during training** (the model learned to ignore the input). The
difference-based CNN was what trained stably and produced the validated
94.95% AUC-ROC / 0.7737 Youden's-J threshold. Building a new
embedding+cosine matcher for this project would be building something never
validated, using weights that were never optimized for it.

Consequence for identification: `GaitModel.identify()` (`src/model/gait_model.py`)
batches the query against every enrolled signature and runs the real
verification forward pass N times (one per candidate), taking the argmax
above threshold. This is cheap - `diff_conv`/`diff_classifier` is a small
CNN, and the model is ~850K parameters total - but it does mean matching
cost scales with gallery size, unlike a cosine-similarity lookup. Fine for
the gallery sizes this project targets (tens of people); would need
revisiting if the gallery ever reached the thousands.

Caveat worth remembering: the 0.7737 threshold was calibrated for 1:1
verification via 13-fold LOOCV. Open-set identification against a growing
gallery changes the false-accept calculus (more candidates = more chances
for a false positive above threshold). Re-derive this threshold once real
enrollment data exists instead of trusting it indefinitely.

## 5. Normalization: deliberately none

`train.py` computes and saves `feature_stats` (per-dimension mean/std) into
checkpoints when available, and the reference `inference.py` z-score
normalizes with them. **This specific checkpoint has no `feature_stats`** -
confirmed by loading it directly, not assumed. `inference.py` itself
handles that by skipping normalization and printing a warning. That means
the only code path this checkpoint's headline numbers (96.11% AUC on this
variant, 90.32% accuracy) were produced through runs on **raw, unnormalized
features**.

An earlier plan in this project (`docs/phase-reports/PHASE_1_UPDATE.md`)
proposed computing feature statistics from enrollment data and normalizing
before inference. That was never implemented or tested against this
checkpoint's actual behavior, and doing so now would silently shift the
input distribution away from what the model was evaluated on. `config.yaml`
sets `model.normalize_features: false`, and `GaitModel` raises rather than
silently normalizing if that's ever flipped without a matching
`feature_stats` mechanism being built first.

## 6. Events / RPA boundary

`src/events/identity_event.py` builds a plain `IdentityEvent` (person,
similarity, threshold, timestamp, device_id) from an `IdentificationResult`
and writes it to `outputs/events/*.json` (optionally POSTs to a webhook).
Nothing upstream of this module knows RPA exists, and nothing in the event
module imports torch/mediapipe - an actual RPA platform (UiPath, Power
Automate, a webhook receiver, whatever gets chosen later) consumes these
events without touching the ML pipeline.

## 7. Known limitations, stated plainly

- 13-subject training set; LOOCV was used to be honest about generalization,
  but the model has still only ever seen 13 people's gait.
- The identification threshold is a verification-task threshold reused for
  identification (see #4's caveat).
- No spoof/liveness detection - this is a biometric identification system,
  not a security-hardened access-control system. Don't market it as one.
- Feature engineering runs on CPU (MediaPipe); it's the pipeline's latency
  bottleneck, more so than the ~850K-parameter model.
