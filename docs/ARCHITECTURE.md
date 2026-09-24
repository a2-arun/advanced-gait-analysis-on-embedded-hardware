# Architecture

This document records what was verified by running the code and measuring,
as opposed to what earlier planning assumed. Where the two disagree, this
file and the code follow the measurements. See `docs/phase-reports/` for the
superseded plans.

## 1. What this project does

*"Who is walking in front of this camera right now?"* - open-set
identification: compare a live walk against everyone enrolled and report the
best match, or UNKNOWN if nobody clears the threshold.

The gait model is **GaitGraph2** (Teepe et al., ResGCN-N51-R4, trained on
OUMVLP-Pose, ~10,000 subjects), used as-is with its released weights - no
training on our data. New in this project: everything that turns it into a
live pipeline - camera capture, walk buffering, the MediaPipe-to-OpenPose
input mapping, enrollment/identification orchestration, the SQLite gallery,
and event emission.

## 2. Pipeline

```
Camera frame
    -> MediaPipe pose, 33 landmarks + capture timestamp (src/pose, src/features)
    -> ~2-3 s of walking buffered (config sequence.min_valid_frames)
    -> GaitModel.embed() (src/model/gait_model.py):
         resample to 25 fps -> 30-frame windows (stride 8)
         -> each window: OpenPose-18 joints in OUMVLP pixel space
         -> joints/velocity/bones multi-input -> ResGCN
         -> 3 test-time views (as-is, time-reversed, left/right-swapped) concatenated
         -> mean over windows -> 384-d unit vector
    -> GaitModel.identify(): centered cosine vs every enrolled signature
    -> IdentificationResult -> IdentityEvent (src/events/identity_event.py)
```

Enrollment embeds each accepted walk pass and stores the mean as the
person's signature (`gait_embeddings`, one 384-float row).

## 3. Input mapping - the part that has to match training

GaitGraph2 was trained on OpenPose/AlphaPose keypoints in OUMVLP's 1280x980
video frames. We feed MediaPipe landmarks instead, so `gait_model.py` maps
them onto what the checkpoint expects:

- **Joints:** OpenPose-18 order; the neck (no MediaPipe landmark) is the
  shoulder midpoint.
- **Scale and position:** per 30-frame window, converted to pixels, then
  scaled/shifted so the joints match the checkpoint's input BatchNorm
  running statistics (x ~677, y ~460, spread of y relative to the neck ~81).
  Per window, not per walk: walking toward the camera the body grows in the
  frame, and one scale for the whole walk cost ~9 points of accuracy.
- **Confidence:** MediaPipe has none per joint, so every joint gets the
  training mean (0.624), which the input BatchNorm maps to zero.
- **Frame rate:** OUMVLP is 25 fps. Live frames are timestamped and
  resampled, because the camera's effective rate drifts with CPU load.
- **Multi-input features:** `multi_input()` reproduces GaitGraph2's
  `transforms/multi_input.py` bit-for-bit (verified to 2e-7), quirks
  included.

The ResGCN code in `src/model/resgcn/` is vendored from GaitGraph v1 (MIT),
which is identical to GaitGraph2's copy except `forward`; GaitGraph2's
forward lives in `GaitModel._forward`. The GaitGraph2 weights carry no
license, so they aren't committed - `scripts/convert_gaitgraph2.py` turns
the release zip into `models/checkpoint/gaitgraph2_oumvlp.pth`.

## 4. How it was validated, and what the numbers are

`scripts/eval_research_videos.py` runs the live pipeline's own code over the
66 videos of 13 subjects in the sibling `deepfake-detection` repo (F = walking
toward the camera, S = side-on) and never trains on them.

| Test (front view, F) | Result |
|---|---|
| Leave-one-video-out, 13 people, same-view gallery | 76% rank-1 (chance 8%) |
| 2 people enrolled: which of the two is walking | 94% |
| 2 people enrolled: strangers vs enrolled (EER) | 24% at threshold 0.756 |

Side view (S) is much weaker: 46% rank-1, ~73% telling two apart, stranger
EER ~40%. Across views (enrolled front, walking side-on) it's near chance.
Hence: enroll and identify walking toward the camera.

**Centering.** Raw GaitGraph2 embeddings from any one camera share a large
common component - every pair scores ~0.99 and a threshold would have to be
0.998. `GaitModel.similarity` subtracts the mean embedding of the 66
research videos (`models/checkpoint/gaitgraph2_center.npy`) before cosine,
which spreads scores to a usable range. A mean over many *different people*
filmed by the deployment camera itself would do better (front-view stranger
EER 11.6% in simulation), but that needs strangers' walks we don't have;
centering on the enrolled people's own signatures was tested and made
things worse.

**Threshold.** 0.75 is the front-view EER point above. A new camera shifts
it; watch the scores `identify.py` prints and adjust
`identification.similarity_threshold`.

### Why the previous model was replaced

The earlier checkpoint (`full_hybrid_best.pth` from the research repo's
ablation study) was a difference-based verification head over 78-dim
hand-engineered features. It could not identify anyone:

1. The research enrollment averaged **every** video of a person - including
   the video being tested - into their signature. Training and evaluation
   pairs overlapped, so the model learned to detect that overlap. With the
   tested video included: 64/66 correct. Held out: **7/66, chance level.**
   Its reported 96% AUC measured overlap, not gait.
2. Its z-score stats were never saved; this repo fed raw features, which
   saturated the head to exactly 1.0 for every pair.
3. Retraining it leak-free on the 13 research subjects (embedding + triplet
   or cosine-softmax loss, leave-one-subject-out) reached 6/66 - 13 people
   are too few to learn a gait embedding that generalizes to new people.

This affects the sibling research project's reported numbers too, since its
evaluation uses the same enrollment averages.

## 5. Events / RPA boundary

`src/events/identity_event.py` builds a plain `IdentityEvent` (person,
similarity, threshold, timestamp, device_id) from an `IdentificationResult`
and writes it to `outputs/events/*.json` (optionally POSTs to a webhook).
Nothing upstream of this module knows RPA exists, and nothing in the event
module imports torch/mediapipe - an actual RPA platform (UiPath, Power
Automate, a webhook receiver, whatever gets chosen later) consumes these
events without touching the ML pipeline.

## 6. Known limitations, stated plainly

- Roughly 1 in 4 strangers is accepted at the default threshold (front
  view, research videos). Raising the threshold trades that for more
  enrolled people coming up UNKNOWN.
- View-dependent: works walking toward the camera, weak side-on, near
  chance across views.
- Validated on 13 people filmed on phones; the live webcam is a different
  camera. Expect to re-tune the threshold.
- Changing the model or its input mapping invalidates enrolled signatures:
  delete `database/gait.db` and re-enroll.
- No spoof/liveness detection - this is a biometric identification system,
  not a security-hardened access-control system. Don't market it as one.
- MediaPipe pose extraction (CPU) is the latency bottleneck; the gait model
  runs once per captured walk.
