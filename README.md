# Gait Edge Identification

Real-time **person identification from gait**, extracted from a live camera
feed, running first on a laptop and eventually on an NVIDIA Jetson Nano.

## Relationship to the research project

This project reuses the gait representation and trained model from a
separate research project, **Deepfake Detection Using Gait Analysis**
(sibling repo), which asks *"does this video's walk match the identity it
claims?"* to catch face-swapped video. A live camera feed can't be a
deepfake - there's no video file to have swapped a face onto - so that
framing doesn't apply here. What this project keeps is the underlying
question the model was actually trained to answer (*"does this gait match
that enrolled signature?"*) and repurposes it for open-set identification:
compare a live gait sequence against everyone enrolled and report the best
match, or "unknown" if nobody clears the threshold.

**Read `docs/ARCHITECTURE.md` before touching the model/matching code** -
several assumptions in earlier planning turned out to be wrong once the
checkpoint and research code were actually inspected (wrong embedding
dimension, an unvalidated normalization plan, and an incorrect assumption
that matching works via cosine similarity on embeddings). That document
records what's actually true and why.

## Pipeline

```
Webcam
  -> Pose extraction (MediaPipe, 12 gait-relevant landmarks)
  -> 78-dim/frame gait features (coordinates + angles + velocities)
  -> 60-frame sequence buffer
  -> Trained CNN+BiLSTM+Transformer verification head, run against
     every enrolled identity
  -> IDENTIFIED: <name> (similarity: 0.94)   or   UNKNOWN PERSON
  -> Identity event (JSON, for downstream automation/RPA)
```

## Project layout

```
config.yaml               # single source of truth for model/camera/db/threshold config
requirements.txt          # laptop dev dependencies (Jetson needs its own, see docs/DEPLOYMENT.md)
init_db.py                # one-time SQLite schema setup

models/                   # model architecture + checkpoint (checkpoint itself is gitignored)
database/                 # SQLite schema definition

src/
  camera/                 # webcam capture
  pose/                   # MediaPipe extraction + the 78-dim feature builder (shared by offline/live)
  features/               # live camera -> gait-sequence buffering
  model/                  # checkpoint loading + 1:N identification
  database/               # enrolled-identity gallery + event log CRUD
  enrollment/             # "walk in front of the camera a few times" flow
  identification/         # ties camera+model+database+events into the live loop
  events/                 # structured identity events for RPA/automation
  utils/                  # config loading, logging

scripts/
  test_camera.py          # verify the webcam works before anything else
  enroll.py               # enroll a new identity
  identify.py             # run live identification

docs/
  ARCHITECTURE.md          # what's actually true about the model/pipeline, and why
  SETUP.md                 # laptop setup, step by step
  DEPLOYMENT.md             # Jetson Nano notes (not started - device not in hand yet)
  phase-reports/            # historical planning docs, superseded by ARCHITECTURE.md

tests/
  test_model_loading.py     # checkpoint/architecture sanity checks
  investigate_checkpoint.py, check_metrics.py   # one-off diagnostic scripts
```

## Quick start

See `docs/SETUP.md` for the full walkthrough. Short version:

```bash
python -m venv .venv && .venv\Scripts\activate   # or source .venv/bin/activate
pip install -r requirements.txt

cp "../Deepfake-Updated/DeepFake-Detection/outputs/ablation/full_hybrid_best.pth" \
   models/checkpoint/full_hybrid_best.pth

python init_db.py
python tests/test_model_loading.py
python scripts/test_camera.py

python scripts/enroll.py --name "Alice"
python scripts/identify.py
```

## Status

Laptop pipeline (camera -> pose -> features -> model -> identification ->
database -> events) is implemented end to end. Not yet validated with a
real enrolled gallery and real walking subjects - that's the next step,
and it needs you and a webcam, not more code. Jetson deployment hasn't
started (see `docs/DEPLOYMENT.md`).

## Known limitations

- The model was trained on 13 subjects; treat identification accuracy
  claims accordingly (see `docs/ARCHITECTURE.md` #7 for the full list).
- The similarity threshold (0.7737) was calibrated for 1:1 verification,
  not open-set identification against a growing gallery - revisit once
  there's real enrollment data.
- No liveness/spoof detection. This is a biometric identification system,
  not an access-control security system.
- MediaPipe pose extraction (CPU) is the pipeline's latency bottleneck, not
  the ~850K-parameter model.
