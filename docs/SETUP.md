# Setup (Laptop)

This is the laptop-first development setup. Jetson Nano deployment is a
separate, later step - see `docs/DEPLOYMENT.md`.

## 1. Python environment

Use a virtual environment, not your system/global Python - this project
pins a specific `mediapipe`/`protobuf` combination that can conflict with
other projects on the same machine (see the note in `requirements.txt`).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Get the model weights

The model is GaitGraph2 (ResGCN-N51-R4) pretrained on OUMVLP-Pose. Its
weights are published without a license, so they're not committed here -
each machine fetches and converts its own copy:

1. Download `model_weights.zip` (~42 MB) from
   https://github.com/tteepe/GaitGraph2/releases/tag/v0.1
2. Convert it (a zip or an already-extracted folder both work):

```bash
python scripts/convert_gaitgraph2.py path/to/model_weights.zip
```

This writes `models/checkpoint/gaitgraph2_oumvlp.pth`. The centering vector
`models/checkpoint/gaitgraph2_center.npy` is committed. Verify:

```bash
python tests/test_gait_model.py
```

Optional, if you have the sibling `deepfake-detection` repo: re-check
accuracy on its 13-subject videos (first run extracts poses, ~12 min):

```bash
python scripts/eval_research_videos.py
```

## 3. Initialize the database

```bash
python init_db.py
```

Creates `database/gait.db` (gitignored - it's local runtime state, not
something to share across machines/commits).

## 4. Test the camera

```bash
python scripts/test_camera.py
```

If this fails, check `config.yaml`'s `camera.device_id` (0 is usually the
built-in/default webcam) and make sure no other application has the camera
open.

## 5. Enroll someone

```bash
python scripts/enroll.py --name "Alice"
```

Walk across the camera's field of view a few times when prompted (full-body
view, decent lighting - MediaPipe needs to see the joints it's tracking).

## 6. Run live identification

```bash
python scripts/identify.py
```

## Troubleshooting

**`MessageFactory.GetPrototype` / mediapipe fails to initialize a
PoseLandmarker.** A `protobuf` version newer than mediapipe's compiled
bindings expect. Confirmed on this project with `protobuf==6.33.0` +
`mediapipe==0.10.21`. Fix: install inside a venv per step 1 above (the
pinned `requirements.txt` avoids this); if it still happens, downgrade
protobuf: `pip install "protobuf<5"`.

**Unicode/`charmap` `UnicodeEncodeError` when running a script.** Some
older scripts in this repo printed `✓`/`❌` characters that Windows'
default console codepage (cp1252) can't encode. These have been replaced
with plain `[OK]`/`[FAIL]` markers in this project's own files. If you hit
this in a script you added yourself, either avoid non-ASCII characters in
`print()` output or add `sys.stdout.reconfigure(encoding="utf-8")` near the
top of the script.
