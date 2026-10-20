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

## 2. Get the model checkpoint

`models/checkpoint/full_hybrid_best.pth` is committed directly in this repo
(it's only 3.5MB) so you don't need access to the sibling research
repository. If it's missing for some reason, and you do have that repo,
copy it from there instead:

```bash
cp "../Deepfake-Updated/DeepFake-Detection/outputs/ablation/full_hybrid_best.pth" \
   models/checkpoint/full_hybrid_best.pth
```

Verify it loads with the architecture this project expects:

```bash
python tests/test_model_loading.py
```

If this fails with a shape mismatch, check `config.yaml`'s `model:` section
against `docs/ARCHITECTURE.md` #1 - the checkpoint needs the *exact*
hyperparameters it was trained with, not `models/full_pipeline.py`'s
`create_model()` defaults.

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
