# Gait Edge Identification

Real-time **person identification from gait**, extracted from a live camera
feed, running on a laptop or on an NVIDIA Jetson Nano.

- **Jetson Nano (JetPack 4.6, Python 3.6):** jump to [Jetson Nano setup](#jetson-nano-setup-jetpack-46--python-36)
- **Laptop (Windows/Linux/macOS, Python 3.8+):** jump to [Laptop quick start](#laptop-quick-start)

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
Camera (USB webcam or MIPI CSI)
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
requirements.txt          # laptop dependencies (Python 3.8+)
requirements-jetson.txt   # Jetson Nano dependencies (Python 3.6) - torch/mediapipe come from wheels, see below
init_db.py                # one-time SQLite schema setup

models/                   # model architecture + checkpoint (full_hybrid_best.pth is committed)
database/                 # SQLite schema definition

src/
  camera/                 # camera capture (USB index or GStreamer/CSI pipeline)
  pose/                   # MediaPipe extraction + the 78-dim feature builder (shared by offline/live)
  features/               # live camera -> gait-sequence buffering
  model/                  # checkpoint loading + 1:N identification
  database/               # enrolled-identity gallery + event log CRUD
  enrollment/             # "walk in front of the camera a few times" flow
  identification/         # ties camera+model+database+events into the live loop
  events/                 # structured identity events for RPA/automation
  utils/                  # config loading, logging

scripts/
  test_camera.py          # verify the camera works before anything else (--no-display for SSH)
  enroll.py               # enroll a new identity
  identify.py             # run live identification

docs/
  ARCHITECTURE.md          # what's actually true about the model/pipeline, and why
  SETUP.md                 # laptop setup, step by step
  DEPLOYMENT.md             # Jetson background: compatibility risks and why
  phase-reports/            # historical planning docs, superseded by ARCHITECTURE.md

tests/
  test_model_loading.py     # checkpoint/architecture sanity checks
  investigate_checkpoint.py, check_metrics.py   # one-off diagnostic scripts
```

---

## Jetson Nano setup (JetPack 4.6 / Python 3.6)

Target: Jetson Nano Developer Kit 4GB, JetPack 4.6.x (L4T R32.7.x),
Ubuntu 18.04, Python 3.6.9, CUDA 10.2.

### Why the normal `pip install -r requirements.txt` fails here

`requirements.txt` asks for `torch>=2.1` and `mediapipe>=0.10`. Both need
Python 3.8+ and neither ships a Python 3.6 / aarch64 build, so pip answers
*"No matching distribution found"*. The Nano cannot move past Python 3.6
without leaving JetPack's supported stack, so instead this repo uses:

| Package | Laptop | Jetson Nano |
|---|---|---|
| PyTorch | `torch>=2.1` from PyPI | **torch 1.10.0**, NVIDIA's CUDA 10.2 wheel (last release for Python 3.6) |
| MediaPipe | 0.10.x Tasks API | **0.8.5**, PINTO0309's prebuilt cp36 aarch64 wheel, legacy `mp.solutions.pose` API |
| OpenCV | `opencv-python` | JetPack's own system OpenCV (has GStreamer for CSI cameras) |

The code handles both automatically (`src/pose/pose_extraction.py` falls
back to `mp.solutions.pose` when the Tasks API is missing;
`src/model/gait_model.py` loads the checkpoint on old and new torch). The
checkpoint contains only plain tensors, so it loads fine on torch 1.10.

> **Status:** these steps were assembled from NVIDIA's and PINTO0309's
> published instructions and checked against this repo's code, but they have
> **not been run end to end on a physical Nano yet**. Each step has a check;
> if one fails, see [Troubleshooting](#jetson-troubleshooting) and tell me the
> output.

### 1. Confirm the board

```bash
cat /etc/nv_tegra_release      # expect R32 (release), REVISION: 7.x
python3 --version              # expect Python 3.6.9
nvcc --version                 # expect CUDA 10.2
free -h                        # ~4GB RAM
df -h /                        # need >= 6GB free on the microSD card
```

### 2. Max performance + swap (do this first - 4GB RAM is tight)

```bash
sudo nvpmodel -m 0             # MAXN power mode (use a 5V 4A barrel-jack supply, jumper J48)
sudo jetson_clocks             # lock clocks at max

# 4GB swap file - importing torch + MediaPipe can otherwise get OOM-killed
sudo fallocate -l 4G /mnt/4GB.swap
sudo chmod 600 /mnt/4GB.swap
sudo mkswap /mnt/4GB.swap
sudo swapon /mnt/4GB.swap
echo '/mnt/4GB.swap none swap sw 0 0' | sudo tee -a /etc/fstab
free -h                        # Swap should now show ~4G
```

### 3. System packages

```bash
sudo apt update
sudo apt install -y git python3-pip python3-venv python3-dev \
    libopenblas-base libopenmpi-dev libomp-dev \
    libjpeg-dev zlib1g-dev python3-matplotlib sqlite3
```

`python3-matplotlib` is needed because MediaPipe 0.8.5 imports it and it
is painful to compile with pip on ARM. JetPack already provides
`python3-opencv` (with GStreamer); confirm:

```bash
python3 -c "import cv2; print(cv2.__version__)"
python3 -c "import cv2; print([l for l in cv2.getBuildInformation().splitlines() if 'GStreamer' in l])"
```

### 4. Clone the repo

```bash
cd ~
git clone https://github.com/a2-arun/advanced-gait-analysis-on-embedded-hardware.git
cd advanced-gait-analysis-on-embedded-hardware
ls models/checkpoint/full_hybrid_best.pth      # the model is committed - no extra download
```

### 5. Virtual environment (with system site-packages)

`--system-site-packages` is what lets the venv see JetPack's OpenCV.

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade "pip==21.3.1" "setuptools==59.6.0" wheel   # last versions supporting Python 3.6

# Nano-specific environment fixes (see Troubleshooting for why)
echo 'export OPENBLAS_CORETYPE=ARMV8' >> ~/.bashrc
export OPENBLAS_CORETYPE=ARMV8
```

Every new terminal: `cd ~/advanced-gait-analysis-on-embedded-hardware && source .venv/bin/activate`.

### 6. Python dependencies

```bash
pip install -r requirements-jetson.txt
```

### 7. PyTorch 1.10.0 (NVIDIA wheel, GPU-enabled)

```bash
pip install 'Cython<3'
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl \
     -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
```

Check (the second line must print `True`):

```bash
python3 -c "import torch; print(torch.__version__)"                 # 1.10.0
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Do **not** `pip install torch` from PyPI on the Nano - those wheels have no
CUDA support for this board.

### 8. MediaPipe 0.8.5 (PINTO0309 prebuilt wheel)

```bash
cd ~
git clone https://github.com/PINTO0309/mediapipe-bin.git
cd mediapipe-bin
chmod +x ./v0.8.5/download.sh && ./v0.8.5/download.sh
unzip -o v0.8.5.zip -d v0.8.5

# --no-deps: the wheel's metadata pulls opencv-python, which would spend hours
# compiling. We use JetPack's OpenCV instead; the other deps are already in step 6.
cd ~/advanced-gait-analysis-on-embedded-hardware
pip install --no-deps "$(find ~/mediapipe-bin -name 'mediapipe-0.8.5_cuda102-cp36-cp36m-linux_aarch64.whl' | grep numpy119x | head -1)"
```

Check that pose is present:

```bash
python3 -c "import mediapipe as mp; print(mp.__version__, mp.solutions.pose.Pose)"
```

If it says `No module named 'X'`, `pip install X` and retry (the wheel was
installed without dependency resolution).

### 9. Verify the model loads and runs (uses the GPU automatically)

```bash
python3 init_db.py
python3 tests/test_model_loading.py
```

Expected: every test prints `[OK] PASS`, and Test 5 reports
`CUDA available: NVIDIA Tegra X1`.

### 10. Verify the camera

**USB webcam:** plug it in, then `ls /dev/video*` should list `/dev/video0`.
The default `camera.device_id: 0` in `config.yaml` is correct.

**MIPI CSI camera (IMX219 etc.):** in `config.yaml`, replace
`device_id: 0` with the GStreamer pipeline string shown in the comment
directly above it. Test the sensor outside Python first:

```bash
gst-launch-1.0 nvarguscamerasrc num-buffers=60 ! 'video/x-raw(memory:NVMM),width=1280,height=720' ! nvvidconv ! fakesink
```

Then test through the project:

```bash
python3 scripts/test_camera.py                  # with a monitor on HDMI: live preview, press q
python3 scripts/test_camera.py --no-display     # over SSH / no monitor: grabs 100 frames, saves outputs/camera_test.jpg
```

### 11. Enroll people, then identify

Stand 2-3 m from the camera so your **whole body** (head to feet) is in
frame, in decent light, and walk **across** the view.

```bash
python3 scripts/enroll.py --name "Alice"        # walk across 3-5 times when prompted
python3 scripts/enroll.py --name "Bob"          # enroll at least one more person to test UNKNOWN/match
python3 scripts/identify.py --device-id jetson-nano-01
```

`enroll.py` opens a live preview: skeleton overlay, `Pass N/5`, a frame
counter for the current pass, and a green border while you're tracked (red
= no pose - get your whole body in view). "Pass N captured" flashes after
each pass; press `q` in the window to cancel without saving. Over SSH it
falls back to terminal-only automatically; add `--no-display` to force that,
or `export DISPLAY=:0` first to show the window on the Nano's monitor.
`identify.py` has no preview.

To remove a test enrollment:

```bash
python3 -c "from src.database.gait_database import GaitDatabase; GaitDatabase().delete_identity('Alice')"
```

### 12. Where the output is

| What | Where |
|---|---|
| Live result | the terminal: `IDENTIFIED: Alice (similarity: 0.94)` or `UNKNOWN PERSON ...` |
| One JSON event per identification | `outputs/events/*.json` |
| Log file | `outputs/logs/gait_identification.log` |
| Enrolled people + every identification attempt | `database/gait.db` |

```bash
sqlite3 database/gait.db "select person_name, num_samples, quality_score from enrolled_identities;"
sqlite3 database/gait.db "select timestamp,status,identified_person_name,top_similarity_score,processing_time_ms from identification_events order by id desc limit 10;"
ls -t outputs/events | head -3
```

For the performance numbers this project still owes (see
`docs/ARCHITECTURE.md` #7), run `tegrastats` in a second terminal while
`identify.py` is running, and read `processing_time_ms` from the query
above. MediaPipe pose extraction on the Nano's CPU is the bottleneck, not
the model. Real-time speed on this board is **unmeasured** - the first run
will tell us.

### Jetson troubleshooting

**`No matching distribution found for torch>=2.1` / `mediapipe>=0.10`.**
You ran `pip install -r requirements.txt`, which is the laptop file. Use
steps 6-8 above.

**`Illegal instruction (core dumped)` on `import numpy` / `import torch`.**
`export OPENBLAS_CORETYPE=ARMV8` (step 5 adds it to `~/.bashrc`; open a new
terminal or `source ~/.bashrc`).

**`cannot allocate memory in static TLS block` on import.**
`export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1` before running.

**`torch.cuda.is_available()` is `False`.** You have a non-NVIDIA torch
wheel. `pip uninstall torch`, redo step 7. The pipeline still runs on CPU
(`model.device: "auto"` in `config.yaml`), just slower.

**Process `Killed` / system freezes.** Out of RAM. Confirm swap is on
(`free -h`), close the desktop browser, and consider running headless
(`sudo systemctl set-default multi-user.target`, reboot).

**`ImportError: ... dataclasses` / `No module named 'dataclasses'`.**
`pip install dataclasses` (it's in `requirements-jetson.txt`; you're
probably not inside the venv).

**`import cv2` fails inside the venv.** The venv was created without
`--system-site-packages`. Delete `.venv` and redo step 5.

**Camera won't open.** USB: `ls /dev/video*`; make sure nothing else is using
it. CSI: `sudo systemctl restart nvargus-daemon`, reseat the ribbon cable
with the board powered off, and re-run the `gst-launch-1.0` check.

**`cv2.imshow` errors over SSH.** No display attached. Use
`python3 scripts/test_camera.py --no-display` / `python3 scripts/enroll.py
--name X --no-display`. `identify.py` never opens a window.

**MediaPipe wheel download fails or `mp.solutions.pose` is missing.** The
PINTO0309 download script pulls from an external host and can rot. Fallback
is building MediaPipe from source on the Nano (many hours), or replacing
pose estimation with `trt_pose` (different landmarks - would need the 78-dim
feature builder adapted and the model re-validated). See `docs/DEPLOYMENT.md`.

**`enroll.py`/`identify.py` crash with `HTTPError: HTTP Error 404: Not
Found`, "Downloading model to .../mediapipe/modules/pose_landmark/
pose_landmark_lite.tflite".** mediapipe 0.8.5's legacy `mp.solutions.pose`
auto-downloads that `.tflite` from `github.com/google/mediapipe/raw/master/`
on first use - that repo has since moved to `google-ai-edge/mediapipe` and
the old master-relative path 404s. Fix: download the file once from a tag
that still has it, and drop it at the exact path the traceback names (skips
the broken download - mediapipe uses the local copy if it's already there):

```bash
wget -O .venv/lib/python3.6/site-packages/mediapipe/modules/pose_landmark/pose_landmark_lite.tflite \
    https://raw.githubusercontent.com/google-ai-edge/mediapipe/v0.8.9/mediapipe/modules/pose_landmark/pose_landmark_lite.tflite
```

---

## Laptop quick start

See `docs/SETUP.md` for the full walkthrough. Short version:

```bash
python -m venv .venv && .venv\Scripts\activate   # or source .venv/bin/activate
pip install -r requirements.txt

# models/checkpoint/full_hybrid_best.pth is already committed in this repo

python init_db.py
python tests/test_model_loading.py
python scripts/test_camera.py

python scripts/enroll.py --name "Alice"
python scripts/identify.py
```

## Status

Laptop pipeline (camera -> pose -> features -> model -> identification ->
database -> events) is implemented end to end. Jetson Nano install path is
documented above and the code is compatible with its Python 3.6 / torch
1.10 / mediapipe 0.8.5 stack, but has not yet been run on the device. Not
yet validated with a real enrolled gallery and real walking subjects on
either platform.

## Known limitations

- The model was trained on 13 subjects; treat identification accuracy
  claims accordingly (see `docs/ARCHITECTURE.md` #7 for the full list).
- The similarity threshold (0.7737) was calibrated for 1:1 verification,
  not open-set identification against a growing gallery - revisit once
  there's real enrollment data.
- On the Jetson the pose backend is MediaPipe 0.8.5's legacy `Pose` in
  tracking mode with the lite model, not the 0.10 Tasks `PoseLandmarker`
  the laptop uses. Landmarks are the same 33-point layout, but small numeric
  differences mean enrollments made on one platform should not be assumed to
  match on the other - enroll and identify on the same device.
- No liveness/spoof detection. This is a biometric identification system,
  not an access-control security system.
- MediaPipe pose extraction (CPU) is the pipeline's latency bottleneck, not
  the ~850K-parameter model.
