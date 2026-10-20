# Gait claimed-identity verification on Jetson Nano

Give the system a walking video and a claimed enrolled ID. It compares gait with
**that person's reference walks only**. With usable input and an approved calibration
policy it returns **TRUE** or **FALSE: gait does not match the claimed identity**.
It never searches for another person, lists candidates, or treats mismatch as proof
of a deepfake. Capture, decoding, pose extraction, frozen GaitPT inference, reference
storage and comparison all run **locally on the original Jetson Nano**.

This branch implements the Nano engineering port of the research rearchitecture.
**No calibrated threshold is bundled.** A usable enrolled query currently returns
`POLICY_NOT_READY` with `verdict: null`. Nano hardware execution, timing/memory,
the capture pilot and new-cohort accuracy remain unvalidated. See
[architecture](docs/ARCHITECTURE.md) and [validation record](docs/NANO_MIGRATION_VALIDATION.md).

## Daily use: enroll, then verify the claimed identity

After completing the one-time Nano setup below, activate its `.venv` and run:

```bash
python scripts/check_nano.py --capture-test
python scripts/gait.py check --require-nano
python scripts/enroll.py --name Abhishek --consent --single-walker-reviewed
python scripts/identify.py --identity Abhishek --consent --single-walker-reviewed
```

The first two commands are startup checks: packages/CUDA/media/camera, then the
real pose graph and frozen model. Repeat them after dependency or device changes.
For daily use you need only the last **two scripts**. Each records a fresh **30-second
walk from `/dev/video0`**, then processes it on the Nano. No display, separate
capture command, manual registration command or `init_db.py` invocation is needed.
`--seconds` changes duration; `--device /dev/video1` selects another USB camera.

`enroll.py` commits a normalized walk reference to the file-backed database and
prints `ENROLLED`, its database path and reference count. Re-running it records a
new independent take and adds another reference. `identify.py` requires the supplied
identity and compares only its references; it does not identify the walker.
`verify.py` is an equivalent entry point. `--subject-id` and `--claimed-id` remain
aliases for `--name` and `--identity`. IDs are case-sensitive single-token IDs;
pseudonyms are recommended.

`--consent` records the operator's consent attestation for this capture; use it only
when consent has actually been obtained. An existing `--consent-ref` can be supplied
instead. `--single-walker-reviewed` attests a reviewed file or a supervised live scene
with one walker and no bystanders; the old pose backend cannot count people.

Camera enrollment is registered as `operational`, outside scientific splits. Query
captures have **no asserted subject identity**: the claim is not a ground-truth label.
Their source files and hashes are preserved under `data/raw/camera/`. Do not mix
operational enrollments with the same ID's study-role enrollment in one registry;
use a separate config/store for managed research.

**The default is still `POLICY_NOT_READY`, not TRUE/FALSE**, until a valid policy for
this exact Nano recipe/reference count is installed. A missing identity, bad gait or
startup error also produces a non-Boolean status. No threshold is invented here.
Physical Nano execution still needs the two startup checks on your actual device.

For an existing original video, add `--video` to either script. Already registered
originals work directly. For an unregistered original, explicitly add
`--register-original --consent` (or `--consent-ref`); never use that option for a
re-encoded/stabilized copy or window. For example:

```bash
python scripts/enroll.py --name Abhishek --video /path/to/reference.mp4 \
  --register-original --consent --single-walker-reviewed
python scripts/identify.py --identity Abhishek --video /path/to/different-query.mp4 \
  --register-original --consent --single-walker-reviewed
```

The query must be a different original from the claimed references. Unknown original
capture dates are retained as unknown for operational imports; registration time is
not fabricated as capture time. Managed study registration below still requires full
actual metadata and explicit development/calibration roles.

## 1. Put this version on the Nano

Start with an original Nano running **JetPack 4.6.x, Ubuntu 18.04, Python 3.6 and
CUDA 10.2**. Orin Nano is a different platform. NVIDIA's
[JetPack archive](https://developer.nvidia.com/embedded/jetpack-archive) and
[PyTorch Jetson compatibility list](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
are the installation references. The 4 GB Nano is the intended target; no memory
or real-time guarantee has been measured.

Use the complete working tree containing this migration. The local branch is
`feat/gait-verification-v2-jetson`; uncommitted changes do not appear in a fresh
GitHub clone. Copy the reviewed tree to the Nano, or commit/push it before checking
out that branch there. Do not use an older `master` checkout with this guide.

On the Nano, from the copied repository:

```bash
cd ~/advanced-gait-analysis-on-embedded-hardware
umask 077
cat /etc/nv_tegra_release
python3 --version
uname -m
sudo apt update
sudo apt install -y python3-venv python3-dev python3-matplotlib \
  libopenblas-base libopenmpi-dev ffmpeg v4l-utils git unzip wget
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install 'pip==21.3.1' 'setuptools==59.6.0' 'wheel==0.37.1'
python -m pip install --only-binary=:all: -r requirements-jetson.txt
```

Keep JetPack's system OpenCV 4.1.1 exposed by `--system-site-packages`. Verify it with
`python -c "import cv2; print(cv2.__version__)"`. If absent, restore the OpenCV
component from your matching JetPack installation; do not replace it with a pip wheel. The Nano requirements
are separate from the modern workstation requirements; install each on its stated
machine. Camera preview uses OpenCV, while verification uses FFmpeg for decoding.

## 2. Install the Nano framework wheels

Use NVIDIA's Python 3.6/aarch64 CUDA-enabled PyTorch 1.10 wheel from its compatibility
list. The link below is listed by NVIDIA; download availability and installation
must still be checked on the device. No torchvision is needed for GaitPT.

```bash
mkdir -p .cache/installers
wget -O .cache/installers/torch-1.10.0-cp36-cp36m-linux_aarch64.whl \
  https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl
python -m pip install --no-deps .cache/installers/torch-1.10.0-cp36-cp36m-linux_aarch64.whl
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()"
```

For MediaPipe 0.8.5 use the **experimental community Nano wheel**, documented by
[its publisher](https://github.com/PINTO0309/mediapipe-bin). This is an explicit
deployment dependency, not an official Google Nano wheel or a tested JetPack 4.6
certification. Download the release and select exactly the CUDA 10.2, Python 3.6,
aarch64, NumPy 1.19 wheel:

```bash
wget -O .cache/installers/mediapipe-v0.8.5.zip \
  https://github.com/PINTO0309/mediapipe-bin/releases/download/v0.8.5/v0.8.5.zip
unzip .cache/installers/mediapipe-v0.8.5.zip -d .cache/installers/mediapipe-v0.8.5
find .cache/installers/mediapipe-v0.8.5 -name '*cuda102*cp36*linux_aarch64.whl'
```

Install the displayed `numpy119x` wheel with `python -m pip install --no-deps
/exact/path/to/mediapipe-0.8.5_cuda102-cp36-cp36m-linux_aarch64.whl`.
`--no-deps` avoids replacing JetPack OpenCV. Then verify:

```bash
python -c "import cv2, numpy, mediapipe as mp; print(cv2.__version__, numpy.__version__, mp.__version__); assert mp.__version__ == '0.8.5'"
```

The runtime requires both `modules/pose_detection/pose_detection.tflite` and
`modules/pose_landmark/pose_landmark_lite.tflite` under the installed MediaPipe
package. It refuses implicit downloads. The lite model was absent in some older
wheels, whose original download URL now fails. If necessary, install this tagged
upstream asset **before calibration**:

```bash
POSE_PACKAGE=$(python -c 'import pathlib, mediapipe; print(pathlib.Path(mediapipe.__file__).parent)')
test -f "$POSE_PACKAGE/modules/pose_detection/pose_detection.tflite"
# Run only if the lite asset is missing; preserve any existing validated asset.
test -f "$POSE_PACKAGE/modules/pose_landmark/pose_landmark_lite.tflite" || \
  wget -O "$POSE_PACKAGE/modules/pose_landmark/pose_landmark_lite.tflite" \
  https://raw.githubusercontent.com/google-ai-edge/mediapipe/v0.8.9/mediapipe/modules/pose_landmark/pose_landmark_lite.tflite
```

This asset/graph combination still needs the on-device check below. Both asset
hashes enter the recipe; changing a wheel or asset invalidates its calibration.

## 3. Install the frozen GaitPT source and safe numeric weights

On the Nano, keep the upstream source unmodified and pinned:

```bash
mkdir -p .cache
git clone https://github.com/UbiquitousCognitiveSystems/GaitPT.git .cache/GaitPT
git -C .cache/GaitPT checkout 4977618a36ef77ee9a2863c9f8e27ab862894d70
```

Use the official **GREW** checkpoint linked by the
[GaitPT authors](https://github.com/UbiquitousCognitiveSystems/GaitPT). Required SHA-256:

```text
e6b287e2c2ae27a9344b8009077c9d4f70811d2d6f3ff20198b99bfa4f824637
```

Prepare a pickle-free bundle once on a **modern workstation** with Python >=3.10
and PyTorch >=2.6. This is offline artifact packaging; video/pose/inference never
uses that workstation. From this repository there:

```bash
python3 -m venv .venv-prepare
source .venv-prepare/bin/activate
python -m pip install -r requirements.txt
python scripts/prepare_gaitpt.py \
  --checkpoint /path/to/grew-checkpoint.pth \
  --output .cache/gaitpt-grew-npz
```

The exporter checks the original hash and uses `weights_only=True`, with no unsafe
fallback. Copy the resulting directory, containing `manifest.json` and `weights.npz`,
to the Nano's `.cache/gaitpt-grew-npz`. Do not deserialize `.pth` files on the Nano.
The numeric loader uses `allow_pickle=False`, checks archive limits, hashes, keys,
shapes and finite float32 arrays, and loads weights strictly. No training occurs.

GaitPT's source is **CC BY-NC-ND 4.0**. Preserve its license and attribution;
permission for the intended use and distribution remains a deployment gate. Bundles
are ignored local artifacts and are not published by these commands.

## 4. Check local inference and camera

Back on the Nano with its `.venv` activated:

```bash
python scripts/gait.py check --require-nano
python init_db.py
v4l2-ctl --list-devices
python scripts/test_camera.py --no-display
```

`check` requires original Nano hardware, checks the pinned artifacts and local pose
graph, runs the frozen model twice, and reports the measured embedding dimension
and recipe hash. It fails if CUDA is unavailable; there is no automatic CPU or remote
fallback. Keep the output for the device validation record. `tegrastats` can record
RAM, temperature and GPU use during actual enrollment/query processing.

For a USB camera, record independent walking takes locally:

```bash
python scripts/gait.py capture --output data/raw/S001_left45_take1.mp4 --seconds 30
python scripts/gait.py capture --output data/raw/S001_toward_take2.mp4 --seconds 30
```

Each command starts a separate V4L2 recording and refuses an existing output.
Capture uses source timestamps through FFmpeg without frame-rate resampling.
Use `--device /dev/video1` for another USB camera. CSI capture and camera settings
are described in [setup](docs/SETUP.md). Existing phone originals can be copied
unchanged into `data/raw/`; supported inputs have one video stream, ordered PTS,
31–7,200 frames, at most 120 seconds and up to 1920×1080 (including portrait rotation).
Keep the entire walker and feet visible. Review each video for one walker and no
bystanders; legacy MediaPipe cannot prove that condition automatically.

## 5. Advanced: register managed study originals and enroll

Register only genuine original takes, never stabilized/re-encoded copies or windows.
Use pseudonymous IDs and actual capture time/session/view/consent references. The
dates below illustrate the syntax; replace them with actual capture metadata.
Subjects used for development and calibration must be disjoint. Locked-final study
data belongs in the protected research workflow and is refused by this Nano port.

```bash
python scripts/gait.py register --video data/raw/S001_left45_take1.mp4 \
  --subject-id S001 --take-id S001_left45_take1 --session-id day1 --view left45 \
  --role development --consent-ref consent-S001 --captured-at '2026-10-20T10:00:00+05:30'
python scripts/gait.py register --video data/raw/S001_toward_take2.mp4 \
  --subject-id S001 --take-id S001_toward_take2 --session-id day1 --view toward \
  --role development --consent-ref consent-S001 --captured-at '2026-10-20T10:02:00+05:30'
python scripts/enroll.py --video data/raw/S001_left45_take1.mp4 \
  --subject-id S001 --single-walker-reviewed
```

To add references, record/register/enroll **different original takes**. One reference
is stored per take; rolling windows are never independent enrollments. The store
derives a normalized centroid from that claim's walk embeddings. No scientifically
validated minimum reference count has been selected. The example holds the whole
second take out as a query. Four views on one day do not establish different-day
reliability.

## 6. Verify an explicit claim from an already registered original

```bash
python scripts/verify.py --video data/raw/S001_toward_take2.mp4 \
  --claimed-id S001 --single-walker-reviewed
```

To test a wrong claim, supply another **enrolled** ID in the same study role. It
compares only that claim's references. It never returns whose gait the video may
actually contain. Reusing a claimed reference take as its query is refused.

| Status | Verdict | Meaning |
|---|---|---|
| `OK` | `true` | TRUE: gait supports the supplied claim |
| `OK` | `false` | FALSE: gait does not match the supplied claim |
| `CLAIM_NOT_ENROLLED` | `null` | No enrollment compatible with this recipe |
| `INSUFFICIENT_GAIT` | `null` | Too little usable full-body walking evidence |
| `INVALID_INPUT` | `null` | Bad claim, unregistered input, source reuse or invalid lineage |
| `PROCESSING_ERROR` | `null` | Local decoder/model/store/runtime failure |
| `POLICY_NOT_READY` | `null` | No approved threshold/evidence for this exact Nano recipe |

Exit code is `0` for either Boolean verdict, `2` for a non-decision or command error.
Parse `verdict` to distinguish TRUE from FALSE. Public events in
`outputs/verification-events/` contain no scores, candidate list or other identity.
Operator failures go to `outputs/logs/verification.log`. New artifacts are private
by filesystem permission, not encrypted or authenticated.

## 7. Calibrate and validate before enabling Boolean decisions

The default `policy: null` is intentional. Use consenting development/calibration
recordings to validate the **Nano MediaPipe recipe**, then freeze an operating point
with subject/take separation and coverage reporting. A research RTMPose threshold,
workstation recipe, legacy threshold or source benchmark cannot substitute.
Optional `--private-diagnostics-dir outputs/private-scores` on `verify.py` writes
restricted, recipe-bound score records for authorized offline evaluation; stdout
and public events remain score-free. It does not fit a threshold.

The [policy contract](docs/DEPLOYMENT.md) describes the operator-approved evidence
bundle that can enable TRUE/FALSE. No arbitrary operational threshold is installed.
Physical Nano performance, quality-policy validation, representative pilot,
new-cohort comparison/calibration, licensing and final evaluation are still required.
GaitPT is the first integrated candidate; GPGait++/GPGait and preserved E0 remain
comparative research systems, not selected winners or silently substituted Nano backends.

## Development and historical material

On the modern workstation: `python -m pytest` after installing `requirements.txt`.
The active runtime retains Python 3.6 syntax/API compatibility. CPU is an explicit
engineering option in a separate config; it is not the Nano CUDA acceptance test.

The old active implementation, commands/config/docs and standalone checkpoint
diagnostics are archived under [legacy/](legacy/README_ARCHIVE.md). The old
`models/`, committed checkpoint, `OLD_README.md`, phase reports, old database and
existing outputs remain historical assets. Old templates cannot be migrated into
GaitPT embeddings; re-enroll independent originals. `scripts/identify.py` now serves
as a claim-only verification alias. Start with the two-script commands above.
