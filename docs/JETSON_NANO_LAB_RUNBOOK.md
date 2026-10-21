# Jetson Nano lab runbook — 22 October 2026

**The final gait model does not exist yet.** Today's session prepares and checks the hardware,
camera, pose pipeline, database, display and model-installation slot. It does **not** enroll
real people, does **not** produce TRUE/FALSE, and must not be presented as biometric recognition.
Nothing below was executed on a Jetson when this file was written: every command is
*prepared but target-device-unverified* until you run it and record the result.

Conventions: run from the kit root (`~/gait-lab/jetson-lab-kit-*`), in the Nano virtual
environment from `README.md` §1–2 (`source .venv/bin/activate`). `PASS/FAIL/WARNING/
NOT_TESTED/BLOCKED` are the only result words used by the tools. Nothing here installs,
upgrades or removes a system package; if a check fails, stop and read Stage 9.

## Stage 0 — Receive and verify the kit (laptop → Nano)

```bash
# USB: copy jetson-lab-kit-*.tar.gz and its .sha256, then on the Nano:
sha256sum -c jetson-lab-kit-*.tar.gz.sha256 && tar xzf jetson-lab-kit-*.tar.gz
# SSH instead:  scp jetson-lab-kit-*.tar.gz* <user>@<nano-ip>:~/
cd jetson-lab-kit-* && python3 scripts/build_lab_kit.py verify .     # "ok": true
```

## Stage 1 — Hardware

```bash
cat /proc/device-tree/model; echo          # expect: NVIDIA Jetson Nano Developer Kit (4 GB, not 2 GB)
cat /etc/nv_tegra_release                  # expect: R32 ... REVISION: 6.x or 7.x  (JetPack 4.6.x)
grep -E '^(NAME|VERSION_ID)=' /etc/os-release   # expect Ubuntu 18.04
dpkg-query -W nvidia-jetpack               # optional: expect 4.6.x
free -m; df -h .                           # ~3.9 GB RAM; keep several GB free
ls /sys/class/drm 2>/dev/null; echo "DISPLAY=$DISPLAY"   # HDMI connector / desktop session
```

Record the outputs; they are the first physical evidence that the assumed profile is real.

## Stage 2 — Environment (the one-command preflight)

```bash
python3 --version                          # expect 3.6.x
python3 scripts/lab.py preflight           # read-only; saves JSON + Markdown
python3 scripts/check_nano.py --skip-camera   # the stricter pre-existing package check
```

`preflight` writes `outputs/lab-reports/preflight-<time>.{json,md}` and exits 2 only on a FAIL.
Expected today: hardware checks PASS or FAIL with a named cause; `model_artifacts` is
**BLOCKED (MODEL_NOT_DEPLOYED)** — that is correct, not an error. `NOT_TESTED` on the
Nano means a probe could not run: investigate it. Likely first-run findings: wheel versions
(Stage 2 of `README.md`), `v4l-utils` missing (WARNING-level for modes), no desktop session.

## Stage 3 — Camera (nothing is recorded unless you run `record`)

```bash
v4l2-ctl --list-devices                    # if v4l-utils is installed
python3 scripts/lab.py camera list         # nodes + pixel formats/sizes/rates (no frames are read)
python3 scripts/lab.py camera record --seconds 5      # explicit, operator-triggered, 1–30 s
python3 scripts/lab.py camera analyze --video data/diagnostics/camera/<clip>.mp4
python3 scripts/test_camera.py --no-display           # optional OpenCV open/grab test
```

`record` writes only to `data/diagnostics/camera/` (never `data/raw/`, never the recording
registry) plus a `.diag.json` report with resolution, frame count, timestamps, measured fps,
gaps, decode result and findings such as `LOW_FPS`, `TIMESTAMP_GAPS`, `NEAR_BLACK`,
`FROZEN_IMAGE`. Use a test object or yourself with **no one else** in view; do not film
participants for this stage. A WARNING is a finding to read, not a failure of the Nano.
`test_camera.py --no-display` saves one still frame under `outputs/`; delete it afterwards if it
shows anyone. To play a clip: `ffplay <clip>` (needs a display). `check_nano.py --capture-test`
(README, daily use) additionally exercises the production capture path.

## Stage 4 — Pose (engineering diagnostic only)

```bash
python3 scripts/lab.py pose check
python3 scripts/lab.py pose extract --video data/diagnostics/camera/<clip>.mp4 \
    --overlay outputs/lab-reports/overlay-1.mp4
```

`pose check` loads the MediaPipe 0.8.5 lite graph and states the backend name, version and
asset hashes. `pose extract` reports frames detected, per-joint mean visibility, in-frame
fraction, ankle motion, the longest fully-observed run and whether the *provisional* Nano
quality gate would pass — it never runs a gait model. Open the overlay and **look at it**:
visibility scores do not prove the skeleton is anatomically right, and the report says
`anatomical_accuracy: NOT_ASSESSED`. Use a clip with a person walking if you have consent
from that person; otherwise a static scene only proves the graph loads. If MediaPipe
fails to start (EGL/GL), try from a desktop session — see `docs/SETUP.md`.

## Stage 5 — Database (no identities are created)

```bash
python3 scripts/lab.py db check            # state ABSENT before first use (the file is not created)
python3 scripts/lab.py db init             # creates schema v2, mode 0600, zero rows
python3 scripts/lab.py db check            # state READY, references 0, integrity ok
```

The database is `data/verification-v2.db`. Do **not** run `enroll.py` today and do **not**
insert test participants: a fake enrollment in the operational database would later look
real. (Synthetic embeddings are used only in isolated temporary databases by the unit tests.)

## Stage 6 — Display (HDMI)

```bash
python3 scripts/lab.py display status --render text                    # any terminal
python3 scripts/lab.py display status --render window --watch 5       # full-screen on HDMI, needs a desktop
python3 scripts/lab.py display status --render png --output outputs/lab-reports/status.png
```

Expected screen: `NOT READY: MODEL NOT DEPLOYED`, with rows for hardware (latest preflight),
camera, last capture, model, database, enrollment and verification. Run `preflight` first so
the hardware row has content. Window mode needs `DISPLAY`; over plain SSH use `png`/`text`
(this is a NOT_TESTED, not a FAIL, of the physical LCD). Check legibility from the intended
viewing distance on your actual panel (the renderer assumes 800×480; no layout is fitted to
your model of LCD). Result screens for later: `display result --event <events/*.json>`.

## Stage 7 — Future model slot (infrastructure only)

```bash
python3 scripts/lab.py model status        # state MODEL_NOT_DEPLOYED, no packages, baseline labeled
```

Expected: `state: MODEL_NOT_DEPLOYED`, `active: null`; if `.cache/gaitpt-grew-npz` is present it
appears as `development_baseline` — `development_only: true`, approval FAIL, never active.
The staged workflow (discovered → structurally valid → runtime compatible → parity tested →
approved → active) is exercised later, with a real package, via
`docs/FINAL_MODEL_ACTIVATION_RUNBOOK.md`. If `pytest` is already installed you may run
`python3 -m pytest -q tests/verification` on the Nano as an extra regression signal (not
required, and not pinned in `requirements-jetson.txt`). `enroll.py` refuses with status
`MODEL_NOT_DEPLOYED` (null verdict) before touching the camera, registry or database, and
`identify.py` returns `CLAIM_NOT_ENROLLED` for an unenrolled claim, else `MODEL_NOT_DEPLOYED`
(workstation-tested only); do not run them with `--consent` today — it attests a recording.

## Stage 8 — Evidence

```bash
ls -l outputs/lab-reports data/diagnostics/camera data/diagnostics/pose
sha256sum outputs/lab-reports/* data/diagnostics/camera/* > outputs/lab-reports/evidence.sha256
tar czf ~/nano-evidence-$(date +%Y%m%d).tar.gz outputs/lab-reports data/diagnostics
```

Copy the archive to USB or `scp` it to the laptop. Keep only clips whose subjects consented to
engineering tests; delete the rest (Stage 9). Hardware-measured facts to bring back: board,
L4T/JetPack, Python, torch/CUDA, RAM, disk, camera modes, measured camera fps, pose graph
load result, display behavior.

## Stage 9 — Shutdown, cleanup and troubleshooting

```bash
fuser -v /dev/video0                      # who holds the camera (stale ffmpeg?)
rm -i data/diagnostics/camera/*.mp4        # diagnostic clips only; never touch data/raw
sync && sudo shutdown -h now
```

| Symptom | Likely cause / action |
|---|---|
| preflight `board` FAIL | not an original Nano (Orin/Xavier/2 GB) — the profile does not apply |
| `cuda` FAIL | PyTorch wheel does not match JetPack/Python; no CPU fallback exists. Do not upgrade blindly |
| `usb_camera` FAIL, no `/dev/video*` | re-seat the USB camera; `dmesg | tail`; different port |
| node present, `no access` | add the user to `video` group (`sudo usermod -aG video $USER`, re-login) |
| `camera record` exit code / `CAPTURE_FAILED` | device busy or format unsupported; see `camera list` modes; configure `camera:` in `config.yaml` |
| `LOW_FPS` / `TIMESTAMP_GAPS` | USB bandwidth/lighting auto-exposure; try 640×480 and good light |
| `pose check` FAIL | MediaPipe is not exactly 0.8.5 or the lite assets are missing (`docs/SETUP.md`) |
| `display` window error | no `DISPLAY`: use a desktop session or `--render png` |
| disk WARNING | free space; recordings are ~tens of MB per 30 s |

Do not hand-edit the SQLite file, `active.json`, or approval/parity records. If something
looks wrong, deactivate (`model deactivate`), keep the evidence, and report the exact output.

## What today does NOT establish

No gait recognition, no accuracy, no threshold, no real-time claim, no proof that
MediaPipe 0.8.5 matches the research pose estimator (RTMPose), and nothing about the final
model. Those are later milestones (`FINAL_MODEL_ACTIVATION_RUNBOOK.md`).
