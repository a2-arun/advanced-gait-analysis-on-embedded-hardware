# Jetson Nano pre-lab readiness report

Date: 21 October 2026 (lab session 22 October 2026). Branch `feat/gait-verification-v2-jetson`,
base HEAD `030ba2c`, work uncommitted. Workstation: x86_64, Python 3.11.17 (research checkout's
`.venv-gait-v2-clean`, borrowed read-only), no Jetson, no Python 3.6 interpreter.

**Scope.** The final gait model does not exist. This report covers hardware, camera, pose,
database, display and model-installation infrastructure only. **No biometric recognition is
demonstrated or claimed.** Nothing below ran on a Jetson.

## 1. IMPLEMENTED AND WORKSTATION-TESTED

Evidence = automated tests on the workstation (246 collected: **245 passed, 1 skipped**; baseline
before this work was 81 passed + 1 skipped). "Tested" means logic with synthetic fixtures and
isolated temporary files — not hardware behavior.

| Capability | Where | Notes |
|---|---|---|
| One-command preflight (17 checks, PASS/FAIL/WARNING/NOT_TESTED/BLOCKED, scope per check, JSON + Markdown, private files, never overwrites, no install, no camera open) | `scripts/lab.py preflight`, `readiness.py` | On a non-Jetson every `requires-nano` check is NOT_TESTED, never PASS. Executed on this laptop: hardware INCOMPLETE, model BLOCKED |
| Model-free camera diagnostics: node/mode listing, explicit short recording (1–30 s), playability/decode/resolution/timestamp/fps/gap/black/frozen/short-clip analysis, separate storage, never registered | `lab.py camera list|record|analyze`, `camera_diag.py` | Analysis tested on real ffmpeg-generated clips; recording tested through a fake capture function. **Real V4L2 recording was never run** |
| Independent pose diagnostics: backend registry, load check, coverage/visibility/in-frame/jitter/longest-run report, provisional gate verdict, optional overlay video, hash of the chosen clip | `lab.py pose check|extract`, `pose_diag.py` | Tested with a scripted synthetic backend. Report states `anatomical_accuracy: NOT_ASSESSED`. The real MediaPipe 0.8.5 backend cannot load here (0.10.32 installed) and correctly reports FAIL |
| Model slot with six independent stages, install/inspect/parity/activate/deactivate, fail-closed active-model resolution, no fallback, development baseline never approvable | `lab.py model ...`, `model_slot.py` | 24 + 10 tests; only the official GaitPT-GREW + MediaPipe 0.8.5 recipe can reach `runtime_compatible = PASS` |
| `MODEL_NOT_DEPLOYED` / `MODEL_INCOMPATIBLE` non-decisions; enroll refuses before camera/registry/database access; `--development-baseline` explicit and labeled | `cli.py`, `service.py` | Public response keys unchanged; two new status values |
| Database readiness: read-only status, explicit init (zero rows), schema/integrity/legacy-inert reporting, connections closed after each store operation (fixed a connection leak), isolated-temp-DB tests for validation, duplicate source/take, stale-model rejection, multi-reference | `lab.py db init|check`, `dbcheck.py`, `store.py` | Schema v2 and claim-only reads unchanged. No synthetic participants are ever written to an operational database |
| HDMI/LCD screens: status (hardware, camera, capture, model, database, enrollment, verification) and result screens; TRUE/FALSE only for `OK` + real boolean; text/PNG/full-screen-window renderers; bounded `--watch` | `lab.py display ...`, `display.py` | Text/PNG tested; window mode only tested for its no-DISPLAY refusal |
| Portable lab kit builder with allowlist, forbidden-content guard (databases, media, weights, wheels, keys, secret patterns), SHA-256 manifest, tar.gz + checksum, verify command usable on the Nano | `scripts/build_lab_kit.py` | Real kit built and verified on the workstation (§9) |
| Python 3.6 source checks | preflight check + tests | AST grammar at `feature_version=(3,6)` for 56 files, plus a lint banning 3.7+ APIs (it caught a real `add_subparsers(required=…)` break, now fixed). **Not executed under Python 3.6** |
| Regression: all 81 pre-existing tests still pass; 2 existing test fixtures got a one-line `allow_development_model` opt-in because enroll/verify now require an activated model | `tests/verification/` | |

## 2. PREPARED BUT TARGET-DEVICE-UNVERIFIED

Everything in `JETSON_NANO_LAB_RUNBOOK.md`, in particular:

* board / L4T / JetPack / Ubuntu / Python / torch / CUDA / RAM facts (the assumed profile is from the
  owner's confirmation, never read from the device);
* real USB camera node discovery, `v4l2-ctl` mode listing, ffmpeg V4L2 recording at 640×480/30,
  measured fps, timestamp quality and decode of the real clip;
* MediaPipe 0.8.5 community wheel load, EGL/GL behavior, pose assets, real pose-extraction output;
* HDMI behavior: DISPLAY detection, full-screen OpenCV window, legibility at 800×480 on the
  real LCD (layout is not fitted to any specific panel);
* SQLite file creation and permissions on the Nano's filesystem;
* peak memory/latency of anything (nothing was measured; no real-time claim);
* Python 3.6 execution of the new modules, and the JetPack OpenCV 4.1.1 / numpy 1.19.4 combination
  with the new code paths.

## 3. BLOCKED BY MISSING DEPENDENCY

| Item | Status on the build machine |
|---|---|
| aarch64 / cp36 wheels for numpy, protobuf, PyYAML, einops, absl-py, attrs, six, typing_extensions, dataclasses | **MISSING offline**: `.cache/nano-dependency-wheels/` is empty. Names, sizes and SHA-256 are in the kit's `DEPENDENCIES.md` (from `.cache/nano-package-metadata-audit.json`); they must be downloaded on a networked machine. Nothing was downloaded here |
| NVIDIA PyTorch 1.10 (JetPack 4.6 / py3.6 / aarch64) | **MISSING** (not in `.cache/installers`, not downloaded) |
| MediaPipe 0.8.5 aarch64 community wheel + pose `.tflite` assets | **MISSING** |
| Pinned official GaitPT source `.cache/GaitPT` | **MISSING** (only the NPZ bundle exists) — needed for any gaitpt-family package's `validate_artifacts` and `gait.py check` |
| `v4l-utils`, ffmpeg with libx264 on the Nano | Unknown until Stage 1/2; apt packages, not packaged |
| `docs/DEPLOYMENT.md`, `docs/NANO_MIGRATION_VALIDATION.md` | Referenced by existing docs/AGENTS context but **do not exist** in this checkout; not fabricated |
| Python 3.6 interpreter on the workstation | Absent; no 3.6 execution possible here |
| Model-only parity test (`GAIT_PARITY_GAITPT_SOURCE`) | Still skipped: no pinned GaitPT clone available to this run |

## 4. WAITING FOR FINAL RESEARCH MODEL

* The model itself, its exported package (`contract.json`, `manifest.json`, `weights.npz`), the
  research-owner `approval.json`, and a parity fixture for it.
* A `GaitBackend`, contract builder and `expected_contract` branch for any model other than the
  pinned GaitPT-GREW recipe; a pose backend if the final pose estimator is not MediaPipe 0.8.5
  (RTMPose on the Nano is not demonstrated).
* Nano numerical parity (`lab.py model parity`) and paired real-video preprocessing evidence.
* Real enrollments under the active recipe (a reprocessing command for preserved originals does
  not exist yet), and any end-to-end verification. Today `MODEL_NOT_DEPLOYED` is the expected state.

## 5. WAITING FOR CALIBRATION EVIDENCE

* No decision policy, threshold or calibration evidence exists or was created; `config.yaml: policy`
  stays `null`; usable claims will return `POLICY_NOT_READY` with a null verdict.
* No TRUE/FALSE demonstration mode exists. The display shows TRUE/FALSE only for a service result
  with status `OK` and a real boolean.

## 6. Preflight facts from this workstation (illustration, not Nano evidence)

`python scripts/lab.py preflight`: PASS — disk, ffmpeg (libx264+v4l2), ffprobe, database
directory, Python 3.6 grammar; BLOCKED — `model_artifacts` (MODEL_NOT_DEPLOYED; a development
baseline bundle is present and labeled); NOT_TESTED — the other 11 (`requires-nano`).

## 7. Restrictions honored

No training, calibration, threshold, 1:N identification, remote inference, model download,
dependency install, camera access, database/enrollment mutation, or commit/push/reset/clean/branch
switch. The research repository was only read (`HEAD 1bce211`, `origin/feat/gait-verification-v2`
`eabfb08`; its pre-existing ` M .gitignore` and `?? outputs/gait_verification/` are unchanged).
Recipe-defining modules (`backend.py`, `media.py`, `contract.py`, `deployment.py`) are unmodified, so
recipe hashes did not change.

## 8. Known limitations of this work

* `lab.py model status` and `preflight` import torch/mediapipe/cv2 (several seconds and
  hundreds of MB on a 4 GB Nano); `display status` avoids this until a model is activated.
* Parity/approval/activation records are operator-controlled files like the policy files; they
  are validated and hash-bound but not cryptographically signed.
* A status display does not prove readiness: it reflects checks that ran, and `ACTIVE` re-runs
  the qualification, but the display itself is not a safety interlock.
* Free-disk, RAM and fps thresholds are operational guidance, not validated requirements.

## 9. Kit

Built with `python scripts/build_lab_kit.py build --output dist/jetson-lab-kit-20261021 --archive`
(location and hashes are recorded in the final handoff message and the kit's `MANIFEST.json`).
It contains source, tests, config template, docs and manifests; it contains no model, database,
recording, dataset, key or wheel.
