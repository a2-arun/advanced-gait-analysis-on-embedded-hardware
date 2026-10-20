# Nano rearchitecture implementation and validation record

20 October 2026. Branch: `feat/gait-verification-v2-jetson`.
Baseline: `773207e`. Scope: post–Phase 2C Nano engineering port; no training,
threshold fitting, model selection, new data collection or final-study evaluation.
Changes are local and uncommitted; nothing has been pushed or deployed.

## Result and changed areas

The active application is now 1:1 claimed-identity verification with a complete local
capture/decode/pose/frozen-GaitPT/reference/comparison path intended for original Nano.
The supplied claim alone selects references. Usable enrolled inputs remain
POLICY_NOT_READY/null until compatible trusted calibration evidence is installed.
No legacy identification, arbitrary threshold, remote inference or other-person
output is used by the new runtime. Physical Nano compatibility remains unverified.

- `src/verification/`: safe weights, timestamped decoding, native skeleton contract,
  provisional quality, raw registry, independent references, claim-only scoring,
  policy/evidence gate, public events and opt-in private diagnostics.
- `scripts/gait.py`, `enroll.py`, `verify.py`, `prepare_gaitpt.py`, `identify.py`:
  local commands and explicit retirement of identity search; `identify.py` now verifies only an explicit claim. `test_camera.py` now
  creates private unique snapshots, handles missing output directories and avoids
  overwriting an existing user snapshot. `init_db.py` initializes the separate V2 store.
- `config.yaml`, both requirements files, `.gitignore`, `pytest.ini`: versioned
  Nano configuration, separated platform dependencies, private artifacts and tests.
- `README.md`, architecture/setup/deployment docs, ADR-001, copied research contracts
  with matching Nano owner-requirement amendments, `AGENTS.md` and `CLAUDE.md`:
  current startup instructions, design/trust boundaries and remaining gates.
- `legacy/`: original application modules/schema, standalone diagnostics, prior
  commands/config/docs and modified utility originals preserved byte-for-byte.

## Executed tests and commands

Development interpreter was
`../deepfake-detection/.venv-gait-v2-cuda/bin/python` (Python 3.11.17), reused read-only
as a tool environment. The target Nano stack was not installed or simulated here.

```bash
../deepfake-detection/.venv-gait-v2-cuda/bin/python -m pytest -q
../deepfake-detection/.venv-gait-v2-cuda/bin/ruff check src tests scripts init_db.py
git diff --check
```

**42 tests passed, none skipped (latest two-script/compatibility continuation).** Ruff and whitespace checks passed. Tests cover
native GREW mapping/height scaling/repeated nose, contiguous/missing/cropped/static/
discontinuous gait, invalid vectors, claim isolation, original uniqueness, private
permissions, source/role/owner validation, protected source/log aliasing, policy
absence/integrity/recipe/reference-count binding, and synthetic TRUE/FALSE behavior.
Actual FFmpeg-generated videos test every-frame PTS, portrait rotation and short-video
abstention. The command-level fixture runs registration, enrollment, two explicit
claims, private diagnostics, default POLICY_NOT_READY and synthetic Boolean policies
through the actual registry/store/CLI/event code. Its encoder and calibration evidence
are fixtures; those Boolean results are not biometric accuracy evidence.

NPZ tests reject object/pickle arrays, nonfinite values and oversized declared tensor
shapes **before allocation**. Header and archive bounds are checked before loading
numeric arrays. CLI `--help` commands were exercised for gait/enrollment/verification/
artifact preparation. `identify.py` now exposes verification help and refuses an omitted claim;
it is not a gallery-search endpoint.

The first failing tests exposed missing implementation, then command setup ordering,
short-video status, reference-count policy compatibility, source/log aliasing, camera
snapshot handling and array-header allocation guards. All those regressions were
fixed before the final passing run. Existing logger's unused import was removed
for the full active-source lint check, with its original archived.

A syntax audit parsed **27 active Python files with Python 3.6 grammar**, including
scripts and camera helpers. This is a syntax check, **not execution on Python 3.6**.
Compatibility code avoids modern annotation syntax and newer subprocess/path APIs;
timezone offsets are normalized for Python 3.6's `strptime` behavior.

Historical diagnostics were standalone scripts that used unrestricted pickle and
model arguments without pytest fixtures. They were archived unchanged; they are
not counted as passing V2 pytest tests. No claim of old-checkpoint parity or replay
of its reported accuracy is made.

## Actual frozen-model evidence

The audited official GREW checkpoint already present in the research cache was
converted with `scripts/prepare_gaitpt.py`, restricted `weights_only=True` and the
fixed original hash. Prepared local artifacts are ignored under
`.cache/gaitpt-grew-npz`; no model or user footage was uploaded. No optimizer ran.

| Item | Observed result |
|---|---|
| Source revision | `4977618a36ef77ee9a2863c9f8e27ab862894d70` |
| Original checkpoint SHA-256 | `e6b287e2c2ae27a9344b8009077c9d4f70811d2d6f3ff20198b99bfa4f824637` |
| Loaded parameters | 4,484,384 |
| Measured embedding dimension | 256 |
| Repeated CPU embedding maximum absolute difference | 0.0 |
| NPZ versus restricted checkpoint CPU embedding maximum absolute difference | 0.0 |
| CPU versus workstation CUDA maximum absolute difference | 0.00000414043 |
| `einops 0.8.1` versus `0.4.1` CPU maximum absolute difference | 0.0 |
| Frozen state | eval mode, every parameter `requires_grad=False`, no-grad inference |

These checks used a deterministic synthetic native `(2,30,18,2)` batch from
`numpy.random.RandomState(7).uniform(0,1,...)`, not participant footage. The workstation
had PyTorch 2.10.0+cu128, NumPy 2.4.6 and an RTX 3050 6 GB laptop GPU. An isolated,
git-ignored `einops==0.4.1` install was tested alongside the environment's 0.8.1
without changing the shared environment. The exact same CPU/CUDA differences were
observed with both versions. The final hardened NPZ loader also loaded and ran the
real model successfully on CPU. PyTorch reports the upstream nonfatal
`norm_first`/nested-tensor warning on this modern stack; upstream source was unchanged.

For a minimal workstation reproduction after preparing the bundle and installing
the modern requirements, run from the repo with the actual pinned source location:

```python
import numpy as np
import torch
from src.verification.backend import LocalGaitPT

source = '/path/to/pinned/GaitPT'
encoder = LocalGaitPT(source, '.cache/gaitpt-grew-npz', 'engineering-only', 'cpu')
batch = np.random.RandomState(7).uniform(0, 1, (2, 30, 18, 2)).astype(np.float32)
a, b = encoder.encode_batch(batch), encoder.encode_batch(batch)
assert np.allclose(a, b, atol=1e-6)
assert not any(p.requires_grad for p in encoder.model.parameters())
print(encoder.embedding_dim)
```

On the actual Nano use `python scripts/gait.py check --require-nano` instead;
that also requires local MediaPipe assets/graph and the intended device/runtime.
No real MediaPipe 0.8.5 video extraction or physical Nano run occurred here.

## Preservation

A Git-baseline byte comparison verified **42 historical or archived files**, including
all moved implementation/diagnostic files, archived modified originals, old models,
checkpoint, `OLD_README.md` and phase reports. Existing user outputs and old database
were not deleted or migrated; V2 uses separate paths. The embedded legacy checkpoint
still has SHA-256:

```text
345f58e6429c8456ea4e937ab5715378b8e5a7285b785c09ef87803c7d6a9904
```

It differs from the audited research shipped E0, and cannot stand in for the absent
controlled LOSO E0 bundle. This task did not modify that research model or historical
experiments. The old schema/modules now live under `legacy/`, with old database data
left in place. Historical reproduction should use an isolated baseline checkout.

## Open requirements and blockers

1. **Physical original Nano:** install and exercise NVIDIA PyTorch 1.10/Python 3.6,
   the experimental community MediaPipe wheel, its graph/assets, camera/CSI/V4L2,
   actual video decoding, memory/latency/thermal behavior and repeatability. Latest
   workstation CUDA and Python syntax checks cannot close this gate.
2. **Pose/input quality:** validate full-body conversion visually on representative
   walking videos and the four-view moving-camera pilot. The Nano gate is provisional
   and depends on operator review for the single-walker/no-bystander assumption.
   No source-dataset pipeline reproduction or liveness guarantee is established.
3. **Scientific evidence:** collect the planned consent-backed pilot/cohort, preserve
   originals and split roles, compare E0/GaitPT/GPGait using common research trials,
   validate this separate Nano pose recipe, calibrate on designated subjects, freeze
   a policy/reference count, and perform authorized final evaluation. No threshold,
   winner, new-cohort accuracy or different-day reliability was selected here.
4. **License/checkpoint deployment:** the exact GREW checkpoint was available locally
   and loaded safely. GaitPT CC BY-NC-ND 4.0 / checkpoint use and distribution permission
   remain to be cleared. Community wheel and NVIDIA wheel installation/downloads
   were documented from their publishers, not fully downloaded/tested on hardware.
5. **Trust/security:** local registration/approval files are trusted operator assertions;
   hashes are not authentication. No encrypted/signed storage, automatic derivative
   discovery, multi-writer coordination, power-loss durability certification or full
   research locked-final controls are claimed. See DEPLOYMENT.md for limits.

No neural training, calibration fit, pilot recording, dataset evaluation, cloud
inference, commit, push or deployment was performed by this migration.


## Two-script and confirmed-profile continuation — 20 October 2026

The owner confirmed fresh USB-camera capture by default and the exact original-Nano
stack: JetPack 4.6.x, Ubuntu 18.04, Python 3.6 and CUDA 10.2. `enroll.py --name` and
`identify.py --identity` now support that daily workflow. Consent-backed operational
records register automatically, enrollment commits to the same file-backed database
used by the next invocation, and query records retain no asserted true subject ID.
V1 study records remain readable; new registry writes use nano-recordings-v2.

New tests exercise camera defaults, persistent/reopened database references, explicit
claim aliases, unknown claim without camera activation, missing consent, bad gait
without a saved reference, model startup before capture, unique/private USB recordings,
optional original-video imports, source reuse refusal, version-conflict diagnostics,
and temporary camera encode/decode cleanup. Camera/pose fixtures are labeled; no
physical capture occurred. Both camera scripts preflight the actual model and pose
graph before recording, so known startup failures do not waste a walk.

A separate-process experiment invoked the actual `scripts/enroll.py` and
`scripts/identify.py` from a different working directory with a temporary config,
using the real frozen GaitPT and synthetic camera/pose boundaries. Enrollment saved
a finite normalized 256-D reference; a fresh process reopened it and produced
POLICY_NOT_READY/null for the claim. It created a different query source, retained
the query's null subject, and did not add a query to the enrollment database. This
is persistence/inference engineering evidence, not actual Nano or gait-accuracy proof.
Synthetic policy tests separately verify TRUE and FALSE with the same claim-only API.

`check_nano.py --capture-test` checks the actual installed target profile, imports,
CUDA/tool/camera access and a private temporary 2-second USB encode/decode. It does
not install packages or mutate the system. The checker correctly rejected this x86
workstation's modern stack. Actual Nano success is still unobserved. On aarch64,
operational runtime startup also validates the profile before live capture.

[Package audit](NANO_PACKAGE_AUDIT.json) records all nine pinned PyPI releases'
published Python constraints, cp36/aarch64 or universal wheel names, sizes and SHA-256.
Their selected wheel metadata has no unsatisfied active runtime Requires-Dist
constraints for the target. Einops omits Requires-Python metadata but its publisher
specifies Python >=3.6, and its 0.4.1 source/inference was separately checked. Optional
extras were excluded. These facts do not validate the external NVIDIA/community
wheels, system OpenCV/Matplotlib, native ABI, camera driver, or physical CUDA execution.
A bulk cross-platform wheel download was stopped after stalling; this record claims
metadata verification, not a completed target binary installation.

Nano instructions now request binary-only pinned PyPI installs, retain JetPack OpenCV
4.1.1, keep system Matplotlib via the venv, and avoid pip dependency resolution for
the separately installed NVIDIA/MediaPipe wheels. Workstation requirements and safe
`.pth` conversion remain separate tools; the Nano operational runtime uses numeric
weights and older APIs. Do not install the modern requirements on the Nano.

No test failures remain in the executed software suite. Physical target startup,
representative gait/quality validation, approved calibration, licenses and the
scientific requirements listed above remain open. No reliable Boolean threshold
was fabricated to make the demonstration appear ready.
