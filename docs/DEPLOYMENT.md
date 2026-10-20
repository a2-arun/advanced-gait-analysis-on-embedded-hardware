# Local policy and deployment contract

Engineering code exists; physical Nano and scientific/deployment readiness remain
unvalidated. Run [README](../README.md) startup checks on the actual Nano before
calling the stack compatible. Record OS/JetPack/framework/pose asset versions,
recipe hash, pose graph check, actual-video results, latency, memory/temperature,
quality coverage and failure cases. Workstation CUDA evidence is not Nano evidence.

## How a Boolean becomes enabled

`policy: null` is the default. A cosine is not a probability, and neither `0.7737`
nor a convenient synthetic threshold is authorized as an operating policy. Validate
this exact Nano MediaPipe/GaitPT recipe on consenting development/calibration data,
select the operating point without locked-final data, and retain the split/trial
manifests and evaluation report, including coverage and subject-cluster uncertainty.
No threshold-fitting command is included in this no-training engineering task.

Authorized offline scoring can opt into
`--private-diagnostics-dir outputs/private-scores`. Private records contain the query
and claimed reference take/hash set, recipe, role, raw cosine, status and verdict.
They are not public events and not verification accuracy evidence on their own.
Prespecify claim trials and source/reference sets, join ground-truth labels from the
restricted capture registry, and report every submitted trial/abstention. Keep all
windows/derivatives of one source in the same role. The research evaluator remains
the home of comparative E0/GaitPT/GPGait experiments; this port does not replicate
its protected locked-final workflow. A Nano diagnostic record is a new schema,
not a drop-in research trial/cache artifact.

After genuine validation, the trusted operator may supply a `nano-policy-v1` JSON
file and point `config.yaml`'s `policy` at it. Required fields:

| Field | Contract |
|---|---|
| `schema` | `nano-policy-v1` |
| `recipe` | Exact hash from the actual Nano `recipe`/`check` command |
| `threshold` | Finite numeric global cosine threshold in [-1,1], from calibration |
| `reference_count` | Exact positive integer number of independent references used in calibration |
| `evidence` | Path to the approved evidence JSON, relative to the policy file |
| `evidence_sha256` | Canonical JSON SHA-256 of that evidence object |

The evidence requires `schema: nano-calibration-evidence-v1`, the same `recipe`,
`threshold` and `reference_count`, `split_role: calibration`, `validation_status: APPROVED`, nonempty
`approved_by`, and actual timezone-aware `approved_at` (seconds precision). It also
requires 64-character lowercase `trial_manifest_sha256`, `split_sha256`, and
`report_sha256` identifying retained calibration artifacts. Canonical JSON uses
sorted keys, separators `(',', ':')`, no NaN and UTF-8, as implemented by
`src.verification.contract.fingerprint`. There is deliberately no installed
example with a fake operational threshold or invented evidence.

Malformed, absent, altered or incompatible policy/evidence yields
`POLICY_NOT_READY`, null verdict. A matching approved policy yields TRUE iff cosine
is greater than or equal to the frozen threshold, otherwise FALSE. Quality failures,
unregistered/reused sources, unenrolled claims and processing errors remain separate
non-decisions regardless of threshold. A policy requires its exact reference count
and recipe-bound centroid semantics. A changed count returns POLICY_NOT_READY until
compatible evidence exists.

## Trust and operational limits

Approval/evidence files are an assertion by a **trusted local operator**. Their hash
checks detect mismatches; they do not independently reproduce a study, authenticate
the approver, verify report contents or provide cryptographic signing. Registry,
configuration, code, weights, templates and policy must be protected from unauthorized
local modification. A user who controls all those files can forge evidence; this is
not a hardened authentication product. Original-source checks cannot detect a
re-encoded derivative that an operator falsely registers as a new original.

Use a single writer/process for this local registry and reference store. File
replacement is atomic and temporary output is private; directory fsync/power-loss
recovery, multi-writer enrollment coordination, encryption, authenticated storage,
retention automation and adversarial input security certification are not provided.
The MediaPipe single-person gate still depends on operator review. No liveness or
forensic deepfake detection is claimed.

GaitPT's CC BY-NC-ND license and checkpoint permissions must be reviewed for the
intended use/distribution. The community MediaPipe binary's provenance and compatibility
must be validated on the actual device. No source-dataset sample reproduction or
real-world biometric reliability is established here. GPGait++/GPGait remains a
research challenger; no scientific winner has been selected.

## Migration and rollback

This is a breaking CLI/data-model change. Live 1:N identification is retired. `enroll.py --name` now records and enrolls
a fresh USB walk; `identify.py --identity` records a different walk and verifies only
that claim. `--subject-id` / `--claimed-id` remain aliases and `verify.py` is equivalent.
Both scripts support optional original `--video`; unregistered originals require
explicit registration/consent flags. Fresh operational records are registered
automatically, and a query never receives the claimed ID as its ground-truth owner. Old templates/checkpoints are not converted into new
embeddings. The default V2 store and events directory are separate; existing outputs
are preserved. Keep historical `database/gait.db`, source footage and model weights.

`legacy/` contains byte-preserved historical implementation/diagnostics and previous
active docs/config. The old scripts contain obsolete assumptions and unrestricted
pickle loading; they are archival, excluded from active imports/tests, and must not
be used for the new runtime. Historical reproduction should use an isolated checkout
of baseline `773207e`, with its original dependency/security limitations disclosed.
No production rollback to identity search is part of this verification-only task.


## Confirmed Nano package profile and startup gate

The owner confirmed JetPack 4.6.x / Ubuntu 18.04 / Python 3.6 / CUDA 10.2 and a USB
webcam. Use only the Nano requirements and NVIDIA/community wheels specified in the
README. Modern `requirements.txt` and restricted `.pth` artifact conversion are
separate workstation tools; they must not upgrade the Nano runtime to modern PyTorch.
Prepared numeric weights are copied once; every operational capture/inference step
executes locally. No unsafe `.pth` fallback is introduced to accommodate old PyTorch.

`check_nano.py --capture-test` reports installed pin/import/architecture/OS/CUDA/tool
conflicts and exercises the same USB encoder/decoder in a private temporary clip.
On aarch64, the runtime also checks the profile before camera use. The scripts
preflight model loading and the actual pose graph before asking for a new walk.
A version check is not ABI/performance certification: the real device must pass the
checks. The experimental MediaPipe binary and installed native libraries remain
physical validation dependencies. Startup failures never become gait mismatches.

Operational captures are not research ground truth or scientific split assignments.
Consent flags record an operator attestation, not an authenticated consent document.
Keep operational diagnostics out of calibration and locked-final study trials.
