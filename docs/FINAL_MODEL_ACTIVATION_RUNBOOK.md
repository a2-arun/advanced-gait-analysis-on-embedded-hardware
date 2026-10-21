# Final-model activation runbook (future)

Use this **after** the research repository has frozen its final model. No final model, approved
policy or Nano parity evidence exists today (21 October 2026). This runbook separates what can
be run now (**EXISTS**) from what must still be written or produced (**FUTURE — does not
exist**). It does not promise that any particular checkpoint will run on the original Nano.

## The package and the states

A deployment package is a directory `<package-id>/` (`[A-Za-z0-9_-]{1,64}`) containing exactly:

| file | produced by | purpose |
|---|---|---|
| `contract.json` | research side | `nano-deployment-contract-v1` (`src/verification/deployment.py`) |
| `manifest.json` | research side | must carry `weights_sha256` equal to the contract's |
| `weights.npz` | research side | float32, finite, pickle-free tensors (`npz-float32-v1`) |
| `approval.json` | research owner | `nano-model-approval-v1`: contract `semantic_id`, `weights_sha256`, `approved_by`, `approved_at` (with UTC offset), `research_reference` |
| `parity_report.json` | the Nano (`lab.py model parity`) | `nano-model-parity-v1` evidence |

`python3 scripts/lab.py model status|inspect` reports six **separate** stages and never one
"ready" flag: `discovered`, `structurally_valid`, `runtime_compatible`, `parity_tested`,
`approved`, `active` (PASS/FAIL/WARNING/NOT_TESTED/BLOCKED each). Activation needs the first
five to be PASS **on the Nano**; a development-only package can never qualify. The runtime
refuses with `MODEL_NOT_DEPLOYED` (no/invalid active model) or `MODEL_INCOMPATIBLE` (runtime
contract differs), and never falls back to another package or the development baseline.

## Honest compatibility limit

Today the Nano build implements one family: the official GaitPT-GREW checkpoint with the
MediaPipe 0.8.5 lite pose recipe. `runtime_compatible` compares a package's contract section by
section with that implementation (`model_slot.expected_contract`), so **only that exact
recipe can pass**. A different checkpoint (even a retrained GaitPT), a different family
(GPGait/GPGait++/E0), RTMPose, other preprocessing or window aggregation will be reported
`FAIL` at `runtime_compatible` — by design — until the code in steps 8a–8c exists.

## Steps

| # | Step | Status (EXISTS = runnable today; FUTURE = does not exist) |
|---|---|---|
| 1 | Freeze the chosen research model (branch + commit, checkpoint hash) | Research side; record it in `contract.json.provenance` |
| 2 | Freeze its pose estimator and preprocessing (joint order, coordinates, normalization, temporal/quality rules, window and walk aggregation) | Research side; each is a contract row and unknowns block activation |
| 3 | Export verified artifacts: tensor-only `weights.npz` + `manifest.json` | EXISTS for GaitPT only: `python scripts/prepare_gaitpt.py --checkpoint <official.pth> --output <dir>` (workstation, torch ≥ 1.13, pinned checkpoint SHA). **FUTURE** for any other model |
| 4 | Generate the deployment contract | EXISTS as a schema/validator (`validate_contract`, `gaitpt_grew_contract` is the only builder). **FUTURE**: a research-side exporter/builder for the final model; hand-written JSON must pass `lab.py model inspect` |
| 5 | Generate reference parity fixtures | EXISTS for GaitPT: `scripts/make_parity_fixture.py` (workstation, read-only on research). **FUTURE**: cases for the final model |
| 6 | Transfer the package to the Nano (USB/SSH) and install | EXISTS: `python3 scripts/lab.py model install <dir>` copies a structurally valid package into `.cache/model-slot/<id>/`, never overwrites, never activates |
| 7 | Check runtime compatibility | EXISTS: `python3 scripts/lab.py model inspect <id>`; `python3 scripts/lab.py preflight`; `python3 scripts/gait.py check --require-nano` (pose graph + model smoke, GaitPT recipe) |
| 8 | Install any optimized backend the model needs | **FUTURE** — nothing exists. 8a: a `GaitBackend` subclass registered in `BACKENDS`; 8b: its contract builder and an `expected_contract` branch in `model_slot.py`; 8c: a pose backend class in `pose_diag.POSE_BACKENDS` plus the estimator in the contract (RTMPose on the Nano is not demonstrated). TensorRT/ONNX work, if any, is a separate measured task |
| 9 | Numerical parity on the Nano | EXISTS: `python3 scripts/lab.py model parity <id> --fixture <fixture.json>` writes `parity_report.json` (levels model_input, walk_embedding, reference_prototype, similarity; tolerances provisional in `parity.py`). Workstation evidence is recorded but only yields `WARNING`; activation needs Nano evidence |
| 10 | Paired real-video preprocessing tests (research pipeline vs Nano pipeline on the same non-sensitive original) | **FUTURE** — no tool; today only `pose extract` offers the Nano-side numbers. PASS only with measured differences, per `docs/RESEARCH_MODEL_DEPLOYMENT_CHECKLIST.md` item 13 |
| 11 | Activate the approved model | EXISTS: `python3 scripts/lab.py model activate <id>` (refused with the failing stages listed unless all five are PASS). Undo with `model deactivate` |
| 12 | Reprocess preserved originals or capture new enrollments | Capture EXISTS: `python3 scripts/enroll.py --name <ID> --consent --single-walker-reviewed` (uses exactly the active package; the recipe changes, so old rows are inert). **FUTURE**: a command that reprocesses preserved hash-verified originals. Embeddings are never converted between models |
| 13 | Validate enrollment and reference compatibility | EXISTS: `python3 scripts/lab.py db check` (counts per recipe, inert legacy rows); verification returns `CLAIM_NOT_ENROLLED` for stale-recipe claims |
| 14 | Install an approved decision policy, only when scientific evidence exists | Format EXISTS (`nano-policy-v1` + `nano-calibration-evidence-v1`, `policy.py`, `config.yaml: policy:`). **FUTURE**: calibration tooling/evidence — none exists and no threshold is bundled. Until then every usable claim returns `POLICY_NOT_READY`, verdict null |
| 15 | Full end-to-end claimed-identity verification | EXISTS (needs 8, 11–14): `python3 scripts/identify.py --identity <ID> --consent --single-walker-reviewed` |
| 16 | Show the result on the HDMI LCD | EXISTS: `python3 scripts/lab.py display result --render window` (newest event) or `--event <file>`; TRUE/FALSE appear only for status `OK` with a real boolean |

## Gate summary before any real use

1. `python3 scripts/lab.py model status` → `state: ACTIVE`, the five qualification stages PASS.
2. `python3 scripts/lab.py db check` → READY, references under the active recipe only.
3. `config.yaml: policy:` set only to an approved, recipe- and count-bound policy.
4. Update `CONTEXT.md` and the two mandatory gait-verification documents with the decision.

Never reuse the old database, bundle or policy across a model change; archive them. Never
silently activate the development baseline: `--development-baseline` exists for engineering
checks of `bundle_dir` only, prints a warning, and is not approvable.
