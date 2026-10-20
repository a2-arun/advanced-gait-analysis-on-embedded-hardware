# Gait verification on original Jetson Nano

Before gait, pose, enrollment, recording, inference or evaluation changes, read in full:
- `docs/gait_verification/GAIT_VERIFICATION_REARCHITECTURE_HANDOFF.md`
- `docs/gait_verification/GAIT_VERIFICATION_TECHNICAL_PRD.md`
- `docs/ARCHITECTURE.md` for the Nano-specific override and implementation boundaries.

Only 1:1 claimed-identity verification is authorized. Fetch the claim's references only.
Usable enrolled inputs with an approved, recipe-bound policy return TRUE/FALSE;
failures/quality/unenrolled/unready policy return explicit statuses and null verdicts.
Never identify/rank others or infer deepfakes from mismatch. No remote inference.
Preserve original RGB, historical models/checkpoint/reports and `legacy/`. No training
or threshold fitting in the engineering phase. Frozen GaitPT first; GPGait family is
a research challenger, not an assumed winner. Python 3.6 compatibility is required
for the Nano runtime. Never deserialize pickle checkpoints on the Nano.

State scope/phase/non-goals/conflicts/tests before edits. Afterward report changed
files, commands/results, licensing blockers and unvalidated hardware/scientific gates.
New owner requirements override documents; record amendments in both documents.
