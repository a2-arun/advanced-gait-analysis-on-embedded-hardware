# Historical 1:N application archive

These files preserve the pre-migration application, standalone checkpoint diagnostics,
config, startup scripts and active documentation from baseline `773207e`. Original
bytes are retained. The old `models/`, committed checkpoint, `OLD_README.md`, historical
phase reports and user outputs remain at their existing paths.

This archive is not imported by `src/verification`, and active tests live in `tests/`.
The old diagnostic file named `test_model_loading.py` was a standalone runner with
parameters that were not pytest fixtures. It is preserved here, not rewritten into
claims that the old model passed new verification tests. Some archived scripts use
unrestricted pickle loading, identity search, optimistic historical metrics or obsolete
paths. Do not run them as the V2 deployment. To investigate historical execution,
use an isolated checkout of `773207e` and assess its original dependencies/security.

Use the repository-root README and new `scripts/gait.py`, `enroll.py`, `verify.py`
for the current claimed-identity-only local Nano application.
