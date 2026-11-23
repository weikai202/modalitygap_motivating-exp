# One-command launcher validation (v0.3)

Verified on 2026-11-13T21:13:48.974990-05:00 (America/New_York).

The training core is byte-for-byte unchanged from the verified v0.2 continuation
implementation. The added components are the shell bootstrap, Python campaign
orchestrator, JSON example configuration and Chinese operating README.

## Automated checks

- Eight final launcher tests pass: path resolution (including spaces and a different
  working directory), read-only preview, configuration rejection, frozen identity,
  adding confirmation seeds, locking across the launcher and a surviving child,
  interrupted preparation preservation, immutable manifest protection, completed-run
  skipping, native resume arguments, pre-checkpoint initialization recovery, complete
  offline model-cache detection and CPU-only analysis without original source JSONL.
- A full 29-test regression snapshot passed on the 3090 with no failures/skips.
  The initialization-recovery test was then added and the final eight-case launcher
  suite rerun successfully. This is not a claim that a 30-test full run was executed.
- Shell syntax, Python compilation, packaging contents and dependency checks pass.

## Actual pretrained 3B smoke run

Using generated MMSD2-format fixtures (17 train / 1 dev / 1 test), the same entry
point as the README completed these steps on an RTX 3090:

1. Prepared and froze manifests from an explicit external source JSONL.
2. Reused the pinned cached model and estimated fixed statistics from two train IDs.
3. Completed L-low, M-low and LM-low pilots, four optimizer steps each.
4. Started an LM-low screening run and deliberately stopped at global step 3.
5. Repeated the command with `--resume`, reused manifests/statistics/model and
   skipped pilots, then continued the checkpoint to epoch 8 / step 16.
6. Ran 2000-resample analysis and wrote seven figures and 45 development evaluations.
7. Repeated `--resume` again: all training was skipped. All 20 checkpoint
   tensor files retained both their SHA256 and modification timestamps.
8. Executed `--setup --stage analyze --resume`, verifying the real bootstrap and
   editable 0.3.0 installation without retraining.

The trial explicitly selected one dataset schema and one screening candidate.
It is a software smoke check, not a full twelve-run study or author-data result.
The full six-candidate command plan and underlying training matrix are separately
covered by the launcher tests and earlier core validation.

Local evidence is under `artifacts/launcher-smoke/` and the
`artifacts/launcher-smoke-*.log`, `launcher-unit-final.log`,
`launcher-all-tests.log` and `launcher-setup-check.log` files.
[The compact machine-readable record](launcher-validation.json) includes source
hashes. Model caches, fixtures and trained outputs are Git-ignored.

See [README](../README.md) for setup, configuration, stage selection, SSH reconnect,
exact resume, output paths and held-out evaluation. Original images/annotations must
be supplied by the user; the normal launcher never substitutes synthetic data.
