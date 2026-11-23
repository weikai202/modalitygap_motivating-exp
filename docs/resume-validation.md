# Exact continuation validation (v0.2)

Verified on 2026-11-13T20:31:09.427600-05:00 (America/New_York), using the same RTX 3090,
Python 3.12 / PyTorch 2.8.0+cu128 / transformers 4.57.1 environment as the
[v0.1 engineering baseline](validation.md). No author-data research results are claimed.

## Results

**22 tests passed, zero failures and zero skips.** The complete pre-existing
training/processor/masking/analysis suite passes alongside the new continuation tests.

The production training loop is tested on CUDA for both task schemas and L/M/LM
scopes, with 19 examples (16 + 3), LoRA dropout, an epoch-2 mid-epoch stop at global
step 3, step checkpoints and a second stop/restart. Final parameters, AdamW moments/
parameter groups, scheduler dictionaries and Python/NumPy/Torch CPU/all-visible-CUDA
RNG states match the uninterrupted run exactly. Per-epoch mean losses and gradient
norms also match. Wall-clock and memory allocator telemetry are not equality targets.

Test-only input jitter and validation consume all four RNG streams, so forgetting
a restore or replaying/skipping the wrong evaluation changes the result. They are
not part of the experimental training method. Epoch-pending evaluation, branching
into a fresh output root, prefix artifact preservation, provenance rejection,
explicit legacy import and multiple restarts are exercised.

Atomic snapshot tests also cover serialization that consumes randomness then fails,
missing/corrupt state, malformed completion markers, unsupported formats, attempts
to save uncommitted gradients, and hidden staging directories that already contain
a completion marker but have not been published. No failed/incomplete snapshot can
silently replace the last committed state.

## Original pretrained 3B, separate OS processes

Each task uses 17 synthetic training examples, one dev and one test fixture,
original Qwen2.5-VL-3B weights, real processor, LM-high, original input/generation
budgets, BF16 autocast, FP32 trainable weights/AdamW state and the full eight-epoch
schedule. No EOS shortcut is applied. Three independent CUDA processes perform:

1. An uninterrupted eight-epoch baseline (16 optimizer steps).
2. A run stopped at step 3: epoch 2, next-example cursor 16.
3. Reconstruction and continuation from that checkpoint through epoch 8.

| Task schema | Final epoch / step | Parameters | AdamW | Scheduler | Python / NumPy / Torch CPU / CUDA RNG |
|---|---|---|---|---|---|
| MMSD2 | 8 / 16 | Bitwise equal | Bitwise equal | Equal | All equal |
| MM-IMDb | 8 / 16 | Bitwise equal | Bitwise equal | Equal | All equal |

Comparisons check both the saved epoch-8 RNG state and the global state after final
validation. Parameter/moment tensors use `torch.equal`, with no floating-point
tolerance. The baseline and recovered job record identical training signatures.
The source snapshots recorded by the full-size workers and the release source
hashes are retained in [the machine-readable record](resume-validation.json).

## Determinism and legacy boundary

Exact continuation uses deterministic PyTorch/cuDNN algorithms, disabled TF32 and
a deterministic cuBLAS workspace. The source, model/data/statistics signatures,
Python/core-library versions, GPU type, CUDA device count and backend settings
must match. Unsupported nondeterministic kernels fail rather than being silently
accepted. This is a tested guarantee in this environment, not a cross-hardware
reproducibility claim.

A historical v0.1 3B continuation had equal RNG states but differing parameters;
it is explicitly not counted as a passing result. Those checkpoints lack the new
cursor/determinism metadata and are rejected by default. `--allow-legacy-resume`
permits compatible complete-epoch continuation with no promise of bitwise agreement
with that old trajectory. The opt-in path is tested with a deterministic donor.

## Reproduce and use

See [the resume runbook](implementation.md#resume-training-v02). Add
`--checkpoint-every-steps 50` to the original run command for mid-epoch snapshots;
repeat the original command with `--resume-from latest` after interruption.
`--stop-after-steps N` is an absolute global optimizer-step stop and does not reset
the full-horizon scheduler. Earlier checkpoints require a fresh output root if
newer completed snapshots already exist.

```bash
# In the GPU/processor environment described in implementation.md:
python -m unittest discover -s tests -v
python scripts/validate_resume.py --output artifacts/my-resume-check
```

Local evidence: `artifacts/resume-release-tests.log`,
`artifacts/resume-deterministic-pretrained/report.json` and its per-worker logs.
The final discovery checks harden uncommitted-directory handling; the original 3B
training math and restoration sequence are unchanged. Model weights/checkpoints
are Git-ignored. The source package contains code, tests, locks and compact records.
