# Implementation and runbook

This implements the proposed v2 protocol. It contains no measured research results.
Run the resource pilot on each new GPU before launching screening. The tested
environment is Linux x86_64, Python 3.12, CUDA PyTorch 2.8.0 and torchvision 0.23.0.

## Installation and CPU checks

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-cu128.lock
pip install --no-deps -e .
python -m unittest discover -s tests -v
modalitygap matrix
modalitygap matrix --confirmation
```

The model interface is tied to transformers 4.57.1. Its
[Qwen assembly implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)
scatters merger outputs into image positions before calling `model.language_model`.
A decoder pre-hook therefore sees the shared input interface without changing
Qwen's image grids, attention mask or mRoPE construction. Cached decode calls are
recognized by nonzero cache positions and are left unchanged. The full-input
hook-equivalence check is mandatory at run startup.

## Source data contract

Download the original images/posters and authors' annotations separately; these
are not distributed here. Export author splits into UTF-8 JSONL with one row:

```json
{"id":"stable-example-id","split":"train","image":"images/example.jpg","text":"Original cleaned post or chosen plot","labels":["sarcastic"]}
```

`split` is `train`, `dev`, or `test`. Image paths are relative to the source JSONL,
or absolute. MMSD2 labels are exactly `not_sarcastic` and `sarcastic`; use corrected
`data/text_json_final` annotations, preserving authors' cleaned text and splits.
For MM-IMDb, labels must match the 23 canonical genres in `protocol.py`. Use raw
posters and plots, not feature matrices. Choose the first nonempty plot in the
released order when there are multiple plots; record this rule in the source
revision description. Do not prepend titles, years, IDs or filenames to text.

This explicit interchange contract avoids guessing layouts of third-party dataset
copies. Native author-format download/conversion is not automated. Validate your
export against source counts and annotation semantics before preparing manifests.

```bash
modalitygap prepare --dataset mmsd2 --source data/mmsd2-source.jsonl \
  --source-revision AUTHORS_COMMIT_AND_EXPORT_RULE --output data/mmsd2-v1
modalitygap prepare --dataset mmimdb --source data/mmimdb-source.jsonl \
  --source-revision AUTHORS_RELEASE_AND_FIRST_NONEMPTY_PLOT --output data/mmimdb-v1
```

Preparation rejects repeated IDs and cross-split exact RGB image or normalized
text duplicates. Resolve reported overlap explicitly in a revised source export;
do not silently change official split assignments. Within-split duplicates remain
and receive connected-component IDs for grouped bootstrap. Near duplicates require
manual inspection; exact hashes do not detect perceptual/semantic duplicates.
Missing/corrupt images and empty text are counted and excluded. Train pools use
class stratification or iterative multilabel stratification, seed 42, at most 6000
examples. Dev/test remain complete after these checks. Review `ocr_audit.jsonl`
manually before freezing the study. Original sources and image files should remain
immutable; manifests hash decoded images and source JSONL for provenance.

## Freeze model and statistics

The CLI defaults to model commit `66285546d2b821cf421d4f5eb2576359d3770cd3`.
The same immutable SHA is used for tokenizer, processor and weights. Never pass
`main`. Set `MODEL_SHA` below to this SHA. Each dataset needs its own statistics.
Use `--cache-dir artifacts/hf-cache` with stats/run/test when using the fetch script cache.

```bash
modalitygap stats --dataset mmsd2 --data data/mmsd2-v1 --revision "$MODEL_SHA" \
  --samples 256 --output artifacts/stats/mmsd2.pt
```

The estimator samples only training examples with seed 2026 and uses population
moments: equal weight per example, equal weight per content token within example.
It runs the original frozen model before any adaptation. The saved moments and
sample IDs are fixed for all runs, including trained-merger configurations.
The replacement is `mean + 0.05 * std * noise`; noise is keyed by SHA256 of noise
seed, ID, modality and expanded token position. Mean-only and seeds 2027/2028 are
separate sensitivity evaluations at epochs 0, 4 and 8.

## Pilot, then screening

Run pilots for L, M and LM on both datasets. Pilot training uses 64 train examples
and 32 dev examples (or available counts) and all three evaluation conditions.
`--pilot-steps` is a maximum; an eight-epoch prefix can end before that limit.
Pilots write separate `_pilot` output directories and analysis ignores them.

```bash
modalitygap run --dataset mmsd2 --data data/mmsd2-v1 --revision "$MODEL_SHA" \
  --stats artifacts/stats/mmsd2.pt --config LM-low --seed 42 \
  --code-revision YOUR_IMPLEMENTATION_COMMIT --pilot-steps 32 --output artifacts
```

Inspect `resources.json`, intervention indices/norms, target parsing and actual
expanded lengths. All trainable LoRA/merger parameters and AdamW moments use FP32;
the frozen backbone and autocast forward use BF16. This prevents small learning
rates from being rounded away in merger weights, particularly normalization scales. Record the hardware/software environment. Encoder initialization verifies the longest
canonical targets fit the 8/128 decoding budget before screening. No quantization or
per-run resolution fallback is applied. If memory fails, revise global input
limits in `DEFAULTS`, version the protocol, regenerate statistics, and rerun every
pilot before screening. Four-bit quantization is not implemented.

Omit `--pilot-steps` to run eight complete epochs. Start every configuration from
the original backbone. Run all six configurations on both datasets with seed 42;
repeat seeds 43 and 44 for full confirmation. `matrix` prints the fixed schedule.
The scheduler is always built for eight epochs. No performance early stopping is
implemented. LoRA and train data order have independently reset seeds; dropout
random consumption can differ by scope. Epoch 0 is evaluated independently for
every run, with no assumption that LoRA initial states are functionally equivalent.

Checkpoints save trainable tensors, optimizer, scheduler and Python/NumPy/CPU/CUDA
RNG states. Exact optimizer-boundary resume is supported; see the resume section below. Existing run/statistics and
manifest destinations are rejected to avoid accidental mixed artifacts.

## Analysis and held-out evaluation

```bash
modalitygap analyze --dataset mmsd2 --root artifacts --output figures
modalitygap analyze --dataset mmimdb --root artifacts --output figures
```

Analysis requires complete epoch 0–8 trajectories per included run. It derives
contributions within each seed, keeps negative differences, plots seed traces and
means, and writes all ordered candidate pairs, two-of-three confirmation flags,
late-epoch differences, rank agreement and top-k retention. Bootstrap uses 2000
paired group resamples with the same draws across configurations, endpoints and
conditions within each training seed. Intervals describe evaluation uncertainty,
not training randomness or multiple-comparison-adjusted inference. Per-class and
per-genre metrics remain in `metrics.csv`; inspect them alongside sensitivity
predictions before interpreting visual reliance. Automated mechanistic conclusions
are deliberately not generated.

GPU time plots use cumulative training time. Validation time is separately stored
for every main and sensitivity evaluation; hardware validation must establish the
wall-time budget. Epoch-0 reuse is disabled, so actual evaluation counts include it
for every run. Selective confirmation/fallback selection should be prespecified
in a versioned study record; it is not automatically chosen by favorable plots.

Before evaluating test, save a locked selection file:

```json
{"locked_at":"ISO-8601 timestamp", "comparisons":[
  {"run":"artifacts/mmsd2/L-low/seed_42","config":"L-low","epoch":2},
  {"run":"artifacts/mmsd2/L-low/seed_42","config":"L-low","epoch":8}
]}
```

```bash
modalitygap test --data data/mmsd2-v1 --stats artifacts/stats/mmsd2.pt \
  --selection locked-comparisons.json --output heldout
```

Only epochs 2/8 are accepted; test is never read by stats, training or screening
analysis. Keep held-out output separate. Do not use test results to choose new
comparisons or extend the horizon.

## Verification status

CPU checks cover invalid-output F1 handling, strict parsing, negative contribution
values, paired/group bootstrap, immutable manifests and overlap rejection, all
seven figure outputs, and deterministic embedding replacement with protected
positions. A randomly initialized tiny Qwen2.5-VL model exercises L/M/LM trainable
scopes, backward passes, inactive-hook equivalence and three-token cached generation
under both masking conditions. It downloads no pretrained weights and makes no
claim about task performance. GPU validation commands and their measured scope are documented below. Original
author-data integration and research experiments remain separate from synthetic
engineering checks.


## Reproduce GPU engineering validation

Install the CUDA lock above and ensure `nvidia-smi` and
`python -c "import torch; assert torch.cuda.is_available()"` work in your shell.
A device-hiding execution sandbox requires approved execution outside that sandbox.
No driver reinstall is needed in that case. For CPU CI only, use
`requirements-cpu.lock` instead of the CUDA lock.

```bash
python scripts/fetch_model.py
export MODALITYGAP_PROCESSOR=artifacts/hf-cache/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3
export MODALITYGAP_GPU_TESTS=1
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1
python -m unittest discover -s tests -v
python scripts/validate_gpu.py
python scripts/validate_full_workflow.py
```

The unittest GPU suite uses the real processor and small Qwen weights. It checks
all twelve dataset/config combinations for eight epochs, independent original-model
embedding-statistic calculations, all 45 development evaluation variants per run,
checkpoint integrity and locked held-out reload, 2000-resample bootstrap, and figures.
Its random-weight fixture applies a strong EOS decoding bias solely to keep the
orchestration test fast. This is not used with pretrained model validation.

`validate_gpu.py` uses the original pretrained 3B weights, 32 unique synthetic train
examples and two dev examples for each task, all three adaptation scopes, image
sizes including the full 256-token visual budget, and maximum-length genre targets.
It runs two optimizer steps per scope/task: the first warmup step has zero learning
rate, so the second is needed to verify actual weight changes. It requires nonzero
gradient norms, changed tensors, finite losses, and bit-identical generated token IDs
after rebuilding the model and reloading the checkpoint in every condition.

`validate_full_workflow.py` uses the resulting fixed statistics and fixtures to run
the original 3B model with LM-high for eight complete epochs per task. It exercises
27 main plus 18 sensitivity development evaluations, locked epoch-2/8 held-out
checks, and 2000-resample analysis plus seven figures per task. This validates the
full pipeline with the largest adaptation scope/rate; it is not twelve full-size
screening runs and is not an experiment on either authors' dataset.

Every validation destination must be fresh; use `--output` to choose another one.
Validation outputs are ignored by Git. See `docs/validation.md` for the measured
run record. Dataset releases and licensed image files are not bundled.

## Token and output handling

Body token IDs are inserted after the fixed field marker and before the protected
user closing/assistant boundary. This avoids BPE merges consuming marker tokens.
Budgeting reserves the longest legal task target for every example, independent of
its reference labels. Image hashes are rechecked during encoding. Supervised loss
covers only the answer and EOS. Decoding removes only terminal EOS/padding;
unexpected structure tokens are preserved for strict parsing. Unmapped generated vocabulary IDs are recorded as
invalid outputs with the token IDs and decoding error; they cannot crash evaluation
or silently disappear from F1 denominators.

The MM-IMDb sampler specifies both fold sizes explicitly for compatibility with
scikit-learn and records the actual stratified pool size, which may differ slightly
from the requested size. Checkpoint completion manifests hash trainable tensors and
optimizer/RNG state. Precision policy is saved in the configuration and masking
statistics metadata; statistics from another policy are rejected. Reload rejects missing/extra keys, wrong shapes and NaNs.


For an independent CPU replay of saved GPU predictions and metrics:

```bash
python scripts/audit_predictions.py --root artifacts/validation/full \
  --processor "$MODALITYGAP_PROCESSOR" --output artifacts/validation/full/strict-decoding-audit.json
```

This replays generated token IDs and verifies every raw string, parsed label,
invalid flag, output length and aggregate/per-label metric without rerunning inference.


## Resume training (v0.2)

New runs use deterministic PyTorch algorithms, deterministic cuDNN, disabled TF32,
and a deterministic cuBLAS workspace (`:4096:8` by default). Unsupported
nondeterministic operations raise an error rather than silently reducing the
reproducibility guarantee. The eight-epoch learning-rate schedule is constructed
once for the complete horizon, including when a run is deliberately interrupted.

```bash
modalitygap run --dataset mmsd2 --data data/mmsd2-v1 \
  --stats artifacts/stats/mmsd2.pt --config LM-low --seed 42 \
  --output artifacts --code-revision FIXED_SOURCE_REVISION \
  --checkpoint-every-steps 50
```

After interruption, repeat the same command with `--resume-from latest`. It selects
the largest saved global optimizer step in this dataset/config/seed's directory;
incomplete staging directories are ignored and corrupt completed checkpoints are
rejected. An explicit checkpoint directory is also accepted:

```bash
modalitygap run --dataset mmsd2 --data data/mmsd2-v1 \
  --stats artifacts/stats/mmsd2.pt --config LM-low --seed 42 \
  --output artifacts --code-revision FIXED_SOURCE_REVISION \
  --checkpoint-every-steps 50 --resume-from latest
```

Epoch checkpoints are always saved. `--checkpoint-every-steps N` adds snapshots at
optimizer boundaries inside an epoch (default 0 means epoch checkpoints only).
`--stop-after-steps N` deliberately exits after global optimizer step N and writes
a resumable checkpoint even when the periodic interval is zero. It is a compute
interruption mechanism, not performance-based early stopping. Ctrl+C/OOM recovery
uses the last completed checkpoint; work since that snapshot is replayed. A partial
gradient-accumulation group is never committed. The next-example cursor resumes the
same seeded epoch permutation, including the final batch smaller than 16.

Snapshots atomically publish `trainable.pt`, `state.pt`, the processor and
`complete.json`. State includes AdamW moments, scheduler position, Python/NumPy/
Torch CPU/all-visible-CUDA RNG states, the data cursor, partial-epoch loss/count
accounting, previous metrics/contributions/resources, and configuration/runtime
signatures. No frozen-backbone weights are duplicated. Rebuilding the model,
optimizer and scheduler and running the inactive-hook check happens before RNG
restoration; RNG is restored last, immediately before continuing the saved phase.
Checkpoint serialization itself is RNG-neutral. An epoch whose training finished
but whose evaluation did not finish replays that evaluation once from the saved
state; metric rows are not duplicated.

Exact resume requires the same source code, Python/core-library versions, GPU type,
visible CUDA-device count, precision settings, model revision, training manifest,
masking statistics, adaptation scope and seed. These signatures are checked before
continuation. Timing and allocator telemetry can differ after a restart; tensor
values, AdamW state, scheduler state and RNG streams are the equality targets.
To branch from an earlier checkpoint, use a new `--output` root. Prior completed
checkpoints and prediction/intervention prefixes are copied there. An in-place
rewind over newer completed checkpoints is rejected.

Version-0.1 epoch checkpoints lack data-cursor and determinism metadata. They are
rejected by default. `--allow-legacy-resume` explicitly permits continuation from a
complete old epoch if its configuration, runtime and epoch resource history match,
but does not promise bitwise agreement with the old uninterrupted trajectory.
A real old 3B comparison showed differing parameters despite equal RNG states;
that comparison is not counted as a passing exact-resume test. New checkpoints
written after the opt-in use v2 metadata. Do not use the opt-in for v2 provenance
mismatches; it does not bypass their checks.

## Reproduce continuation tests

The full GPU suite includes both task schemas and L/M/LM scopes, 19 training
examples (16 + a last batch of 3), a stop at step 3 in epoch 2, periodic step
snapshots, repeated stop/resume, pending epoch evaluation, output branching,
legacy opt-in, mismatch rejection and corruption/atomic-write checks. Test-only
augmentation and validation deliberately consume all four RNG streams; they are
not added to the experimental training method. Comparisons use `torch.equal` and
exact recursive state comparison, not tolerances.

```bash
# With the GPU test environment variables from the earlier section:
python -m unittest discover -s tests -v
# Original pretrained 3B: separate OS processes for baseline, stop and resume.
# This uses generated engineering fixtures, not original author data.
python scripts/validate_resume.py
```

The full-size continuation check uses LM-high, 17 train examples per task (16 + 1),
and an interruption after optimizer step 3, followed by completion to epoch 8.
Saved epoch-8 parameters, optimizer/scheduler state, all RNG snapshots and the
post-validation RNG states are compared exactly between separate processes.
See [the continuation validation record](resume-validation.md) for measured results.
