# GPU engineering validation (v0.1 baseline)

Verified on 2026-11-13T19:12:18.312648-05:00 (America/New_York).
These are software checks with generated image/text fixtures. No MMSD2.0 or MM-IMDb
research results, ranking reversals, or dataset-level memory guarantees are claimed.

Environment: RTX 3090 (24 GiB), NVIDIA driver 580.173.02, Linux x86_64,
Python 3.12.3, PyTorch 2.8.0+cu128, CUDA 12.8,
transformers 4.57.1, torchvision 0.23.0+cu128 and PEFT 0.17.1.
Model/processor/tokenizer commit: `66285546d2b821cf421d4f5eb2576359d3770cd3`.
Exact runtime dependencies are in `requirements-cu128.lock`.

## Release test suite

**17 tests passed, zero failures and zero skips**, including real-processor tests
and CUDA orchestration. The twelve small-Qwen dataset/config combinations each
completed eight epochs, 27 main and 18 sensitivity evaluations. The suite checks
independent embedding moments with unequal token counts, FP32 optimizer moments,
full-horizon scheduler state, checkpoint reload, paired bootstrap, ranking-reversal
confirmation/top-k behavior, strict decoding, corrupt JPEG exclusion, reproducible
sampling and all seven figures per dataset. Only the random-weight orchestration
fixture uses a strong EOS bias. Pretrained model checks use normal greedy decoding.

## Original pretrained 3B resource checks

Each task has 32 synthetic train examples, two dev examples and two reserved test
examples. Input limits remain 128/384 body tokens, 128–256 visual tokens and a
1536-token total limit. Fixtures include the maximum image size and the full
23-genre target. Each scope performs two optimizer steps at micro-batch 1 and
accumulation 16, using the original eight-epoch cosine/warmup schedule. The first
warmup step has zero learning rate; changed tensors are checked after the second.

BF16 autocast is used for forward computation. All trainable LoRA/merger parameters
and AdamW moments are FP32, so small updates to normalization scales survive.
Frozen vision/LLM/head weights stay frozen and BF16. All five merger parameter
tensors change, and rebuilding from the original backbone plus the saved trainable
tensors reproduces generated token IDs exactly in all three conditions.

| Task schema | Config | Max training tokens | Peak allocated GiB | Peak reserved GiB | Changed tensors |
|---|---|---:|---:|---:|---:|
| mmsd2 | L-low | 428 | 8.17 | 9.25 | 288 |
| mmsd2 | M-low | 428 | 8.61 | 9.29 | 5 |
| mmsd2 | LM-low | 428 | 8.72 | 9.41 | 293 |
| mmimdb | L-low | 838 | 9.04 | 11.15 | 288 |
| mmimdb | M-low | 838 | 9.49 | 11.18 | 5 |
| mmimdb | LM-low | 838 | 9.60 | 11.28 | 293 |

Memory values are PyTorch training peaks per process on these fixtures. CUDA context
and other-process allocations are not included. They are not substitutes for a pilot
with the actual author data on the destination GPU. Timing artifacts are retained but
are not a standalone throughput benchmark; some engineering checks ran concurrently.

## Original pretrained 3B full workflow

The largest adaptation scope/rate, LM-high, completes the entire fixed horizon for
each task schema. This checks the original full-size architecture and unmodified
8/128 generation budgets, not just a smaller surrogate model.

| Task schema | Config | Epochs | Optimizer steps | Dev main + sensitivity | Held-out condition evaluations | Figures |
|---|---|---:|---:|---:|---:|---:|
| mmsd2 | LM-high | 8 | 16 | 27 + 18 | 6 | 7 |
| mmimdb | LM-high | 8 | 16 | 27 + 18 | 6 | 7 |

Every run stores epochs 0–8, optimizer/scheduler/RNG state, per-example predictions,
intervention indices/norms and aggregate metrics. Held-out checks use a selection
file locked before reading synthetic test fixtures and only epochs 2/8. Analysis
recomputes 2000 paired bootstrap resamples and writes seven plots per task.
An independent release-decoder replay verifies every saved generated token sequence
reproduces its raw text, parsed labels, invalid flag and all aggregate/per-label
metrics. Only terminal EOS/padding is removed; unexpected structure tokens remain
visible and are invalid under strict parsing.

## Reproduction and remaining scope

See [the runbook](implementation.md#reproduce-gpu-engineering-validation) for exact
commands. Successful local records are in:

- `artifacts/gpu-tests-verified.log`
- `artifacts/pretrained-fp32-master-validation/report.json`
- `artifacts/pretrained-fp32-full-workflow/report.json`
- Each validation root's `strict-decoding-audit.json`

[The v0.1 machine-readable record](validation-report-v0.1.json) includes numerical resource
measurements and release source hashes. Full validation tensors, model weights and
synthetic outputs are Git-ignored; the source distribution contains code, tests,
protocol, dependency locks and this compact record.

Original author annotation/image releases have not been downloaded or converted.
Their split counts, duplicate/OCR audits and 6000-example pool must be checked using
the documented source JSONL contract before research screening. Author-format
conversion remains an explicit data-export step. Twelve full-size research runs,
confirmation seeds and actual held-out task results were not performed. Training resume was not exposed in v0.1; v0.2 adds it with separate
[exact-continuation validation](resume-validation.md). Other GPUs and
fresh CPU CI installations are not claimed to have been executed locally.
