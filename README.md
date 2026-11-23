# Full MMSD2.0 and MM-IMDb Experiment

This repository runs one experiment: **all retained official training examples**, the complete retained development split, six adaptation configurations, three seeds, and eight epochs on each of MMSD2.0 and MM-IMDb.

The full matrix contains **36 training runs**. Each run evaluates epochs 0-8 under `full`, `mask_image`, and `mask_text`: **972 development condition evaluations** in total. Test evaluation is a separate, explicitly locked operation.

**No full-experiment research results are included.** Automated tests use temporary engineering fixtures to verify the implementation; they are not dataset performance measurements.

## Setup

Use Linux, Python 3.12, and an NVIDIA GPU that supports BF16. The pinned environment uses PyTorch 2.8.0 with CUDA 12.8.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-cu128.lock -r requirements-data.lock
.venv/bin/python -m pip install --no-deps -e .
nvidia-smi
```

Run the following commands from the repository root.

## Download and export the complete datasets

```bash
.venv/bin/python scripts/download_full_data.py --root data/full-inputs
.venv/bin/python scripts/prepare_full_experiment.py \
  --inputs data/full-inputs/inputs.json \
  --output data/full-export-v1 \
  --workspace artifacts/full-study-v1
```

This creates one configuration: `data/full-export-v1/experiment.json`. It uses `pool_size: "all"`, all six configurations, and seeds 42, 43, and 44. Dataset preparation never selects a smaller training pool.

MMSD2.0 uses the authors' corrected `text_json_final` text and labels. MM-IMDb uses original posters, the first nonempty released plot, the official split, and the fixed 23-genre vocabulary. Exported image paths are relative to `source.jsonl`.

## Run

```bash
# Inspect the full plan without downloading, training, or writing results.
bash scripts/run_experiment.sh --config data/full-export-v1/experiment.json --dry-run

# Start all 36 full-data runs, then analyze the completed matrix.
bash scripts/run_experiment.sh --config data/full-export-v1/experiment.json

# Resume the same campaign after interruption.
bash scripts/run_experiment.sh --config data/full-export-v1/experiment.json --resume
```

Use `tmux` or a batch scheduler for long jobs. The launcher runs sequentially on the selected GPU. Set `CUDA_VISIBLE_DEVICES` before launch when a server has multiple GPUs.

The launcher audits data, downloads/reuses the pinned model, freezes training-only embedding statistics, trains the full matrix, and analyzes development results. It does not run test automatically.

## Read the protocol and operating instructions

- [Experimental protocol](docs/experiment-plan.md): inputs, adaptation, masking, metrics, and analysis.
- [Full experiment operations](docs/full-experiment.md): data sources, duplicate policy, resume, test locking, outputs, and tests.

The only example configuration is [full-experiment.example.json](configs/full-experiment.example.json). Use it for manually exported data, replacing source paths and provenance with those of the actual exports.

## Output

All generated files belong to the configured workspace:

```text
artifacts/full-study-v1/
  pipeline.json
  manifests/<dataset>/
  stats/<dataset>.pt
  runs/<dataset>/<configuration>/seed_<seed>/
    config.json
    metrics.csv
    contributions.csv
    predictions/
    interventions/
    resources.json
    checkpoints/epoch_0 ... epoch_8/
    checkpoints/step_<step>/
  figures/<dataset>/
  logs/
```

Checkpoint restoration includes trainable parameters, AdamW, the eight-epoch scheduler, the sample cursor, and Python/NumPy/Torch/CUDA random states. All epoch checkpoints are retained; by default only the newest two intermediate step checkpoints are kept. Completed runs are verified before being skipped. Use a new workspace for a new experiment.
