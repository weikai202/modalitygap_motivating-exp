# Full Experiment Operations

## Requirements

Use Linux, Python3.12, a BF16-capable NVIDIA GPU, and the pinned CUDA12.8 environment from the README. Full development generation and 36 training runs are expensive. Reserve storage for original data, the model cache, nine epoch snapshots per run, intermediate checkpoints, predictions, and logs. Run the campaign under `tmux` or a batch scheduler.

All instructions below are for the full experiment. Small generated examples exist only inside automated tests.

## Data acquisition

```bash
.venv/bin/python scripts/download_full_data.py --root data/full-inputs
.venv/bin/python scripts/prepare_full_experiment.py \
  --inputs data/full-inputs/inputs.json \
  --output data/full-export-v1 --workspace artifacts/full-study-v1
```

The downloader obtains corrected MMSD2.0 annotations at author revision `1627811f7a1e0acedce5b9df8b58919d86dd638e` and image bytes from `coderchen01/MMSD2.0` at revision `5b39e5b844accf2331d1738a3d91d4cb59005a50`. The mirror is an image source; the author annotations always define the labels and text. Complete mirror split ID coverage is checked.

MM-IMDb uses the original raw archive with published MD5 `38e154264b3771f70840c6cb5f170a93`. Existing mismatching archives are rejected. The archive downloader supports cached ranges and interrupted transfers. Export never uses precomputed feature inputs.

Official release counts before auditing are:

| Dataset | Train | Development | Test |
|---|---:|---:|---:|
| MMSD2.0 | 19816 | 2410 | 2409 |
| MM-IMDb | 15552 | 2608 | 7799 |

These are release counts, not guaranteed post-audit counts. The workspace manifest records actual retained counts, image/text hashes, missing/corrupt exclusions, and every cross-split duplicate exclusion. The separate OCR audit checklist is for manual image inspection and does not select or limit training examples. Training uses every retained training row; development and test use every retained row of their respective splits.

Exports are immutable and refuse existing destinations. Move the export folder with its `images/` directory. Generated `experiment.json` contains machine-local paths: regenerate or update those paths after moving to another machine, before initializing a workspace.

For manually obtained original data:

```bash
.venv/bin/python -m modalitygap.full_data mmsd2 \
  --annotations /path/to/MMSD2.0/data/text_json_final \
  --images /path/to/original/images --output data/mmsd2-export-v1
.venv/bin/python -m modalitygap.full_data mmimdb \
  --archive /path/to/mmimdb.tar.gz --output data/mmimdb-export-v1
```

Copy `configs/full-experiment.example.json`, set the two source paths, and copy source revisions from the exported provenance files. The full-study loader rejects a reduced training pool, missing configurations, incomplete seed lists, unsupported datasets, or another masking method. A dataset may be placed in its own workspace to run the two complete dataset matrices on separate GPUs.

## Start and resume

```bash
bash scripts/run_experiment.sh --config data/full-export-v1/experiment.json --dry-run
bash scripts/run_experiment.sh --config data/full-export-v1/experiment.json
```

Stages are `all`, `prepare`, `stats`, `train`, and `analyze`. Every training job always has the eight-epoch horizon and uses the complete retained training/development splits.

```bash
# Optional staged execution, with the same full-data configuration.
bash scripts/run_experiment.sh --config data/full-export-v1/experiment.json --stage prepare
bash scripts/run_experiment.sh --config data/full-export-v1/experiment.json --stage train --resume
bash scripts/run_experiment.sh --config data/full-export-v1/experiment.json --stage analyze --resume
```

For SSH sessions:

```bash
tmux new -s modalitygap-full
bash scripts/run_experiment.sh --config data/full-export-v1/experiment.json --resume
# Detach with Ctrl-b d. Reconnect later:
tmux attach -t modalitygap-full
```

Omit `--resume` for a new workspace; include it when the workspace has already been initialized. If the process has ended, rerun the same command with `--resume`. A live workspace lock prevents concurrent launchers. Do not remove the lock to start a second trainer.

Checkpoint restoration includes parameters, AdamW, the full scheduler, sample cursor, partially completed epoch accounting, and all Python/NumPy/Torch/CUDA random states. Save only at optimizer boundaries. Evaluation interrupted at an epoch checkpoint is replayed without duplicating metric rows. The newest two complete step snapshots are retained by default; all epoch0-8 snapshots remain available.

Keep the same source checkout, model revision, dataset files, frozen means, GPU type, and compatible runtime for exact continuation. Changed inputs or runtime are rejected. Use a new workspace for a new experiment. Full-dataset training is never automatically reduced after an error.

Read progress without running any training:

```bash
.venv/bin/python scripts/full_experiment_status.py --config data/full-export-v1/experiment.json
```

## Explicit test lock

After development analysis, explicitly choose the comparisons to test:

```bash
.venv/bin/python scripts/lock_heldout_selection.py \
  --config data/full-export-v1/experiment.json \
  --configs L-low L-high LM-low --seeds 42 43 44 \
  --output selections/full-study-v1
```

The command verifies completed runs and checkpoint hashes, writes a per-dataset locked list containing only epoch2/8, and prints the exact test commands. It does not execute test. Review the selections and execute the printed commands.

Selections and test destinations refuse overwrite. If test evaluation is interrupted, preserve its partial output and rerun the printed command with a fresh `--output` version. No training needs to be repeated.

## Output interpretation

- `pipeline.json`: frozen input, source, and scientific-setting identity.
- `manifests/<dataset>/`: complete retained splits, provenance, exclusions, and duplicate grouping.
- `stats/<dataset>.pt`: original-model training-only moments.
- `runs/<dataset>/<configuration>/seed_<seed>/metrics.csv`: all27 development evaluations.
- `contributions.csv`: the two ablation score differences at epochs0-8.
- `predictions/`: per-example generated IDs, raw output, parsed labels, and errors.
- `interventions/`: actual hidden-key and decoder-layer trace checks, including the initial integrity-check forward pass; these trace events are not additional scored dataset evaluations.
- `resources.json`: per-epoch loss, throughput, sequence lengths, and GPU memory.
- `checkpoints/`: trainable weights, optimizer, scheduler, RNG, and cursor state.
- `figures/<dataset>/`: development ranking, contribution, and bootstrap analysis.
- `logs/`: appended stage logs.

No performance numbers or plots are shipped as research results. They are generated only by running the experiment on actual data.

## Implementation tests

```bash
OMP_NUM_THREADS=1 MODALITYGAP_GPU_TESTS=1 \
MODALITYGAP_PROCESSOR=/absolute/path/to/pinned/processor/snapshot \
.venv/bin/python -m unittest discover -s tests -v
```

GPU integration tests use tiny randomly initialized Qwen models and temporary examples to check all36 orchestration paths, eight-epoch schedules, strict parsing, retained step snapshots, exact restoration, and locked test reload. Processor tests use the actual pinned processor. These engineering checks do not measure research performance.

To re-score a completed experiment from its saved token IDs:

```bash
.venv/bin/python scripts/audit_predictions.py \
  --root artifacts/full-study-v1/runs \
  --processor /absolute/path/to/pinned/processor/snapshot \
  --output artifacts/full-study-v1/prediction-audit.json
```
