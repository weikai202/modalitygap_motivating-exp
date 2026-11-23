# Full Experiment Protocol

## Scope

Train MMSD2.0 and MM-IMDb separately using all complete, retained examples in their official training partitions. Evaluate the complete retained development partitions at epochs 0-8. The code does not subsample training or development for a research run.

The model is `Qwen/Qwen2.5-VL-3B-Instruct`, pinned to revision `66285546d2b821cf421d4f5eb2576359d3770cd3`. Both datasets use the same model, processor, input budgets, optimizer, masking method, and eight-epoch horizon. No ranking reversal or improvement is assumed in advance.

## Data

| Dataset | Input | Target | Primary score |
|---|---|---|---|
| MMSD2.0 | Original image and author-cleaned post | `not_sarcastic` or `sarcastic` | Macro-F1 |
| MM-IMDb | Original poster and first nonempty released plot | Canonically ordered JSON array of the 23 genres | Micro-F1 |

MMSD2.0 annotations must come from corrected `text_json_final`; image mirrors never override author text or labels. MM-IMDb must use the author raw archive and official `split.json`. Project raw genre tags onto the fixed 23-class task vocabulary and audit omitted tags. Do not add titles, IDs, years, filenames, or external metadata to prompts.

Before training:

1. Validate unique IDs and official splits.
2. Fully decode images; record missing/corrupt images and empty text.
3. Find decoded-RGB and whitespace-normalized text duplicates, including transitive duplicate components.
4. Apply the configured policy before looking at model outputs. `prefer-heldout` retains a duplicate component only in its highest-priority original split: test, then development, then training. Rows are never moved or relabeled. `error` rejects any cross-split content duplicate instead.
5. Freeze every retained training, development, and test row, label order, content hashes, exclusions, provenance, and duplicate grouping.

Report both official input counts and post-audit counts. All configurations and seeds share these immutable manifests. Reading test metadata for duplicate auditing does not authorize model evaluation on test.

## Model and training

Use generative assistant-only classification loss. The task instruction, allowed labels, image structure tokens, field delimiter, and assistant boundary are protected. No classification head, additional encoder, auxiliary objective, or modality dropout is trained.

| Configuration | Trainable components | Learning rate |
|---|---|---|
| L-low | Language-model attention LoRA | 2e-5 |
| L-high | Language-model attention LoRA | 1e-4 |
| M-low | Visual merger only | 2e-5 |
| M-high | Visual merger only | 1e-4 |
| LM-low | Language-model attention LoRA and visual merger | 2e-5 |
| LM-high | Language-model attention LoRA and visual merger | 1e-4 |

Each configuration runs seeds 42, 43, and 44, for 18 runs per dataset and 36 total. Each run starts from the original backbone. Freeze the vision encoder, original language weights, token embeddings, and LM head. LoRA targets only language-layer q/k/v/o attention projections, with rank16, alpha32, and dropout0.05.

Use BF16 forward operations and FP32 trainable/master parameters; AdamW, weight decay0.01; microbatch1, accumulation16, gradient clipping1.0; gradient checkpointing; training cache disabled. Construct cosine decay with warmup0.05 for the full eight epochs from the start. Early checkpoints are prefixes of that same schedule.

MMSD2.0 body text is limited to128 tokens; MM-IMDb to384. Use 128-256 merged visual tokens and a total limit of1536, including visual structure and target tokens. Truncate only body text. Canonical output budgets are8 tokens for MMSD2.0 and128 for MM-IMDb.

## Symmetric masking

There is one masking protocol: `symmetric-attention-v2`.

Estimate fixed per-dimension means once from256 deterministically selected training examples using the original frozen model, with equal example weighting and equal token weighting within each example. This statistics sample does not limit the training data. Freeze the resulting means across epochs, configurations, and seeds.

At evaluation:

- `full`: preserve original image and body.
- `mask_image`: replace image-content embeddings with the frozen image mean and block those keys in every decoder layer.
- `mask_text`: replace body-token embeddings with the frozen text mean and block those keys in every decoder layer.

Apply key blocking during prefill and cached decoding, after original mRoPE construction. Preserve sequence length, image grids, positions, instructions, labels, delimiters, and the assistant boundary. Exclude hidden prompt token IDs from the repetition-penalty history while retaining generated history.

These are interventions on the same trained model. Image-contained writing remains available under text masking. Retained structure can affect outputs. Interpret the differences as conditional performance reliance, not causal semantic attribution or embedding-space distance.

## Evaluation and analysis

Use deterministic greedy generation and strict target parsing. Invalid outputs remain invalid and count toward score errors; do not silently map unsupported label synonyms. Save generated token IDs, raw text, parsed targets, invalid rates, output length, and truncation.

Define `C_image = score(full) - score(mask_image)` and `C_text = score(full) - score(mask_text)`. Retain negative values. If plotting `gap = C_text - C_image`, label that definition explicitly. Do not compare MM-IMDb micro-F1 and other macro-F1 values as identical task metrics.

Save all three scores at epoch0 and every epoch1-8: 27 evaluations per run and972 across the full matrix. Analyze early/final rankings, fixed epoch2/8 comparisons, all three seeds, and paired duplicate-group bootstrap intervals. Report negative and inconclusive results.

Test evaluation is never automatic. Lock chosen configurations and seeds after development analysis, and evaluate only their epoch2 and8 checkpoints under all three conditions. Do not sweep test epochs or revise the training horizon to obtain a favorable crossing.
