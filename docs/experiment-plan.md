# Screening Protocol: Delayed Visual Utilization and Ranking Reversals

- **Version:** 2.1, symmetric-production amendment
- **Date:** November 23, 2026
- **Model:** `Qwen/Qwen2.5-VL-3B-Instruct`
- **Datasets:** MMSD2.0 and MM-IMDb
- **Hardware target:** one RTX 3090 with 24 GB VRAM
- **Status:** implementation and small real-data pilots validated; the complete screening/confirmation study remains to be run. Production masking is the symmetric protocol below.

## 1. Research questions

1. Does a configuration that underperforms after epoch 2 outperform another configuration after the fixed full training budget of 8 epochs?
2. Do such reversals coincide with an increase in dependence on the correct image input?
3. Does this pattern differ between a task requiring image–text interpretation, MMSD2.0, and a potentially text-dominated task, MM-IMDb?

We do **not** assume that either dataset will exhibit reversals. Negative results must be reported. Masking measures intervention-dependent input reliance; it does not identify causal modality contributions or prove that fusion improves. With a frozen vision encoder, we study delayed utilization of visual features, not slower learning inside the vision encoder.

## 2. What should be fine-tuned?

Use **LLM LoRA as the primary adaptation method**, with **visual-merger-only** and **LLM LoRA plus visual merger** as controls.

Qwen's visual merger aligns visual representations with the LLM input space. It is not a conventional classification fusion head that separately consumes image and text features. Joint image–text processing occurs inside the LLM. Training a new fusion MLP on separately pooled features would define a different model and is outside this first screening experiment.

Use generative classification SFT without adding a randomly initialized classification head. This avoids conflating early modality dynamics with learning a new output head. Keep the original LM head frozen.

## 3. Data preparation

Train and analyze the two datasets separately. Do not pool their examples or average their task metrics.

| Dataset | Inputs | Target | Primary metric |
|---|---|---|---|
| MMSD2.0 | Original image and cleaned post text | Sarcastic / not sarcastic | Macro-F1 |
| MM-IMDb | Original poster and plot synopsis | Multiple labels from 23 genres | Micro-F1 |

### MMSD2.0

- Use the authors' corrected `data/text_json_final` annotations and corresponding original images, not the original MMSD labels.
- Preserve the released train/dev/test split. Verify actual file counts rather than assuming sizes from third-party copies.
- Select a fixed, class-stratified pool of 6,000 training examples. If fewer complete examples are available, use all available examples and document the count.
- Use the complete development split for checkpoint evaluation.
- Preserve the official text cleaning. Do not remove text-dominated examples based on model performance.

### MM-IMDb

- Use the authors' train/dev/test split, raw posters, plot text, and 23 genre labels, not precomputed image or text features.
- Select a fixed pool of approximately 6,000 training movies using multilabel stratification. Use the complete development split.
- Fix the plot-selection rule if multiple synopses exist.
- Do not include external movie titles, IMDb IDs, years, filenames, or other metadata in prompts.
- Names or titles remaining inside posters and plots can still trigger pretrained knowledge; this protocol does not eliminate pretraining contamination.

### Shared checks

- Check split overlap by example/movie ID and inspect duplicate images and text across splits. Document exclusions or grouping decisions before training.
- Exclude incomplete image–text examples consistently and report counts.
- Save immutable manifests, label order, source revisions, and split counts.
- All configurations share the same training pool and development examples.
- Audit a fixed random sample for text embedded in images. A text-masked VLM may recover some textual information through OCR.
- The 6,000-example pool defines the full training data for this screening study; results are not full-dataset training results.
- Keep test data outside screening and hyperparameter selection.

## 4. Inputs and supervised targets

Pin model, processor, tokenizer, and software revisions.

Use a fixed system instruction defining the task and allowed labels. The user input contains the image, a fixed field marker, and the post or plot. Preserve the instruction and label list in all validation conditions.

- **MMSD2.0:** output exactly one prescribed category marker, e.g. `sarcastic` or `not_sarcastic`. Freeze the exact vocabulary before running.
- **MM-IMDb:** output a JSON array of genre names in a fixed canonical order, without explanations.
- Apply training loss only to assistant answer tokens, not the prompt, image tokens, or padding.
- Train on complete image–text examples only. Do not introduce modality dropout, learned missing tokens, or auxiliary losses in this study.

Initial input limits:

| Setting | Value |
|---|---|
| MMSD2.0 body text | At most 128 tokenizer tokens |
| MM-IMDb plot | At most 384 tokenizer tokens |
| Text truncation | Keep the beginning; use the same rule everywhere |
| `min_pixels` | `128 * 28 * 28` |
| `max_pixels` | `256 * 28 * 28` |
| Initial total sequence limit | 1,536 tokens, including expanded visual tokens and target |

Verify actual expanded lengths. If shortening is necessary, shorten the body text without truncating image structure tokens or supervised labels. Final limits must be fixed before comparing runs.

## 5. Training settings and candidate matrix

### Shared starting settings

| Setting | Value |
|---|---|
| Precision | BF16, subject to the resource pilot |
| Micro-batch size | 1 |
| Gradient accumulation | 16; nominal effective batch size 16 |
| Gradient checkpointing | Enabled |
| Training `use_cache` | `False` |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Gradient clipping | Maximum norm 1.0 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| LoRA targets | LLM-layer `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Training horizon | 8 epochs |
| Schedule | Cosine decay with warmup ratio 0.05 |
| Training seeds | 42 for screening; 43 and 44 for confirmation |

Explicitly restrict LoRA module paths to the LLM. A broad suffix match must not accidentally adapt vision attention. Save the actual trainable module list and parameter counts. Freeze the vision encoder, original LLM weights, embeddings, and LM head in all configurations.

Construct the schedule for the complete 8-epoch run. Epochs 1 and 2 are prefixes of this run, not separately trained short-schedule experiments.

### Six configurations per dataset

| ID | Trainable components | Learning rate |
|---|---|---|
| L-low | LLM LoRA; frozen visual merger | `2e-5` |
| L-high | LLM LoRA; frozen visual merger | `1e-4` |
| M-low | Full visual merger; no LLM LoRA | `2e-5` |
| M-high | Full visual merger; no LLM LoRA | `1e-4` |
| LM-low | LLM LoRA and full visual merger | `2e-5` for both groups |
| LM-high | LLM LoRA and full visual merger | `1e-4` for both groups |

Merger-only adaptation is a control, not a LoRA method. This matrix screens a finite candidate set; it does not establish that one adaptation scope is generally superior. A later method comparison would require a fair tuning budget for each scope.

Start every run from the original backbone, not another configuration's checkpoint. Match LoRA initialization across L and LM configurations. Pair data order and seeds where possible, while recording differences caused by random-number consumption.

### Memory contingency

These values are starting choices, not guarantees of fitting in 24 GB. First run a resource pilot. If needed, reduce input budgets uniformly and freeze the revised protocol for all configurations. Do not silently use lower resolution for one configuration. If 4-bit QLoRA is required, create a separate group using the same quantization protocol for all candidates; do not mix its rankings with BF16 LoRA results.

## 6. Execution stages and budget

### A. Interface and resource pilot

- Use 32–64 examples from each dataset to verify inputs, target parsing, and all three validation paths.
- Test L, M, and LM adaptation scopes for up to approximately 100 optimizer steps each.
- Record allocated/reserved GPU memory, throughput, input lengths, gradient behavior, loss, and output-format errors.
- Lock any protocol changes before screening. Do not select learning rates because their pilot curves look likely to cross.

### B. Initial screening

- Run 6 configurations × 2 datasets × seed 42 = **12 complete training runs**.
- Run all candidates to epoch 8 without performance-based early stopping.
- Evaluate epoch 0 and epochs 1–8 under three conditions: **27 condition–checkpoint evaluations per run**, or 324 before reuse.
- Epoch-0 evaluations may be shared within a dataset only after verifying functional equivalence of initial checkpoints. Count the actual evaluations and cost.
- Estimate wall time from the pilot; do not assume a fixed GPU-hour budget before measuring throughput.

### C. Confirmation

Prefer repeating all configurations with seeds 43 and 44, for **36 total training runs**.

If compute is limited, independently confirm two or three configurations per dataset involved in a prespecified reversal candidate. If no reversal appears, confirm the fixed fallback set L-low, L-high, and LM-low. Explicitly label this as exploration followed by selective confirmation; it cannot estimate an unbiased overall reversal frequency.

### D. Held-out evaluation

After locking the protocol and selected comparisons, evaluate only the prespecified epoch-2 and epoch-8 checkpoints on test, under all three conditions. Do not search test checkpoints for a favorable crossing. Do not extend the horizon after seeing results just to obtain a reversal.

## 7. Three validation conditions at every checkpoint

Let higher values of the task score be better.

| Condition | Input intervention | Score |
|---|---|---|
| Full | Original image and original text | `S_full` |
| Mask image | Replace visual embeddings; preserve original text | `S_text` |
| Mask text | Preserve original image; replace body-text embeddings | `S_vision` |

These are interventions on the **same trained model**, not separately trained unimodal models.

### Current production protocol: symmetric mean placeholder plus attention exclusion

Use `symmetric-attention-v2` for the complete study. Replace image/body content embeddings with their frozen training means and block those content keys in every decoder layer, during prefill and cached generation. Keep task instructions, labels, delimiters, assistant boundary, sequence length, original image grid and mRoPE positions. Exclude hidden body token IDs from the repetition-penalty history. No encoder, missing-token embedding, or auxiliary objective is trained.

There are exactly 27 main evaluations per eight-epoch run. Stochastic-noise sensitivity runs below apply only to the legacy protocol; do not mix those results with this protocol. Fixed-length and OCR diagnostics show that retained structure and text inside images must be considered when interpreting the F1 differences.

For full releases, resolve exact content duplicate connected components before sampling, retaining rows only in their highest-priority original split (`test > dev > train`). Do not move rows or inspect model outputs to make this decision. Record every exclusion and report post-audit counts; use the entire retained development/test splits. Strict rejection remains available via `overlap_policy=error`.

### Historical protocol: fixed mean-plus-noise replacement

For modality `m`, replace each content-token embedding with:

`replacement_m = mu_m + 0.05 * std_m * epsilon`, where `epsilon ~ N(0, I)`.

- Intervene at the shared LLM input-embedding interface: after the visual merger for image tokens, and after token embedding lookup for body-text tokens.
- Compute per-dimension means and standard deviations once, using training examples and the original frozen checkpoint.
- Estimate visual statistics from original merger outputs and text statistics from body-token embeddings. Use equal example weighting, with equal token weighting within each example. Save the sampled IDs and exact estimator.
- Freeze these statistics across checkpoints and configurations, including runs that update the merger.
- Use noise seed 2026, with stable keys derived from example ID, modality, and token position. Reuse identical noise for each example across epochs and configurations.
- Preserve sequence lengths, attention masks, image grids, positional/mRoPE structure, field markers, task instructions, label lists, special tokens, and the assistant generation boundary.
- Replace only poster/image content or post/plot content, not the entire user message.

The coefficient 0.05 is a preregistered starting choice, not a theoretically neutral missing-modality distribution. Do not tune it after viewing the final curves.

### Implementation checks and limitations

- Inject replacements inside multimodal embedding assembly; do not assume an external `inputs_embeds` call is compatible with every generation path.
- Verify normal full-input inference remains unchanged when the hook is installed but inactive.
- Check prefill and cached decoding: masked content must not reappear through subsequent embedding reconstruction.
- Verify targeted spans by token indices, not character offsets. Confirm protected prompt and visual structure tokens remain intact.
- Log intervention location, tensor shapes, and embedding norms.
- Retained text length and image grid/aspect ratio still leak metadata. Images may also contain readable text.
- Fixed replacements may become more out of distribution as the merger changes. Treat masking scores as protocol-dependent reliance, not information-theoretic modality removal.

The legacy mean-noise protocol has exactly three main conditions per epoch. At epochs 0, 4, and 8, run separately accounted sensitivity checks with two additional noise seeds and mean-only replacement. A fixed small audit set may also use shuffled image–text pairings; this is another diagnostic, not ground truth for contribution.

## 8. Decoding and metrics

- Use `eval()`, disable dropout, and set `do_sample=False`.
- Start validation with batch size 1.
- Use `max_new_tokens=8` for MMSD2.0 and 128 for MM-IMDb, after confirming the longest valid target fits.
- Freeze parsing and normalization rules for all epochs, configurations, and conditions.
- MMSD2.0 accepts only the specified category markers. MM-IMDb accepts only the canonical 23 genre names in a JSON array, with duplicate removal and predefined whitespace/case normalization.
- Do not use an LLM judge, fuzzy semantic matching, or condition-specific scoring adjustments.
- Count invalid outputs as errors or empty label sets; never drop them. For binary classification, invalid output is a distinct invalid prediction, not a real class. Report class F1 with the fixed real-label set, ensuring an invalid prediction contributes a false negative for its true class. Also report invalid-output rate separately.

Report macro-F1, accuracy, and sarcastic-class F1 for MMSD2.0. Report micro-F1, macro-F1, and per-genre precision/recall/F1 for MM-IMDb. Also record truncation rate, average predicted-label count, and output length.

If early improvement is mainly a reduction in invalid output, do not attribute it to better modality utilization.

## 9. Contribution curves and required figures

For each checkpoint and each training seed:

```text
C_image(e) = S_full(e) - S_text(e)
C_text(e)  = S_full(e) - S_vision(e)
G(e)       = S_full(e) - max(S_text(e), S_vision(e))
           = min(C_image(e), C_text(e))
```

For **each dataset separately**, produce:

1. **Image contribution:** `C_image` versus epochs 0–8.
2. **Text contribution:** `C_text` versus epochs 0–8.
3. **Gain over the best ablated condition:** `G` versus epochs 0–8.

Use F1 percentage-point differences, preserve negative values, and draw a zero reference line. Use color for adaptation scope and line style for learning rate. The third curve is derived from the first two; do not label it causal synergy or count it as independent evidence.

Required supporting plots:

- Full-condition task performance for all six candidates, to identify ranking reversals.
- Raw full, mask-image, and mask-text scores.
- Invalid-output rates.
- Performance against cumulative GPU time, because equal epochs do not imply equal compute across adaptation scopes. Account for validation separately as well.

Calculate differences and the maximum within each seed before aggregation. Show individual-seed traces and means. If using a standard-deviation band, label it as such, not as a confidence interval.

## 10. Prespecified ranking-reversal criteria

The primary early checkpoint is **epoch 2**; the final checkpoint is **epoch 8**. Epoch 1 is supplementary.

For configurations A and B:

```text
d_2 = S_full_A(2) - S_full_B(2)
d_8 = S_full_A(8) - S_full_B(8)
```

Flag A as a candidate late winner over B when:

```text
d_2 <= -0.01 and d_8 >= +0.01
```

This requires a margin of at least one F1 percentage point at both endpoints. It is an exploratory practical threshold, not a statistical significance test.

For confirmation:

- Compute endpoint differences separately for every training seed; report all seeds. Seek the same reversal direction in at least two of three seeds.
- Inspect epochs 6–8 to distinguish sustained superiority from a transient crossing.
- Use 2,000 paired bootstrap resamples of development examples, recomputing F1 differences and `G`. Reuse each resample across configurations, epochs, and conditions.
- Resample movies for MM-IMDb. For MMSD2.0, resample examples, or shared-post/image groups if duplicates remain.
- Bootstrap intervals conditional on a trained checkpoint measure evaluation-sample uncertainty, not training randomness. Do not present three seeds as a precise population estimate.
- If making inferential claims across many pairs, address multiple comparisons; otherwise explicitly label results exploratory.

Report all candidate pairs, epoch-2 versus epoch-8 rank agreement, and whether early top-k selection retains the final winner. Report uncertainty and negative results, not only the most favorable crossing.

## 11. Interpreting possible delayed visual utilization

For every reversal pair, examine changes in all three raw scores and contribution measures between epochs 2 and 8.

An increase in `C_image` alone is insufficient: it can occur because `S_text` falls while full performance stays constant. That pattern may reflect degradation under the intervention rather than improved visual utilization.

A more supportive pattern is:

- Full performance improves and the configuration overtakes its comparator.
- Image-removal sensitivity increases.
- Text-preserved performance does not simply collapse.
- The effect is not explained by output formatting or a single noisy class.
- The direction remains reasonably stable under the prespecified masking sensitivity checks.

Use fixed per-genre analyses for MM-IMDb and class-wise analyses for MMSD2.0. Audit image-contained text and repeated inputs without selecting groups based on favorable results.

Even this pattern is only **consistent with delayed utilization of visual information**. Establishing a causal mechanism would require additional controlled training interventions. Neither dataset nor this protocol guarantees that the desired phenomenon occurs.

## 12. Artifacts and reproducibility

Save:

- Dataset manifests, source revisions, preprocessing rules, and split counts.
- Model/processor/software revisions and exact trainable-module lists.
- Per-epoch adapters, trained merger weights where applicable, optimizer/scheduler state, and RNG states.
- Masking means, standard deviations, estimator details, example IDs, and seeds.
- Per-example records: dataset, run/config ID, training seed, epoch, step, sample ID, condition, noise seed, reference labels, raw output, parsed predictions, invalid-output flag, and generated length.
- Aggregate metrics, all three raw scores, contribution values, GPU memory, and training/validation time.

Suggested outputs:

```text
artifacts/<dataset>/<config>/seed_<seed>/
  config.json
  checkpoints/epoch_<e>/
  predictions/epoch_<e>_<condition>.jsonl
  metrics.csv
  contributions.csv
figures/<dataset>/
  full_performance.png
  image_contribution.png
  text_contribution.png
  gain_over_best_ablation.png
  raw_condition_scores.png
  invalid_output_rate.png
  performance_vs_gpu_time.png
```

The best candidate is defined only within the six configurations and the fixed endpoint budget. This screening is intended to motivate later modality-aware low-fidelity selection, not establish a globally optimal pipeline or a complete RSI system.

## References

- [MMSD2.0: authors' code and corrected data](https://github.com/joeying1019/mmsd2.0)
- [MM-IMDb: authors' code and data entry point](https://github.com/johnarevalo/gmu-mmimdb)
- [Gated Multimodal Units for Information Fusion](https://arxiv.org/abs/1702.01992)
- [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)
- [Qwen fine-tuning documentation](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune)
- [EMAP: interpreting multimodal interactions](https://aclanthology.org/2020.emnlp-main.62/)

All hyperparameters, screening margins, and staged budgets above are proposed experimental choices, not literature-established optimal settings.

## Authorized symmetric masking extension

The independent `symmetric-attention-v2` protocol uses the same training recipe and fixed training-set means, replacing image/body content embeddings with their modality mean and excluding the corresponding keys at every language decoder layer during prefill and cached generation. Original sequence lengths, outer-model mRoPE positions, image grids, task instructions and generation boundaries are preserved. Repetition penalty applies to visible prompt and generated history only. Training receives complete image/text inputs; this extension introduces no training objective or modality dropout.

Select it explicitly with `--masking-method symmetric-attention-v2` or the launcher's `masking_method` field. Run configurations, pipeline identities and exact-resume signatures freeze this choice; never mix its curves with `mean-noise-v1`. Its deterministic mean/key-exclusion intervention has three evaluations per epoch (27 over epochs 0–8); noise-seed and mean-only sensitivity evaluations apply only to the original stochastic protocol. See README and `configs/real-symmetric-8epoch.json` for the separately versioned real-data subset trajectory.
