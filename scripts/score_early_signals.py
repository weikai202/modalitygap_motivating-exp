"""Score canonical answer tokens on the existing symmetric-mask dev checkpoints."""
import argparse
import csv
import json
import re
import time
from pathlib import Path

from modalitygap.protocol import CONDITIONS, DEFAULTS, digest, read_jsonl, target, write_json
from modalitygap.runner import load_trainable, make_intervention, verified_data

SOURCE = Path('artifacts/real-symmetric-8epoch-v1')
OUTPUT = Path('artifacts/early-signals-v1')
DATASETS = ('mmsd2', 'mmimdb')
EPOCHS = (0, 1, 2)


def run_path(dataset):
    return SOURCE / 'runs' / dataset / 'L-low' / 'seed_42'


def score_tokens(model, hook, encoder, offsets, row, condition, device):
    import torch

    batch, body, _ = encoder(row, training=True)
    answer = target(encoder.dataset, row['labels'])
    encoded = offsets(answer + offsets.eos_token, add_special_tokens=False,
                      return_offsets_mapping=True)
    labels = batch.pop('labels')[0]
    positions = labels.ne(-100).nonzero().flatten()
    if labels[positions].tolist() != encoded['input_ids']:
        raise ValueError('Scoring tokenizer does not match checkpoint target encoding')
    spans = [(0, len(answer))] if encoder.dataset == 'mmsd2' else [
        match.span(1) for match in re.finditer(r'"([^"]+)"', answer)
    ]
    is_label = [any(start < hi and end > lo for lo, hi in spans)
                for start, end in encoded['offset_mapping']]
    if spans and not any(is_label):
        raise ValueError('No token overlaps the reference labels')
    batch = {name: value.to(device) for name, value in batch.items()}
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16,
                                                 enabled=device == 'cuda'):
        with hook.use(batch, body, condition):
            logits = model(**batch, use_cache=False,
                           logits_to_keep=(positions - 1).to(device)).logits[0]
        losses = torch.nn.functional.cross_entropy(
            logits.float(), labels[positions].to(device), reduction='none').tolist()
    hook.logs.clear()
    return {**{f'{name}_nll_sum': sum(loss for loss, label in zip(losses, is_label)
                                   if label == select_label)
               for name, select_label in [('label', True), ('format', False)]},
            'label_tokens': sum(is_label), 'format_tokens': len(is_label) - sum(is_label)}


def evaluate(dataset, device, *, run=None, output=OUTPUT):
    import torch
    from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
    from modalitygap.model import Encoder, adapt, load_model

    run = run_path(dataset) if run is None else run
    output.mkdir(parents=True, exist_ok=True)
    saved = json.loads((run / 'config.json').read_text())
    data = SOURCE / 'manifests' / dataset
    verified_data(data, dataset)
    stats_path = SOURCE / 'stats' / f'{dataset}.pt'
    if digest(stats_path) != saved['stats_sha256'] or digest(
            data / 'manifest.json') != saved['data_manifest_sha256']:
        raise ValueError('Dev manifest or mask statistics differ from the saved run')
    rows = read_jsonl(data / 'dev.jsonl')
    options = dict(revision=saved['revision'], cache_dir='artifacts/hf-cache',
                   local_files_only=True)
    if device == 'cpu':
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            DEFAULTS['model'], dtype=torch.float32, attn_implementation='sdpa', **options)
        processor = AutoProcessor.from_pretrained(
            DEFAULTS['model'], min_pixels=DEFAULTS['min_pixels'],
            max_pixels=DEFAULTS['max_pixels'], use_fast=False, **options)
    else:
        model, processor = load_model(saved['revision'], options['cache_dir'])
    offsets = AutoTokenizer.from_pretrained(DEFAULTS['model'], use_fast=True, **options)
    model, _ = adapt(model, saved['scope'], saved['seed'])
    model.gradient_checkpointing_disable()
    model.eval()
    encoder = Encoder(processor, dataset)
    hook = make_intervention(model, torch.load(stats_path, map_location='cpu', weights_only=True),
                             saved['masking_method'])
    try:
        for epoch in EPOCHS:
            checkpoint = run / 'checkpoints' / f'epoch_{epoch}'
            metadata = json.loads((checkpoint / 'complete.json').read_text())
            weights = checkpoint / 'trainable.pt'
            if digest(weights) != metadata['trainable_sha256']:
                raise ValueError(f'Checkpoint checksum mismatch: {checkpoint}')
            load_trainable(model, torch.load(weights, map_location='cpu', weights_only=True))
            started = time.monotonic()
            scores = []
            for row in rows:
                for condition in CONDITIONS:
                    score = score_tokens(model, hook, encoder, offsets, row, condition, device)
                    scores.append(dict(id=row['id'], condition=condition, **score))
            write_json(output / f'{dataset}-epoch-{epoch}.json', {
                'dataset': dataset, 'epoch': epoch, 'device': device,
                'precision': 'fp32' if device == 'cpu' else 'bf16 autocast',
                'checkpoint_sha256': metadata['trainable_sha256'],
                'seconds': time.monotonic() - started, 'scores': scores,
            })
            print(dataset, epoch, len(scores), round(time.monotonic() - started, 1), flush=True)
    finally:
        hook.close()


def summarize():
    records, slopes = [], []
    for dataset in DATASETS:
        run = run_path(dataset)
        with (run / 'metrics.csv').open() as stream:
            metrics = {(int(row['epoch']), row['condition']): row
                       for row in csv.DictReader(stream) if row['condition'] == row['variant']}
        epochs = {}
        for epoch in EPOCHS:
            path = OUTPUT / f'{dataset}-epoch-{epoch}.json'
            scores = read_jsonl(run / 'predictions' / f'epoch_{epoch}_full.jsonl')
            token_rows = json.loads(path.read_text())['scores'] if path.exists() else []
            row = {}
            for condition in CONDITIONS:
                metric = metrics[epoch, condition]
                subset = [score for score in token_rows if score['condition'] == condition]
                if subset and {score['id'] for score in subset} != {score['id'] for score in scores}:
                    raise ValueError('Scored dev IDs differ from the original predictions')
                for group in ('label', 'format'):
                    count = sum(score[f'{group}_tokens'] for score in subset)
                    row[f'{condition}_{group}_log_ppl'] = (
                        sum(score[f'{group}_nll_sum'] for score in subset) / count if count else None)
                row[f'{condition}_f1'] = float(metric['score'])
                row[f'{condition}_invalid_rate'] = float(metric['invalid_rate'])
            for modality in ('image', 'text'):
                row[f'{modality}_contribution'] = row['full_f1'] - row[f'mask_{modality}_f1']
                masked = row[f'mask_{modality}_label_log_ppl']
                row[f'{modality}_loss_gap'] = (
                    masked - row['full_label_log_ppl'] if masked is not None else None)
            epochs[epoch] = row
            records.append(dict(dataset=dataset, epoch=epoch, **row))
        # OLS on equally spaced epochs 0, 1, 2 has slope (y2 - y0) / 2.
        slope = {key: (epochs[2][key] - epochs[0][key]) / 2
                 if all(epochs[epoch][key] is not None for epoch in EPOCHS) else None
                 for key in epochs[0]}
        slopes.append(dict(dataset=dataset, **slope))
    for filename, values in [('epochs.csv', records), ('slopes.csv', slopes)]:
        with (OUTPUT / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=values[0])
            writer.writeheader()
            writer.writerows(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summarize()
    if not args.summarize_only:
        for dataset in DATASETS:
            evaluate(dataset, args.device)
            summarize()


if __name__ == '__main__':
    main()
