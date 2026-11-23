import hashlib
import json
from pathlib import Path

MODEL_REVISION = '66285546d2b821cf421d4f5eb2576359d3770cd3'

GENRES = ('Action', 'Adventure', 'Animation', 'Biography', 'Comedy',
          'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir',
          'History', 'Horror', 'Music', 'Musical', 'Mystery', 'Romance',
          'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western')
LABELS = {'mmsd2': ('not_sarcastic', 'sarcastic'), 'mmimdb': GENRES,
          'newsclippings': ('matched', 'mismatched'),
          'crisismmd': ('not_informative', 'informative'),
          'hatefulmemes': ('not_hateful', 'hateful')}
SINGLE_LABEL = frozenset(LABELS) - {'mmimdb'}
TASK_PROMPTS = {
    'mmsd2': 'Classify sarcasm. Output exactly one category marker: not_sarcastic or sarcastic.',
    'newsclippings': 'Decide whether the news image and its caption match the same real-world context. Output exactly one category marker: matched or mismatched.',
    'crisismmd': 'Classify whether this disaster-related social media post provides information useful for crisis response. Output exactly one category marker: not_informative or informative.',
    'hatefulmemes': 'Classify whether this meme expresses hate toward a protected group. Consider the image and text together. Output exactly one category marker: not_hateful or hateful.',
}


def generation_limit(dataset):
    return 128 if dataset == 'mmimdb' else 8


def task_prompt(dataset):
    if dataset == 'mmimdb':
        return 'Classify movie genres. Output only a JSON array in this canonical order: '+json.dumps(LABELS[dataset])+'.'
    return TASK_PROMPTS[dataset]

CONDITIONS = ('full', 'mask_image', 'mask_text')
CONFIGS = {f'{scope}-{rate}': {'scope': scope, 'lr': lr}
           for scope in ('L', 'M', 'LM')
           for rate, lr in (('low', 2e-5), ('high', 1e-4))}
DEFAULTS = dict(precision='bf16-mixed-fp32-trainable', model='Qwen/Qwen2.5-VL-3B-Instruct', epochs=8,
                accumulation=16, rank=16, alpha=32, dropout=0.05,
                weight_decay=0.01, warmup=0.05, max_grad_norm=1.0,
                min_pixels=128*28*28, max_pixels=256*28*28,
                max_length=1536, noise_seed=2026, noise_scale=0.05)


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def target(dataset, labels):
    ordered = [x for x in LABELS[dataset] if x in labels]
    return ordered[0] if dataset in SINGLE_LABEL else json.dumps(ordered)


def parse(dataset, raw):
    if dataset in SINGLE_LABEL:
        value = raw.strip()
        return ([value], False) if value in LABELS[dataset] else ([], True)
    try:
        values = json.loads(raw.strip())
    except (ValueError, TypeError):
        return [], True
    if not isinstance(values, list) or any(not isinstance(x, str) for x in values):
        return [], True
    lookup = {x.lower(): x for x in GENRES}
    normalized = [lookup.get(x.strip().lower()) for x in values]
    if None in normalized:
        return [], True
    return [x for x in GENRES if x in normalized], False


def metrics(dataset, rows):
    if not rows:
        raise ValueError('Cannot score an empty evaluation set')
    per_label = {}
    totals = [0, 0, 0]
    for label in LABELS[dataset]:
        tp = sum(label in r['reference'] and label in r['prediction'] for r in rows)
        fp = sum(label not in r['reference'] and label in r['prediction'] for r in rows)
        fn = sum(label in r['reference'] and label not in r['prediction'] for r in rows)
        p, recall = tp / (tp+fp) if tp+fp else 0., tp / (tp+fn) if tp+fn else 0.
        per_label[label] = dict(precision=p, recall=recall, f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
        totals = [a+b for a, b in zip(totals, (tp, fp, fn))]
    tp, fp, fn = totals
    result = dict(macro_f1=sum(x['f1'] for x in per_label.values())/len(per_label),
                  micro_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                  invalid_rate=sum(r['invalid'] for r in rows)/len(rows),
                  predicted_label_count=sum(len(r['prediction']) for r in rows)/len(rows),
                  output_length=sum(r.get('generated_length', 0) for r in rows)/len(rows),
                  truncation_rate=sum(r.get('truncated', False) for r in rows)/len(rows),
                  per_label=per_label, count=len(rows))
    if dataset in SINGLE_LABEL:
        result.update(accuracy=sum(r['reference'] == r['prediction'] for r in rows)/len(rows))
    if dataset == 'mmsd2':
        result['sarcastic_f1'] = per_label['sarcastic']['f1']
    result['score'] = result['macro_f1' if dataset in SINGLE_LABEL else 'micro_f1']
    return result


def contributions(scores):
    image = scores['full'] - scores['mask_image']
    text = scores['full'] - scores['mask_text']
    return dict(image_contribution=image, text_contribution=text,
                gain_over_best_ablation=min(image, text))
