"""Normalize explicit source exports, audit, and freeze full-experiment manifests."""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .protocol import LABELS, SINGLE_LABEL, digest, read_jsonl, write_json, write_jsonl


def prepare(source, output, dataset, revision, pool_size='all', seed=42, overlap_policy='error'):
    from PIL import Image
    if dataset not in LABELS or pool_size!='all':
        raise ValueError('Full experiments require all retained training examples')
    if overlap_policy not in ('error','prefer-heldout'):
        raise ValueError('Unknown overlap policy')
    source, output = Path(source).resolve(), Path(output)
    if output.exists():
        raise ValueError('Manifest directory already exists; use a new immutable version')
    rows = read_jsonl(source)
    seen, complete, excluded = set(), [], []
    hashes = {kind: defaultdict(list) for kind in ('image', 'text')}
    for row in rows:
        sid, split = str(row['id']), row['split']
        if split not in ('train', 'dev', 'test'):
            raise ValueError(f'Invalid split: {split}')
        if sid in seen:
            raise ValueError(f'Duplicate ID / split overlap: {sid}')
        seen.add(sid)
        labels = row['labels']
        if (not isinstance(labels, list) or not labels or
                len(set(labels)) != len(labels) or
                any(x not in LABELS[dataset] for x in labels) or
                (dataset in SINGLE_LABEL and len(labels) != 1)):
            raise ValueError(f'Invalid labels: {sid}')
        image = (source.parent / row['image']).resolve()
        text = row.get('text', '')
        if not isinstance(text, str):
            raise ValueError(f'Text must be a string: {sid}')
        try:
            with Image.open(image) as im:
                im.verify()
            # JPEG verify() alone does not decode pixels and can miss truncated payloads.
            with Image.open(image) as im:
                rgb = im.convert('RGB')
                ihash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        except (OSError, ValueError, Image.DecompressionBombError):
            excluded.append(dict(id=sid, split=split, reason='missing/corrupt image'))
            continue
        if not text.strip():
            excluded.append(dict(id=sid, split=split, reason='empty text'))
            continue
        # Decoded RGB hashes ignore container metadata differences.
        thash = hashlib.sha256(' '.join(text.split()).encode()).hexdigest()
        item = dict(id=sid, split=split, image=str(image), text=text,
                    labels=[x for x in LABELS[dataset] if x in labels],
                    image_hash=ihash, text_hash=thash)
        complete.append(item)
        for kind, value in [('image', ihash), ('text', thash)]:
            hashes[kind][value].append(dict(id=sid, split=split))
    original_complete_counts=dict(Counter(r['split'] for r in complete))
    duplicates = {kind: [v for v in values.values() if len(v)>1]
                  for kind, values in hashes.items()}
    cross = {kind: [v for v in groups if len({r['split'] for r in v})>1]
             for kind, groups in duplicates.items()}
    # Keep complete duplicate components without training-pool sampling.
    # only in their highest-priority official split; never move or relabel rows.
    overlap_excluded=[]
    if any(cross.values()) and overlap_policy=='error':
        write_json(str(output) + '.overlap.json', cross)
        raise ValueError('Cross-split duplicate content; resolve before freezing manifests (see overlap report)')
    # Union shared image/text groups for grouped bootstrap.
    parent = {r['id']: r['id'] for r in complete}
    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for groups in duplicates.values():
        for group in groups:
            for row in group[1:]:
                parent[root(row['id'])] = root(group[0]['id'])
    for row in complete:
        row['group'] = root(row['id'])
    if any(cross.values()) and overlap_policy=='prefer-heldout':
        priority={'train':0,'dev':1,'test':2}
        highest={}
        for row in complete:
            group=row['group'];highest[group]=max(highest.get(group,-1),priority[row['split']])
        kept=[]
        for row in complete:
            if priority[row['split']]<highest[row['group']]:
                overlap_excluded.append(dict(id=row['id'],split=row['split'],group=row['group'],reason='cross-split duplicate component; prefer test > dev > train'))
            else:kept.append(row)
        complete=kept
    train = sorted((r for r in complete if r['split']=='train'), key=lambda r: r['id'])
    if not train or not any(r['split']=='dev' for r in complete):
        raise ValueError('Nonempty train and development splits are required')
    splits = {s: sorted(train if s=='train' else [r for r in complete if r['split']==s], key=lambda r:r['id'])
              for s in ('train', 'dev', 'test')}
    for split, items in splits.items():
        write_jsonl(output / f'{split}.jsonl', items)
    audit = random.Random(2026).sample(train, min(64, len(train)))
    write_jsonl(output/'ocr_audit.jsonl', [dict(id=r['id'], image=r['image'], embedded_text_review=None) for r in audit])
    write_json(output/'manifest.json', dict(dataset=dataset, source=str(source), source_sha256=digest(source),
        source_revision=revision, labels=LABELS[dataset], seed=seed, pool_size=len(train), requested_pool_size=pool_size,
        label_counts={s:{label:sum(label in r['labels'] for r in items) for label in LABELS[dataset]} for s,items in splits.items()},
        counts={s:len(v) for s,v in splits.items()}, excluded=excluded, duplicates=duplicates,
        overlap_policy=overlap_policy,cross_split_duplicates=cross,overlap_excluded=overlap_excluded,
        split_hashes={s:digest(output/f'{s}.jsonl') for s in splits},
        original_complete_counts=original_complete_counts,
        retained_complete_counts=dict(Counter(r['split'] for r in complete)),
        estimator_grouping='connected components of exact RGB image / whitespace-normalized text duplicates'))
