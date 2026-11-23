"""Pinned original-data acquisition and model-blind 64/32 pilot selection.

Never selects on model outputs; never trains/evaluates on test records.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import time

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT/'data/downloads/expansion-v1'
OUTPUT = ROOT/'data/real-expansion-v1'
PINS = {'crisis': ('QCRI/CrisisMMD','10a5626ba112ab9c50369c473a2062bcf205d708'),
        'hateful': ('neuralcatcher/hateful_memes','d201c488dc7024623d1ecbcc987b3f132c4c2e12')}


def sha(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        for attempt in range(4):
            try:
                r = requests.get(url, timeout=90)
                r.raise_for_status()
                tmp = path.with_suffix(path.suffix+'.part')
                tmp.write_bytes(r.content)
                tmp.replace(path)
                break
            except requests.RequestException:
                if attempt == 3: raise
                time.sleep(attempt+1)
    return path


def normalized(text):
    return ' '.join(text.split())


def balanced(rows, split, n, labels, used_text):
    """Stable class-balanced sample; reject duplicate body groups across pilot splits."""
    rng = random.Random(42)
    selected = []
    for label in labels:
        candidates = sorted((r for r in rows if r['labels']==[label]), key=lambda r:r['id'])
        rng.shuffle(candidates)
        count = 0
        for row in candidates:
            text = normalized(row['text'])
            if not text or text in used_text: continue
            selected.append(dict(row, split=split)); used_text.add(text); count += 1
            if count == n//len(labels): break
        if count != n//len(labels): raise ValueError('Insufficient distinct candidate text groups')
    return sorted(selected,key=lambda r:r['id'])


def freeze(name, rows, provenance):
    out = OUTPUT/name
    if (out/'source.jsonl').exists():
        raise ValueError(f'Refusing to overwrite frozen source: {out}')
    audit = []
    for r in rows:
        file = Path(r['image'])
        with Image.open(file) as im:
            rgb = im.convert('RGB'); rgb.load()
            rgb_hash = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
            size = list(rgb.size)
        audit.append(dict(id=r['id'],split=r['split'],labels=r['labels'],
                          original_byte_sha256=sha(file),rgb_sha256=rgb_hash,size=size,
                          source_path=r.get('original_image_path'),original_record=r.get('original_record')))
    # Fail closed on selected train/dev duplicate decoded pictures or normalized body text.
    for key in ('rgb_sha256',):
        splits = defaultdict(set)
        for a in audit: splits[a[key]].add(a['split'])
        if any(len(s)>1 for s in splits.values()): raise ValueError('Selected cross-split RGB duplicates')
    source = ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    out.mkdir(parents=True,exist_ok=True)
    (out/'source.jsonl').write_text(source)
    save(out/'provenance.json',dict(**provenance,seed=42,selection='model-blind; class balanced 64 train / 32 dev',
         source_sha256=sha(out/'source.jsonl'),counts=dict(Counter(r['split'] for r in rows)),
         images=audit,test_used_for_training_or_evaluation=False))
    print(name,'frozen',len(rows),flush=True)


def crisis():
    repo,pin=PINS['crisis']; author=ROOT/'data/sources/expansion-v1/crisis-author/data/task_data'
    rows=[]; counts={}; used=set(); tables={}
    hf={s:{r['image_id']:r for r in json.loads((DOWNLOADS/f'crisis-{s}.json').read_text())} for s in ('train','dev','test')}
    for s in ('train','dev','test'):
        f=author/f'task_informative_text_img_agreed_lab_{s}.tsv'
        data=list(csv.DictReader(f.open(),delimiter='\t'));counts[s]=len(data);tables[s]=sha(f)
        for r in data:
            assert r['label']==r['label_text']==r['label_image']
            h=hf[s][r['image_id']]
            assert (h['label'],h['image_path'],h['tweet_text']) == (r['label'],r['image'],r['tweet_text'])
        if s=='test':continue
        candidates=[dict(id=r['image_id'],text=r['tweet_text'],labels=[r['label']],
                         original_image_path=r['image'],original_record=r) for r in data]
        rows+=balanced(candidates,s,64 if s=='train' else 32,('not_informative','informative'),used)
    out=OUTPUT/'crisismmd'
    def load(r):
        r['image']=str(fetch(f'https://huggingface.co/datasets/{repo}/resolve/{pin}/{r["original_image_path"]}',out/r['original_image_path']))
    with ThreadPoolExecutor(8) as ex:list(ex.map(load,rows))
    freeze('crisismmd',rows,dict(official_image_repo=repo,image_revision=pin,
        official_split_repo='firojalam/multimodal_social_media@8862149a36f485f980cfd6ac719cba65e296647c',
        split_sha256=tables,official_agreed_label_counts=counts,
        crosscheck='Every agreed-label row checked against QCRI split image_id, text, image path, and label; text/image labels equal.',
        text_encoding='Original released Unicode retained, including pre-existing mojibake; no text repair.',
        task='informativeness; author text-image-agreed labels'))


def hateful():
    repo,pin=PINS['hateful']; used=set(); rows=[]; counts={}; hashes={}
    for official,s,n in [('train','train',64),('dev_seen','dev',32),('dev_unseen','unused',0)]:
        f=DOWNLOADS/f'neuralcatcher--hateful_memes--{official}.jsonl'
        data=[json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        counts[official]=len(data);hashes[official]=sha(f)
        assert len({r['id'] for r in data})==len(data)
        if not n:continue
        candidates=[dict(id=str(r['id']),text=r['text'],labels=[('not_hateful','hateful')[r['label']]],
                         original_image_path=r['img'],original_record=r) for r in data]
        rows+=balanced(candidates,s,n,('not_hateful','hateful'),used)
    assert counts==dict(train=8500,dev_seen=500,dev_unseen=540)
    out=OUTPUT/'hatefulmemes'
    for name in ('train','dev_seen','dev_unseen'):
        assert (DOWNLOADS/f'hateful-secondary-{name}.jsonl').read_bytes()==(DOWNLOADS/f'neuralcatcher--hateful_memes--{name}.jsonl').read_bytes()
    primary_files={x['rfilename'] for x in json.loads((DOWNLOADS/'neuralcatcher--hateful_memes-info.json').read_text())['siblings']}
    def load(r):
        source_repo,source_pin=repo,pin
        if r['original_image_path'] not in primary_files:
            source_repo,source_pin='panjiyarsunil/hateful-memes-data','759279f46912fafad49dfef472924e899198ec69'
        r['image_source']=source_repo+'@'+source_pin
        r['image']=str(fetch(f'https://huggingface.co/datasets/{source_repo}/resolve/{source_pin}/{r["original_image_path"]}',out/r['original_image_path']))
    with ThreadPoolExecutor(8) as ex:list(ex.map(load,rows))
    freeze('hatefulmemes',rows,dict(mirror_repo=repo,mirror_revision=pin,
        annotation_sha256=hashes,secondary_mirror="panjiyarsunil/hateful-memes-data@759279f46912fafad49dfef472924e899198ec69; all three annotation files byte-identical",official_release_counts=counts,development_split='dev_seen',
        provenance_limit='Public mirror of original-release JSONL/PNG. Author release schema/counts and published label meaning verified; no official original-archive checksum available to authenticate byte identity.',
        ocr_caveat='Original PNG includes rendered body text: mask_text hides the separate body input, not text visible inside the image.',
        reported_metric='Macro-F1 for modality contribution; not official benchmark AUROC.'))


def news_select():
    metadata=json.loads((DOWNLOADS/'twelcone--VisualNews--data.json').read_text())
    lookup={r['id']:r for r in metadata}; used=set();selected=[];counts={};hashes={}
    for split,official,n in [('train','train',32),('dev','val',16),('test','test',0)]:
        file=DOWNLOADS/f'news-{official}.json'; annotations=json.loads(file.read_text())['annotations']
        counts[official]=len(annotations);hashes[official]=sha(file)
        pairs=defaultdict(list)
        for i,a in enumerate(annotations):pairs[a['id']].append((i,a))
        assert all(len(v)==2 and {a['falsified'] for _,a in v}=={False,True} for v in pairs.values())
        if not n:continue
        candidates=sorted(pairs);random.Random(42).shuffle(candidates)
        taken=0
        for cid in candidates:
            text=lookup[cid]['caption']; key=normalized(text)
            if not key or key in used:continue
            used.add(key)
            for idx,a in pairs[cid]:
                image_path=lookup[a['image_id']]['image_path'].removeprefix('./')
                selected.append(dict(id=f'{official}-{idx}',split=split,text=text,
                  labels=['mismatched' if a['falsified'] else 'matched'],
                  image=str(OUTPUT/'newsclippings'/'images'/image_path),original_image_path=image_path,original_record=a))
            taken+=1
            if taken==n:break
        assert taken==n
    save(OUTPUT/'newsclippings'/'selection.json',dict(rows=selected,official_counts=counts,
        annotations_sha256=hashes,metadata_sha256=sha(DOWNLOADS/'twelcone--VisualNews--data.json')))
    print('news selection frozen',len(selected),flush=True)


def news_freeze():
    plan=json.loads((OUTPUT/'newsclippings/selection.json').read_text())
    metadata_audit=DOWNLOADS/'visualnews-metadata-audit.json'
    assert json.loads(metadata_audit.read_text())['mirror_byte_identical_to_author_metadata']
    freeze('newsclippings',plan.pop('rows'),dict(**plan,metadata_original_byte_verified=True,metadata_audit_sha256=sha(metadata_audit),
       annotation_repo='g-luo/news-clippings@a2bbe9a8125f47a5c26338125c71e05be0266cf7',
       metadata_mirror='twelcone/VisualNews@8a68a1f31d3ed8ac30a86e8e78478f45dd80bc24',
       original_image_archive='https://www.cs.rice.edu/~vo9/visualnews/origin.tar',
       pair_selection='32 train / 16 dev complete caption pairs, matched and mismatched each; caption=lookup[id], image=lookup[image_id].',
       metadata_limit='Metadata byte-identical (SHA256 verified) to author origin/data.json; original JPEG bytes extracted directly from VisualNews author archive.'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['crisis','hateful','news-select','news-freeze']);args=ap.parse_args()
    {'crisis':crisis,'hateful':hateful,'news-select':news_select,'news-freeze':news_freeze}[args.stage]()
