"""Independently check each pilot export against released annotations and byte hashes."""
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image
from modalitygap.protocol import digest, read_jsonl, write_json

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/real-expansion-v1'
DOWNLOADS=ROOT/'data/downloads/expansion-v1'
ARTIFACTS=ROOT/'artifacts/real-expansion-symmetric-8epoch-v1'


def main():
    report={}
    assert digest(DOWNLOADS/'visualnews-official-data.json')==digest(DOWNLOADS/'twelcone--VisualNews--data.json')
    news=json.loads((DOWNLOADS/'visualnews-official-data.json').read_text())
    news={r['id']:r for r in news}
    for task in ('newsclippings','crisismmd','hatefulmemes'):
        source=DATA/task/'source.jsonl';rows=read_jsonl(source)
        provenance=json.loads((DATA/task/'provenance.json').read_text())
        assert digest(source)==provenance['source_sha256']
        assert len(rows)==96
        hashes={r['id']:r for r in provenance['images']}
        official={}
        if task=='newsclippings':
            for s,alias in [('train','train'),('dev','val')]:
                aa=json.loads((DOWNLOADS/f'news-{alias}.json').read_text())['annotations']
                for i,a in enumerate(aa):official[f'{alias}-{i}']=(s,a)
        elif task=='crisismmd':
            folder=ROOT/'data/sources/expansion-v1/crisis-author/data/task_data'
            for s in ('train','dev'):
                for a in csv.DictReader((folder/f'task_informative_text_img_agreed_lab_{s}.tsv').open(),delimiter='\t'):
                    official[a['image_id']]=(s,a)
        else:
            for s,alias in [('train','train'),('dev','dev_seen')]:
                for a in read_jsonl(DOWNLOADS/f'neuralcatcher--hateful_memes--{alias}.jsonl'):
                    official[str(a['id'])]=(s,a)
        for r in rows:
            s,a=official[r['id']];assert s==r['split'] and a==r['original_record']
            if task=='newsclippings':
                assert r['text']==news[a['id']]['caption']
                assert r['original_image_path']==news[a['image_id']]['image_path'].removeprefix('./')
                assert r['labels']==['mismatched' if a['falsified'] else 'matched']
            elif task=='crisismmd':
                assert r['text']==a['tweet_text'] and r['original_image_path']==a['image']
                assert r['labels']==[a['label']]==[a['label_text']]==[a['label_image']]
            else:
                assert r['text']==a['text'] and r['original_image_path']==a['img']
                assert r['labels']==[('not_hateful','hateful')[a['label']]]
            assert digest(r['image'])==hashes[r['id']]['original_byte_sha256']
            with Image.open(r['image']) as im:
                rgb=im.convert('RGB');rgb.load()
                assert hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()==hashes[r['id']]['rgb_sha256']
        assert {r['id'] for r in rows if r['split']=='train'}.isdisjoint(r['id'] for r in rows if r['split']=='dev')
        for s,n in [('train',64),('dev',32)]:
            chosen=[r for r in rows if r['split']==s];assert len(chosen)==n
            assert sorted(sum(r['labels']==[label] for r in chosen) for label in {r['labels'][0] for r in chosen})==[n//2,n//2]
        if task=='newsclippings':
            for s in ('train','dev'):
                selected=[r for r in rows if r['split']==s]
                for cid in {r['original_record']['id'] for r in selected}:
                    pair=[r for r in selected if r['original_record']['id']==cid]
                    assert len(pair)==2 and {tuple(r['labels']) for r in pair}=={('matched',),('mismatched',)}
        workspace=ARTIFACTS/task;manifest=json.loads((workspace/'manifests'/task/'manifest.json').read_text())
        assert manifest['source_sha256']==digest(source) and manifest['counts']==dict(train=64,dev=32,test=0)
        assert not manifest['excluded']
        train=read_jsonl(workspace/'manifests'/task/'train.jsonl');dev=read_jsonl(workspace/'manifests'/task/'dev.jsonl')
        for k in ('image_hash','text_hash'):assert {r[k] for r in train}.isdisjoint(r[k] for r in dev)
        stats=json.loads((workspace/'stats'/f'{task}.json').read_text())
        assert set(stats['ids'])=={r['id'] for r in train}
        report[task]=dict(status='passed',selected=96,official_sample_ids_text_labels_image_mapping_and_splits_match=True,
                         all_original_byte_and_decoded_rgb_hashes_match=True,train_dev_content_disjoint=True,
                         statistics_only_use_64_training_examples=True,source_sha256=digest(source),
                         image_provenance_limit=provenance.get('provenance_limit',provenance.get('metadata_limit')))
    write_json(ARTIFACTS/'data-audit.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
