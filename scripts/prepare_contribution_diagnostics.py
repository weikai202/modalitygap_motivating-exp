"""Freeze a model-blind expanded-dev diagnostic; never modify prior datasets/results."""
from collections import Counter
import csv,json,random,hashlib
from pathlib import Path
from PIL import Image
from prepare_expansion import ROOT,DOWNLOADS,PINS,fetch,normalized
from modalitygap.protocol import read_jsonl,write_json,write_jsonl,digest

OUT=ROOT/'artifacts/contribution-cause-diagnostics-v1'


def main():
    if OUT.exists():raise ValueError('Use a fresh diagnostic version')
    OUT.mkdir(parents=True)
    write_json(OUT/'plan.json',dict(datasets=['crisismmd','hatefulmemes'],checkpoint_epoch=8,
        original_dev=32,additional_dev=96,total_dev=128,selection_seed=2026,
        conditions=['full','mask_image','mask_text'],probes_per_dataset=8,
        controls=['same-structure hidden-content replacement','fixed masked-body lengths 1 and 64','joint key blocking','Crisis URL/mention removal','image-only OCR'],
        selection='Class-balanced deterministic shuffled official-dev candidates; no model output used; reject repeated normalized text/RGB including training data.',
        no_training=True,no_test_evaluation=True,old_results_untouched=True))
    for task in ('crisismmd','hatefulmemes'):
        source=ROOT/'data/real-expansion-v1'/task/'source.jsonl';allrows=read_jsonl(source)
        old=[r for r in allrows if r['split']=='dev'];rows=list(old)
        train_ids={r['id'] for r in allrows if r['split']=='train'};ids={r['id'] for r in allrows}
        texts={normalized(r['text']) for r in allrows};rgb=set()
        for r in allrows:
            with Image.open(r['image']) as im:
                x=im.convert('RGB');rgb.add(hashlib.sha256(str(x.size).encode()+x.tobytes()).hexdigest())
        if task=='crisismmd':
            path=ROOT/'data/sources/expansion-v1/crisis-author/data/task_data/task_informative_text_img_agreed_lab_dev.tsv'
            candidates=[dict(id=r['image_id'],text=r['tweet_text'],labels=[r['label']],original_image_path=r['image'],original_record=r) for r in csv.DictReader(path.open(),delimiter='\t')]
            repo,pin=PINS['crisis'];labels=('not_informative','informative')
        else:
            candidates=[dict(id=str(r['id']),text=r['text'],labels=[('not_hateful','hateful')[r['label']]],original_image_path=r['img'],original_record=r) for r in read_jsonl(DOWNLOADS/'neuralcatcher--hateful_memes--dev_seen.jsonl')]
            repo,pin=PINS['hateful'];labels=('not_hateful','hateful')
        excluded=[];rng=random.Random(2026)
        for label in labels:
            pool=sorted((r for r in candidates if r['labels']==[label]),key=lambda r:r['id']);rng.shuffle(pool);count=0
            for r in pool:
                if r['id'] in ids or normalized(r['text']) in texts:continue
                image_repo,image_pin=repo,pin
                if task=='hatefulmemes':
                    info=json.loads((DOWNLOADS/'neuralcatcher--hateful_memes-info.json').read_text())
                    files={x['rfilename'] for x in info['siblings']}
                    if r['original_image_path'] not in files:image_repo,image_pin='panjiyarsunil/hateful-memes-data','759279f46912fafad49dfef472924e899198ec69'
                image=fetch(f'https://huggingface.co/datasets/{image_repo}/resolve/{image_pin}/{r["original_image_path"]}',OUT/'data'/task/r['original_image_path'])
                with Image.open(image) as im:
                    x=im.convert('RGB');h=hashlib.sha256(str(x.size).encode()+x.tobytes()).hexdigest()
                if h in rgb:excluded.append(dict(id=r['id'],reason='RGB duplicate'));continue
                rgb.add(h);texts.add(normalized(r['text']));ids.add(r['id'])
                rows.append(dict(r,split='dev',image=str(image),image_hash=h,source_revision=f'{image_repo}@{image_pin}'))
                count+=1
                if count==48:break
            assert count==48
        assert len(rows)==128 and not ({r['id'] for r in rows}&train_ids)
        folder=OUT/task;folder.mkdir(parents=True)
        write_jsonl(folder/'dev128.jsonl',rows)
        write_json(folder/'data-audit.json',dict(source_sha256=digest(source),original_dev_ids=[r['id'] for r in old],
            original32_preserved=True,additional96=[r['id'] for r in rows[32:]],counts=dict(Counter(r['labels'][0] for r in rows)),
            train_id_text_rgb_disjoint=True,excluded=excluded,dev_manifest_sha256=digest(folder/'dev128.jsonl')))
        print(task,'frozen 128 development examples',flush=True)


if __name__=='__main__':main()
