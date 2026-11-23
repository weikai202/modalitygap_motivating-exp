"""Fixed real-data pilot: official corrected text/labels, mirrored original JPEG bytes."""
import io
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image
import pyarrow.parquet as pq

from modalitygap.data import prepare
from modalitygap.protocol import digest,write_json,write_jsonl


def main():
    root=Path('data/real-pilot/mmsd2')
    if root.exists():raise ValueError('Use a fresh real pilot directory')
    author=Path('data/sources/mmsd2-authors/data/text_json_final')
    mirror=Path('data/hf-cache/datasets--coderchen01--MMSD2.0/snapshots/5b39e5b844accf2331d1738a3d91d4cb59005a50/mmsd-v2')
    official={split:json.loads((author/f'{source}.json').read_text()) for split,source in (('train','train'),('dev','valid'),('test','test'))}
    ids={s:{str(r['image_id']) for r in rows} for s,rows in official.items()}
    if any(len(ids[s])!=len(rows) for s,rows in official.items()):raise ValueError('Duplicate official IDs')
    if ids['train'] & ids['dev'] or ids['train'] & ids['test'] or ids['dev'] & ids['test']:raise ValueError('Official ID overlap')
    selected={}
    for split,n in (('train',64),('dev',32)):
        rng=random.Random(42)
        selected[split]=sorted([r for label in (0,1) for r in rng.sample(sorted([r for r in official[split] if r['label']==label],key=lambda r:str(r['image_id'])),n//2)],key=lambda r:str(r['image_id']))
    metadata={}; locations={}; mismatches=[]
    for split,pattern in (('train','train-*.parquet'),('dev','validation-*.parquet')):
        rows={}; location={}
        for path in sorted(mirror.glob(pattern)):
            file=pq.ParquetFile(path)
            for group in range(file.num_row_groups):
                for index,r in enumerate(file.read_row_group(group,columns=['id','text','label']).to_pylist()):
                    sid=str(r['id'])
                    if sid in rows:raise ValueError('Mirror duplicate ID')
                    rows[sid]=r;location[sid]=(path,group,index)
        if set(rows)!=ids[split]:raise ValueError('Mirror does not preserve the complete official split IDs')
        for r in official[split]:
            other=rows[str(r['image_id'])]
            if r['label']!=other['label'] or r['text']!=other['text']:
                mismatches.append(dict(split=split,id=str(r['image_id']),label_equal=r['label']==other['label'],text_equal=r['text']==other['text']))
        metadata[split]=rows;locations[split]=location
    root.mkdir(parents=True)
    output=[];image_hashes={}
    cache={}
    for split,rows in selected.items():
        for row in rows:
            sid=str(row['image_id']);path,group,index=locations[split][sid]
            key=(str(path),group)
            if key not in cache:
                cache.clear();cache[key]=pq.ParquetFile(path).read_row_group(group,columns=['image']).to_pylist()
            raw=cache[key][index]['image']['bytes']
            with Image.open(io.BytesIO(raw)) as im:
                im.verify()
            image=root/'images'/f'{sid}.jpg';image.parent.mkdir(exist_ok=True)
            image.write_bytes(raw) # no image re-encoding
            image_hashes[sid]=digest(image)
            output.append(dict(id=sid,split=split,image=str(image.resolve()),text=row['text'],labels=['sarcastic' if row['label']==1 else 'not_sarcastic']))
    source=root/'source.jsonl'
    write_jsonl(source,output)
    revision='MMSD2.0 authors 1627811f7a1e0acedce5b9df8b58919d86dd638e data/text_json_final; image bytes coderchen01/MMSD2.0 mmsd-v2@5b39e5b844accf2331d1738a3d91d4cb59005a50; fixed seed42 class-stratified 64 train/32 dev pilot only'
    prepare(source,root/'manifest','mmsd2',revision)
    report=dict(dataset='MMSD2.0',kind='real-data resource pilot subset, not full development screening',
        official_repo='https://github.com/JoeYing1019/MMSD2.0',official_revision='1627811f7a1e0acedce5b9df8b58919d86dd638e',
        official_annotation_sha256={s:digest(author/f'{s}.json') for s in ('train','valid','test')},
        official_counts={s:len(v) for s,v in official.items()},
        official_label_counts={s:dict(Counter(str(r['label']) for r in v)) for s,v in official.items()},
        official_split_ids_disjoint=True,mirror_complete_train_dev_id_match=True,
        mirror='https://huggingface.co/datasets/coderchen01/MMSD2.0',mirror_revision='5b39e5b844accf2331d1738a3d91d4cb59005a50',
        mirror_text_label_mismatches=mismatches,annotations_authoritative='authors corrected text_json_final, never mirror labels',
        sampling='seed42, sorted IDs, class-stratified, fixed before model evaluation',
        selected={s:[str(r['image_id']) for r in v] for s,v in selected.items()},
        selected_image_sha256=image_hashes,test_used=False)
    write_json(root/'provenance.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('selected_image_sha256','mirror_text_label_mismatches','selected')},indent=2))
    print('mirror annotation mismatches',len(mismatches))


if __name__=='__main__':main()
