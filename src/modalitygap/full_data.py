"""Portable full-release exporters: official labels/splits, original image bytes."""
import argparse
from collections import Counter
import hashlib
import json
import os
import shutil
from pathlib import Path
import tarfile

from .protocol import GENRES,digest,write_json,write_jsonl

MMSD_REVISION='1627811f7a1e0acedce5b9df8b58919d86dd638e'
MMSD_IMAGE_REVISION='5b39e5b844accf2331d1738a3d91d4cb59005a50'
MMIMDB_MD5='38e154264b3771f70840c6cb5f170a93'


def fresh(output):
    output=Path(output).resolve()
    if output.exists():raise ValueError('Export directory already exists; choose a new immutable version')
    output.mkdir(parents=True);return output


def relative(path,output):
    return os.path.relpath(Path(path).resolve(),output)


def finish(output,dataset,rows,provenance):
    write_jsonl(output/'source.jsonl',rows)
    write_json(output/'provenance.json',dict(dataset=dataset,**provenance,
        counts=dict(Counter(r['split'] for r in rows)),source_sha256=digest(output/'source.jsonl'),
        image_paths='relative to source.jsonl; move the export together with images',
        sampling='No sampling. Official train/dev/test exported; training pool is selected later by campaign prepare.'))


def mmsd2(annotations,output,images=None,parquet=None,revision=MMSD_REVISION):
    annotations=Path(annotations);official={s:json.loads((annotations/f'{f}.json').read_text()) for s,f in [('train','train'),('dev','valid'),('test','test')]}
    lookup={};splits={}
    for split,rows in official.items():
        for row in rows:
            sid=str(row['image_id'])
            if sid in lookup:raise ValueError(f'Duplicate official ID / split overlap: {sid}')
            if row['label'] not in (0,1) or not isinstance(row['text'],str):raise ValueError('Invalid official annotation')
            lookup[sid]=row;splits[sid]=split
    if bool(images)==bool(parquet):raise ValueError('Provide exactly one of images or parquet')
    output=fresh(output);locations={};mismatches=[]
    if images:
        if output.is_relative_to(Path(images).resolve()):raise ValueError('Keep export outside original image directory')
        (output/'images').mkdir()
        for p in Path(images).rglob('*'):
            if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png') and p.stem in lookup:
                if p.stem in locations:raise ValueError(f'Ambiguous original image ID: {p.stem}')
                dest=output/'images'/p.name;shutil.copyfile(p,dest)
                locations[p.stem]=relative(dest,output)
    else:
        import pyarrow.parquet as pq
        target=output/'images';target.mkdir();seen=set()
        aliases={'train':'train','validation':'dev','test':'test'}
        for alias,split in aliases.items():
            files=sorted(Path(parquet).glob(f'{alias}-*.parquet'))
            if not files:raise ValueError(f'Missing full {split} parquet files')
            for file in files:
                reader=pq.ParquetFile(file)
                for batch in reader.iter_batches(batch_size=32):
                    for row in batch.to_pylist():
                        sid=str(row['id'])
                        if sid not in lookup or splits[sid]!=split or sid in seen:raise ValueError(f'Mirror ID/split mismatch: {sid}')
                        seen.add(sid);author=lookup[sid]
                        if row['text']!=author['text'] or row['label']!=author['label']:mismatches.append(sid)
                        raw=row['image'].get('bytes')
                        if raw:
                            path=target/f'{sid}.jpg';path.write_bytes(raw);locations[sid]=relative(path,output)
        if seen!=set(lookup):raise ValueError('Mirror split ID coverage differs from complete author release')
    rows=[]
    for sid,row in lookup.items():
        rows.append(dict(id=sid,split=splits[sid],image=locations.get(sid,f'images/{sid}.jpg'),text=row['text'],labels=['sarcastic' if row['label']==1 else 'not_sarcastic']))
    finish(output,'mmsd2',rows,dict(source_revision=f'MMSD2.0 corrected text_json_final@{revision}; original images '+('local author image files' if images else f'coderchen01/MMSD2.0 mmsd-v2@{MMSD_IMAGE_REVISION}'),
        annotations_sha256={s:digest(annotations/f'{f}.json') for s,f in [('train','train'),('dev','valid'),('test','test')]},
        missing_images=sorted(set(lookup)-set(locations)),mirror_annotation_mismatches=mismatches,author_text_and_labels_always_used=True))
    return output


def mmimdb(archive,output,verify_md5=True):
    archive=Path(archive)
    with archive.open('rb') as handle:
        checksum=hashlib.file_digest(handle,'md5').hexdigest()
    if verify_md5 and checksum!=MMIMDB_MD5:raise ValueError('MM-IMDb official archive checksum mismatch')
    output=fresh(output);(output/'images').mkdir();(output/'raw-json').mkdir()
    split=None;found={};metadata={}
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            if not member.isfile():continue
            path=Path(member.name)
            if path.name=='split.json':
                if split is not None:raise ValueError('Duplicate split.json')
                split=json.load(tar.extractfile(member));continue
            if path.suffix not in ('.json','.jpeg'):continue
            sid=path.stem
            if not sid.isdigit():continue
            key=(sid,path.suffix)
            if key in found:raise ValueError(f'Duplicate archive member: {key}')
            content=tar.extractfile(member).read()
            dest=output/('images' if path.suffix=='.jpeg' else 'raw-json')/path.name
            dest.write_bytes(content);found[key]=dest
            if path.suffix=='.json':metadata[sid]=json.loads(content)
    return finish_mmimdb(output,split,metadata,checksum,verify_md5)


def finish_mmimdb(output,split,metadata,checksum,verify_md5):
    if (output/'source.jsonl').exists():raise ValueError('Full export already published')
    if split is None:raise ValueError('Missing author split.json')
    assigned={};rows=[];empty=[];projected=[]
    for key,ids in split.items():
        name='dev' if key in ('val','valid','validation') else key
        if name not in ('train','dev','test'):raise ValueError('Unknown raw split')
        for value in ids:
            sid=str(value)
            if sid in assigned:raise ValueError(f'Official ID overlap: {sid}')
            assigned[sid]=name
            if sid not in metadata:raise ValueError(f'Missing raw JSON: {sid}')
            movie=metadata[sid];genres=movie['genres']
            extra=sorted(set(genres)-set(GENRES))
            if extra:projected.append(dict(id=sid,omitted_raw_tags=extra))
            if not set(genres)&set(GENRES):raise ValueError(f'No supported task genre: {sid}')
            plots=movie.get('plot',[]);plots=[plots] if isinstance(plots,str) else plots
            text=next((p for p in plots if isinstance(p,str) and p.strip()),'')
            if not text:empty.append(sid)
            rows.append(dict(id=sid,split=name,image=f'images/{sid}.jpeg',
                text=text,labels=[g for g in GENRES if g in genres]))
    finish(output,'mmimdb',rows,dict(source_revision=f'MM-IMDb original raw archive MD5 {checksum}; official split.json; first nonempty released plot',
        archive_md5=checksum,official_checksum_verified=verify_md5,official_split=split,empty_plot_ids=empty,
        genre_rule='Project original genres onto fixed official 23-genre task vocabulary; audit omitted raw tags',raw_tag_projection=projected,
        plot_rule='First nonempty released plot; no title, year, ID or filename appended.'))
    return output


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='dataset',required=True)
    m=sub.add_parser('mmsd2');m.add_argument('--annotations',required=True);g=m.add_mutually_exclusive_group(required=True);g.add_argument('--images');g.add_argument('--parquet');m.add_argument('--revision',default=MMSD_REVISION);m.add_argument('--output',required=True)
    m=sub.add_parser('mmimdb');m.add_argument('--archive',required=True);m.add_argument('--output',required=True)
    a=p.parse_args()
    if a.dataset=='mmsd2':out=mmsd2(a.annotations,a.output,a.images,a.parquet,a.revision)
    else:out=mmimdb(a.archive,a.output)
    print(out/'source.jsonl')


if __name__=='__main__':main()
