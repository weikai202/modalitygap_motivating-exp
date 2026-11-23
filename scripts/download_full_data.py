"""Acquire pinned release inputs; model weights and private credentials are never bundled."""
import argparse,hashlib,json
from pathlib import Path
import requests
from huggingface_hub import snapshot_download
from modalitygap.full_data import MMSD_REVISION,MMSD_IMAGE_REVISION,MMIMDB_MD5
from modalitygap.protocol import write_json
import download_mmimdb_raw


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='data/full-inputs');p.add_argument('--dataset',choices=['mmsd2','mmimdb','both'],default='both');a=p.parse_args()
    root=Path(a.root).resolve();root.mkdir(parents=True,exist_ok=True)
    result=json.loads((root/'inputs.json').read_text()) if (root/'inputs.json').exists() else {}
    if a.dataset in ('mmsd2','both'):
        folder=root/'mmsd2-annotations';folder.mkdir(exist_ok=True)
        for split in ('train','valid','test'):
            target=folder/f'{split}.json'
            url=f'https://raw.githubusercontent.com/JoeYing1019/MMSD2.0/{MMSD_REVISION}/data/text_json_final/{split}.json'
            response=requests.get(url,timeout=90);response.raise_for_status();response.json()
            if target.exists() and target.read_bytes()!=response.content:raise ValueError('Existing author annotations differ')
            target.write_bytes(response.content)
        snap=snapshot_download('coderchen01/MMSD2.0',repo_type='dataset',revision=MMSD_IMAGE_REVISION,allow_patterns=['mmsd-v2/*.parquet'],cache_dir=root/'hf-cache')
        result['mmsd2']=dict(annotations=str(folder),parquet=str(Path(snap)/'mmsd-v2'),author_revision=MMSD_REVISION,image_revision=MMSD_IMAGE_REVISION)
    if a.dataset in ('mmimdb','both'):
        archive=root/'mmimdb.tar.gz'
        valid=False
        if archive.exists():
            with archive.open('rb') as handle:valid=hashlib.file_digest(handle,'md5').hexdigest()==MMIMDB_MD5
        if archive.exists() and not valid:raise ValueError('Existing MM-IMDb archive checksum differs; preserve it and choose a new root')
        if not valid:
            download_mmimdb_raw.ROOT=root;download_mmimdb_raw.PARTS=root/'mmimdb-parts';download_mmimdb_raw.main()
        result['mmimdb']=dict(archive=str(archive),verified_md5=MMIMDB_MD5)
    write_json(root/'inputs.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
