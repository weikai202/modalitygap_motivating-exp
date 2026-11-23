"""Download the authors' recommended raw archive by verified resumable ranges."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import os
from pathlib import Path
import time

import requests

URL='https://dn790006.ca.archive.org/0/items/mmimdb/mmimdb.tar.gz'
SIZE=8693248518
MD5='38e154264b3771f70840c6cb5f170a93'
CHUNK=64*1024*1024
ROOT=Path('data/downloads')
PARTS=ROOT/'mmimdb-parts'


def fetch(start):
    end=min(SIZE,start+CHUNK)-1
    path=PARTS/f'{start:012d}.part'
    expected=end-start+1
    for attempt in range(5):
        got=path.stat().st_size if path.exists() else 0
        if got==expected:return start,expected
        if got>expected:raise ValueError('Oversized partial chunk')
        offset=start+got
        try:
            with requests.get(URL+f'?raw_chunk={start}',headers={'Range':f'bytes={offset}-{end}'},stream=True,timeout=(30,120)) as r:
                r.raise_for_status()
                if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {offset}-{end}/{SIZE}':
                    raise ValueError(f'Incorrect range: {r.status_code} {r.headers.get("Content-Range")}')
                with path.open('ab') as f:
                    for data in r.iter_content(4*1024*1024):
                        f.write(data)
            if path.stat().st_size==expected:return start,expected
        except Exception as e:
            print('Retry',start,attempt+1,str(e)[:180],flush=True)
            if attempt==4:raise
            time.sleep(min(2**attempt,16))
    raise ValueError('Incomplete range')


def main():
    ROOT.mkdir(parents=True,exist_ok=True);PARTS.mkdir(exist_ok=True)
    starts=list(range(0,SIZE,CHUNK))
    previous=ROOT/'mmimdb.tar.gz.part'
    if previous.exists():
        n=previous.stat().st_size
        with previous.open('rb') as f:
            for start in starts:
                size=min(CHUNK,SIZE-start)
                if start+size>n:break
                target=PARTS/f'{start:012d}.part'
                data=f.read(size)
                if not target.exists():target.write_bytes(data)
    total=0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(fetch,start):start for start in starts}
        for future in as_completed(futures):
            start,size=future.result();total+=size
            print(f'Raw archive ranges complete: {total}/{SIZE} ({100*total/SIZE:.1f}%)',flush=True)
    target=ROOT/'mmimdb.tar.gz'
    temp=ROOT/'mmimdb.tar.gz.assembling'
    md5=hashlib.md5()
    with temp.open('wb') as out:
        for start in starts:
            with (PARTS/f'{start:012d}.part').open('rb') as f:
                for data in iter(lambda:f.read(4*1024*1024),b''):
                    out.write(data);md5.update(data)
    if temp.stat().st_size!=SIZE or md5.hexdigest()!=MD5:raise ValueError('Official archive checksum mismatch')
    os.replace(temp,target)
    print('VERIFIED RAW ARCHIVE',str(target),md5.hexdigest(),flush=True)


if __name__=='__main__':main()
