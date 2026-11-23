"""Stream author TAR ranges, save only preselected original images plus a reusable index.

No full 98 GB archive is stored. Segment checkpoints resume at verified TAR headers.
"""
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import tarfile
import time
import requests

from prepare_expansion import OUTPUT, DOWNLOADS, save, sha

URL='https://www.cs.rice.edu/~vo9/visualnews/origin.tar'
SIZE=97916641280
ETAG='"16cc496000-5dcdf02f54740"'
SEGMENT=4*1024**3
# Keep completed large-segment prefixes, then scan only unvisited ranges.
JOBS_FILE=DOWNLOADS/'visualnews-range-jobs-v2.json'
if JOBS_FILE.exists():
    JOBS=json.loads(JOBS_FILE.read_text())
else:
    JOBS=[]
    for i in range((SIZE+SEGMENT-1)//SEGMENT):
        state_file=DOWNLOADS/f'visualnews-segment-{i:02d}.json'
        state=json.loads(state_file.read_text()) if state_file.exists() else {}
        if state.get('complete'):continue
        start=state.get('next_offset',i*SEGMENT);stop=min((i+1)*SEGMENT,SIZE)
        while start<stop:
            end=min(start+1024**3,stop);JOBS.append([start,end]);start=end
    save(JOBS_FILE,JOBS)



class Prepend:
    def __init__(self, prefix, raw): self.prefix=io.BytesIO(prefix); self.raw=raw
    def read(self,n=-1):
        if n<0: raise ValueError('Bounded reads only')
        b=self.prefix.read(n)
        if len(b)<n:b+=self.raw.read(n-len(b))
        return b


def worker(index, wanted):
    start,stop=JOBS[index]
    state_file=DOWNLOADS/f'visualnews-v2-segment-{index:02d}.json'
    state=json.loads(state_file.read_text()) if state_file.exists() else {}
    if state.get('complete'):return state
    cursor=state.get('next_offset',start);found=state.get('found',[])
    if cursor>=stop:
        state=dict(next_offset=cursor,found=found,complete=True)
        save(state_file,state);return state
    index_file=DOWNLOADS/f'visualnews-v2-index-{index:02d}.tsv'
    for attempt in range(6):
        try:
            with requests.get(URL,headers={'Range':f'bytes={cursor}-{min(stop+32*1024**2,SIZE)-1}',
                  'If-Range':ETAG,'Accept-Encoding':'identity'},stream=True,timeout=(30,120)) as r:
                r.raise_for_status()
                if r.status_code!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {cursor}-'):
                    raise RuntimeError(f'Invalid range response {r.status_code}: {r.headers.get("Content-Range")}')
                if r.headers.get('ETag')!=ETAG:raise RuntimeError('Author archive changed')
                # A boundary can fall deep inside a large JSON/JPEG payload.
                # Scan aligned blocks until a verified header or this segment's end.
                header=None;scanned=0;prefix=b''
                while cursor+scanned<stop:
                    prefix=r.raw.read(min(4*1024**2,stop-cursor-scanned))
                    if not prefix:break
                    for offset in range(0,len(prefix)-511,512):
                        try:
                            member=tarfile.TarInfo.frombuf(prefix[offset:offset+512],encoding='utf-8',errors='strict')
                            if member.name=='origin' or member.name.startswith('origin/'):
                                header=offset;break
                        except (tarfile.TarError,UnicodeError,ValueError):pass
                    if header is not None:break
                    scanned+=len(prefix)
                if header is None:
                    state=dict(next_offset=stop,found=found,complete=True)
                    save(state_file,state);return state
                base=cursor+scanned+header; raw=Prepend(prefix[header:],r.raw); last=time.monotonic()
                with tarfile.open(fileobj=raw,mode='r|',bufsize=1024*1024) as archive,index_file.open('a') as idx:
                    for m in archive:
                        absolute=base+m.offset
                        if absolute>=stop:break
                        # Each index offset points directly at original, unmodified file payload.
                        if m.isfile():idx.write(f'{m.name}\t{base+m.offset_data}\t{m.size}\n')
                        key=m.name.removeprefix('origin/')
                        if key in wanted and m.isfile():
                            path=Path(wanted[key]);path.parent.mkdir(parents=True,exist_ok=True)
                            content=archive.extractfile(m).read()
                            if len(content)!=m.size:raise IOError('Truncated selected image')
                            if path.exists() and path.read_bytes()!=content:raise ValueError('Existing original image differs')
                            path.write_bytes(content)
                            found.append(dict(path=key,offset=base+m.offset_data,size=m.size,sha256=sha(path)))
                        # Let the stream consume this member before recording its successor offset.
                        cursor=base+m.offset_data+((m.size+511)//512)*512
                        if cursor>=stop:break
                        if time.monotonic()-last>30:
                            save(state_file,dict(next_offset=cursor,found=found,complete=False))
                            print(f'segment {index}: {(cursor-start)/1024**3:.2f} GiB; selected={len(found)}',flush=True)
                            last=time.monotonic()
                state=dict(next_offset=cursor,found=found,complete=True)
                save(state_file,state);print('segment',index,'complete',len(found),flush=True);return state
        except (requests.RequestException,OSError,tarfile.TarError) as e:
            save(state_file,dict(next_offset=cursor,found=found,complete=False,error=str(e)))
            print('retry',index,attempt,type(e).__name__,str(e)[:100],flush=True)
            if attempt==5:raise
            time.sleep(2)
    raise AssertionError('Unreachable')


if __name__=='__main__':
    plan=json.loads((OUTPUT/'newsclippings/selection.json').read_text())
    wanted={r['original_image_path']:r['image'] for r in plan['rows']}
    with ThreadPoolExecutor(64) as pool:
        states=list(pool.map(lambda i:worker(i,wanted),range(len(JOBS))))
    missing=[key for key,path in wanted.items() if not Path(path).exists()]
    save(DOWNLOADS/'visualnews-extraction.json',dict(url=URL,etag=ETAG,size=SIZE,
         selected_image_count=len(wanted),missing=missing,segments=states))
    if missing:raise RuntimeError(f'Missing original images: {missing}')
    print('ALL ORIGINAL SELECTED IMAGES EXTRACTED',len(wanted),flush=True)
