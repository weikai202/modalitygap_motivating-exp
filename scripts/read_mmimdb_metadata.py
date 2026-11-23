"""Read only official IDs, labels and split metadata via verified HTTP byte ranges."""
import io
import json
from pathlib import Path

import h5py
import requests

SIZE=15641130233
URL='https://archive.org/download/mmimdb/multimodal_imdb.hdf5'


class RangeFile(io.RawIOBase):
    def __init__(self,cache):
        self.position=0
        self.cache=Path(cache)
        self.cache.mkdir(parents=True,exist_ok=True)
        self.session=requests.Session()
        self.block=1024*1024
        self.requested=[]
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else SIZE+offset
        if self.position<0:raise ValueError('negative seek')
        return self.position
    def read(self,n=-1):
        if n<0:n=SIZE-self.position
        if n>4*1024*1024:raise ValueError('Refusing a large HDF read; only metadata is permitted')
        result=[]
        remaining=min(n,SIZE-self.position)
        while remaining>0:
            start=(self.position//self.block)*self.block
            end=min(SIZE-1,start+self.block-1)
            path=self.cache/f'{start}.bin'
            if not path.exists():
                print(f'HDF metadata range {start}-{end}',flush=True)
                response=self.session.get(URL+f'?metadata_range={start}',headers={'Range':f'bytes={start}-{end}'},timeout=90)
                response.raise_for_status()
                if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{SIZE}' or len(response.content)!=end-start+1:
                    raise ValueError('Server did not honor the exact byte range')
                path.write_bytes(response.content)
                self.requested.append([start,end])
            data=path.read_bytes()
            offset=self.position-start
            count=min(remaining,len(data)-offset)
            result.append(data[offset:offset+count]);self.position+=count;remaining-=count
        return b''.join(result)
    def readinto(self,b):
        data=self.read(len(b));b[:len(data)]=data;return len(data)


def main():
    root=Path('data/downloads')
    with RangeFile(root/'mmimdb-h5-blocks') as remote:
        with h5py.File(remote,'r') as h:
            print('ROOT KEYS',list(h.keys()),flush=True)
            ids=[x.decode() for x in h['imdb_ids'][:]]
            labels=h['genres'][:].tolist()
            genres=json.loads(h['genres'].attrs['target_names'])
            splits={}
            for row in h.attrs['split']:
                if row['source'].decode()=='imdb_ids':
                    splits[row['split'].decode()]=[int(row['start']),int(row['stop'])]
            output=dict(source=URL,source_size=SIZE,source_published_md5='644a126a5b1dd7871c2739098651b0a4',
                        ids=ids,labels=labels,genre_order=genres,split_bounds=splits,
                        usage='Only imdb_ids, genres and split attrs read; no precomputed feature or resized image input')
            (root/'mmimdb-official-metadata.json').write_text(json.dumps(output,indent=2)+'\n')
            print('OFFICIAL MM-IMDB',len(ids),genres,splits,flush=True)


if __name__=='__main__':main()
