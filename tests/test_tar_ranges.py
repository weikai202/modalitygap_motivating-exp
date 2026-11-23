"""TAR range traversal must handle boundaries inside large unselected payloads."""
import importlib.util
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch


class TarRangeTests(unittest.TestCase):
    def test_large_member_crossing_segment_and_headerless_continuation(self):
        scripts=Path(__file__).resolve().parents[1]/'scripts'
        sys.path.insert(0,str(scripts))
        try:
            spec=importlib.util.spec_from_file_location('tar_range_downloader',scripts/'fetch_visualnews_images.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        finally:sys.path.pop(0)
        archive=io.BytesIO()
        with tarfile.open(fileobj=archive,mode='w') as t:
            member=tarfile.TarInfo('origin/data.json');member.size=40*1024**2
            t.addfile(member,io.BytesIO(b'x'*member.size))
        content=archive.getvalue()
        class Response:
            status_code=206
            def __init__(self,start,end):
                self.headers={'ETag':module.ETAG,'Content-Range':f'bytes {start}-{end}/{len(content)}'}
                self.raw=io.BytesIO(content[start:end+1])
            def raise_for_status(self):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
        def get(url,headers,**kwargs):
            start,end=map(int,headers['Range'].removeprefix('bytes=').split('-'))
            return Response(start,end)
        with tempfile.TemporaryDirectory() as tmp,patch.object(module,'DOWNLOADS',Path(tmp)),patch.object(module,'SIZE',len(content)),patch.object(module,'SEGMENT',1024),patch.object(module.requests,'get',side_effect=get):
            first=module.worker(0,{})
            self.assertTrue(first['complete'])
            self.assertGreater(first['next_offset'],32*1024**2)
            second=module.worker(1,{})
            self.assertTrue(second['complete'])
            self.assertEqual(second['next_offset'],2048)
            # A saved successor beyond the small segment is already complete.
            import json
            state=Path(tmp)/'visualnews-segment-00.json';data=json.loads(state.read_text());data['complete']=False;state.write_text(json.dumps(data))
            self.assertTrue(module.worker(0,{})['complete'])
