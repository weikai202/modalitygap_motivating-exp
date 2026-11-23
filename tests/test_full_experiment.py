"""Full-release portability, fixed overlap policy, matrix planning and retention."""
import contextlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from PIL import Image
from modalitygap.full_data import mmsd2,mmimdb
from modalitygap.data import prepare
from modalitygap.protocol import read_jsonl,write_json,write_jsonl
from modalitygap.checkpoint import prune_steps
import test_launcher


class FullExperimentTests(unittest.TestCase):
    def test_mmsd_exports_all_official_splits_author_labels_and_portable_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ann=root/'annotations';ann.mkdir();images=root/'original';images.mkdir()
            for i,(name,label) in enumerate([('train',0),('valid',1),('test',0)]):
                write_json(ann/f'{name}.json',[dict(image_id=100+i,text=f'Corrected text {i}',label=label)])
                Image.new('RGB',(8,8),(i*30,20,50)).save(images/f'{100+i}.jpg')
            out=mmsd2(ann,root/'export',images=images,revision='fixture')
            rows=read_jsonl(out/'source.jsonl');self.assertEqual([r['split'] for r in rows],['train','dev','test'])
            self.assertEqual(rows[1]['labels'],['sarcastic'])
            for r in rows:
                self.assertFalse(Path(r['image']).is_absolute())
                self.assertEqual((out/r['image']).read_bytes(),(images/Path(r['image']).name).read_bytes())
            with self.assertRaises(ValueError):mmsd2(ann,out,images=images)

    def test_parquet_mirror_never_overrides_corrected_author_annotations(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ann=root/'annotations';ann.mkdir();mirror=root/'mirror';mirror.mkdir()
            raw=io.BytesIO();Image.new('RGB',(8,8),'blue').save(raw,format='PNG')
            for name,alias,sid in [('train','train','100'),('valid','validation','200'),('test','test','300')]:
                write_json(ann/f'{name}.json',[dict(image_id=int(sid),text='author text '+sid,label=1)])
                table=pa.Table.from_pylist([dict(id=sid,text='incorrect mirror text',label=0,image=dict(bytes=raw.getvalue(),path=None))])
                pq.write_table(table,mirror/f'{alias}-00000-of-00001.parquet')
            out=mmsd2(ann,root/'export',parquet=mirror)
            rows=read_jsonl(out/'source.jsonl')
            self.assertTrue(all(r['labels']==['sarcastic'] and r['text']=='author text '+r['id'] for r in rows))
            self.assertEqual(len(json.loads((out/'provenance.json').read_text())['mirror_annotation_mismatches']),3)
            self.assertTrue(all((out/r['image']).read_bytes()==raw.getvalue() for r in rows))

    def test_mmimdb_raw_split_first_nonempty_plot_and_checksum_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/'raw.tar.gz'
            with tarfile.open(archive,'w:gz') as t:
                files={'raw/split.json':json.dumps(dict(train=['01'],dev=['02'],test=['03'])).encode()}
                for sid in ('01','02','03'):
                    files[f'raw/dataset/{sid}.json']=json.dumps(dict(genres=['Drama','Comedy','News'],plot=['','Plot '+sid,'ignored second plot'],title='Must not enter input')).encode()
                    im=io.BytesIO();Image.new('RGB',(8,8),'red').save(im,format='JPEG');files[f'raw/dataset/{sid}.jpeg']=im.getvalue()
                for name,content in files.items():
                    info=tarfile.TarInfo(name);info.size=len(content);t.addfile(info,io.BytesIO(content))
            with self.assertRaisesRegex(ValueError,'checksum'):mmimdb(archive,root/'bad')
            out=mmimdb(archive,root/'export',verify_md5=False)
            rows=read_jsonl(out/'source.jsonl');self.assertEqual(len(rows),3)
            self.assertEqual(rows[0]['text'],'Plot 01');self.assertEqual(rows[0]['labels'],['Comedy','Drama'])
            self.assertTrue(all((out/r['image']).exists() for r in rows))

    def test_overlap_component_resolution_never_moves_heldout_or_leaks_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[]
            for i,split in enumerate(['train','train','dev','dev','test','train']):
                image=root/f'{i}.png';Image.new('RGB',(8,8),(i*20,30,40)).save(image)
                rows.append(dict(id=str(i),split=split,image=str(image),text=f'body {i}',labels=['sarcastic']))
            # Transitive: train0 -- same text -- dev2 -- same RGB -- test4.
            rows[2]['text']=rows[0]['text'];rows[4]['image']=rows[2]['image']
            write_jsonl(root/'source.jsonl',rows)
            with self.assertRaisesRegex(ValueError,'Cross-split'):prepare(root/'source.jsonl',root/'strict','mmsd2','fixture')
            prepare(root/'source.jsonl',root/'clean','mmsd2','fixture',pool_size='all',overlap_policy='prefer-heldout')
            self.assertEqual([r['id'] for r in read_jsonl(root/'clean/train.jsonl')],['1','5'])
            self.assertEqual([r['id'] for r in read_jsonl(root/'clean/dev.jsonl')],['3'])
            self.assertEqual([r['id'] for r in read_jsonl(root/'clean/test.jsonl')],['4'])
            meta=json.loads((root/'clean/manifest.json').read_text());self.assertEqual(len(meta['overlap_excluded']),2)
            self.assertEqual(meta['original_complete_counts']['train'],3)

    def test_full_matrix_confirmation_and_frozen_pool_policy(self):
        launcher=test_launcher.launcher
        with tempfile.TemporaryDirectory() as tmp:
            path,_=test_launcher.LauncherTests().config(Path(tmp));raw=json.loads(path.read_text())
            raw.update(seeds=[42,43,44],masking_method='symmetric-attention-v2',pool_size='all',overlap_policy='prefer-heldout',keep_step_checkpoints=2)
            raw['datasets']['mmimdb']=dict(raw['datasets']['mmsd2']);write_json(path,raw);cfg=launcher.load_config(path)
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream):launcher.preview(cfg,'all',False,None)
            text=stream.getvalue();self.assertEqual(text.count('modalitygap.cli run'),36) # 36 full training jobs
            self.assertNotIn('--pool-size',text);self.assertIn('--overlap-policy prefer-heldout',text)
            self.assertIn('--keep-step-checkpoints 2',text)
            p=launcher.Pipeline(cfg)
            with launcher.workspace_lock(cfg['workspace']) as p.fd:p.initialize('prepare')
            raw['overlap_policy']='error';write_json(path,raw);changed=launcher.load_config(path)
            with launcher.workspace_lock(cfg['workspace']) as fd:
                q=launcher.Pipeline(changed,resume=True);q.fd=fd
                with self.assertRaisesRegex(ValueError,'Frozen pipeline'):q.initialize('prepare')

    def test_step_retention_preserves_epoch_and_unpublished_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for step in (2,10,9):write_json(root/f'step_{step}/complete.json',dict(step=step))
            write_json(root/'epoch_2/complete.json',dict(step=10));(root/'step_11').mkdir()
            prune_steps(root,2)
            self.assertFalse((root/'step_2').exists())
            self.assertTrue(all((root/name).exists() for name in ('step_9','step_10','step_11','epoch_2')))
            prune_steps(root,0);self.assertTrue((root/'step_9').exists())
