"""Launch planning, safe reuse, ownership and lock behavior without model downloads."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from modalitygap.protocol import CONFIGS, write_json, digest

REPO=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('campaign_launcher',REPO/'scripts/run_experiment.py')
launcher=importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def config(self,root):
        source=root/'input files'/'source.jsonl'
        source.parent.mkdir(parents=True,exist_ok=True)
        source.write_text('{}\n')
        path=root/'settings'/'experiment.json'
        write_json(path,dict(workspace='../output space',cache_dir='../cache',
            datasets=dict(mmsd2=dict(source='../input files/source.jsonl',source_revision='fixture-author-revision')),
            configs=list(CONFIGS),seeds=[42],stats_samples=2,pilot_steps=4,checkpoint_every_steps=2))
        return path,launcher.load_config(path)

    def test_paths_and_dry_run_full_matrix_have_no_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            path,cfg=self.config(root)
            self.assertEqual(cfg['workspace'],root/'output space')
            self.assertEqual(cfg['datasets']['mmsd2']['source'],root/'input files/source.jsonl')
            output=io.StringIO()
            with contextlib.redirect_stdout(output):
                launcher.preview(cfg,'all',True,None)
            text=output.getvalue()
            self.assertEqual(text.count('modalitygap.cli run'),9) # 3 pilots + 6 screening
            self.assertIn('--bootstrap-samples 2000',text)
            self.assertFalse(cfg['workspace'].exists())
            result=subprocess.run(['bash',str(REPO/'scripts/run_experiment.sh'),'--setup','--config',str(path),'--dry-run'],
                                  cwd=root,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('DRY RUN',result.stdout)
            self.assertFalse(cfg['workspace'].exists())

    def test_config_rejects_placeholders_bad_types_and_source_inside_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,cfg=self.config(Path(tmp))
            original=launcher.read_json(path)
            for edit in (dict(seeds=[42,42]),dict(configs=[{}]),dict(pilot_steps=2),dict(unknown=True)):
                write_json(path,dict(original,**edit))
                with self.assertRaises(ValueError):
                    launcher.load_config(path)
            bad=json.loads(json.dumps(original))
            bad['datasets']['mmsd2']['source_revision']='REPLACE_WITH_RELEASE'
            write_json(path,bad)
            with self.assertRaisesRegex(ValueError,'source_revision'):
                launcher.load_config(path)
            bad['datasets']['mmsd2']['source_revision']='actual release'
            bad['datasets']['mmsd2']['source']='../output space/input.jsonl'
            write_json(path,bad)
            with self.assertRaisesRegex(ValueError,'outside'):
                launcher.load_config(path)

    def test_identity_allows_confirmation_seeds_but_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,cfg=self.config(Path(tmp))
            p=launcher.Pipeline(cfg)
            with launcher.workspace_lock(cfg['workspace']) as p.fd:
                p.initialize('prepare')
            raw=launcher.read_json(path);raw['seeds']=[42,43,44];write_json(path,raw)
            updated=launcher.load_config(path)
            with launcher.workspace_lock(cfg['workspace']) as fd:
                p=launcher.Pipeline(updated,resume=True);p.fd=fd;p.initialize('prepare')
                with self.assertRaisesRegex(ValueError,'--resume'):
                    launcher.Pipeline(updated).initialize('prepare')
                updated['datasets']['mmsd2']['source'].write_text('changed\n')
                with self.assertRaisesRegex(ValueError,'Frozen pipeline'):
                    p.initialize('prepare')

    def test_workspace_lock_and_child_inheritance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            child=None
            try:
                with launcher.workspace_lock(root) as fd:
                    with self.assertRaisesRegex(ValueError,'already running'):
                        with launcher.workspace_lock(root):
                            pass
                    code="import os,sys; os.fstat(int(sys.argv[1])); print('ready',flush=True); sys.stdin.read(1)"
                    child=subprocess.Popen([sys.executable,'-c',code,str(fd)],pass_fds=(fd,),
                        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                    self.assertEqual(child.stdout.readline().strip(),'ready')
                # The launcher FD is now closed; the live trainer still holds the lock.
                with self.assertRaisesRegex(ValueError,'already running'):
                    with launcher.workspace_lock(root):
                        pass
            finally:
                if child is not None:
                    child.communicate('x',timeout=10)
            self.assertEqual(child.returncode,0)
            with launcher.workspace_lock(root):
                pass

    def test_incomplete_preparation_is_preserved_but_frozen_data_is_not_replaced(self):
        from modalitygap.data import prepare
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            path,cfg=self.config(root)
            source=cfg['datasets']['mmsd2']['source']
            rows=[]
            for i,split in enumerate(('train','dev','test')):
                image=root/f'{i}.png';Image.new('RGB',(8,8),(i*40,20,30)).save(image)
                rows.append(dict(id=str(i),split=split,image=str(image),text=f'Unique text {i}',labels=['sarcastic']))
            source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            folder=cfg['workspace']/'manifests/mmsd2'
            folder.mkdir(parents=True)
            (folder/'manifest.json').write_text('{')
            p=launcher.Pipeline(cfg,resume=True)
            def execute(command,label,offline=False):
                prepare(source,folder,'mmsd2','fixture-author-revision')
            with patch.object(p,'execute',side_effect=execute):
                p.prepare()
                self.assertEqual(len(list((cfg['workspace']/'recovery').iterdir())),1)
                self.assertEqual(launcher.manifest_valid(cfg,'mmsd2')['counts']['train'],1)
                (folder/'train.jsonl').write_text('modified')
                with self.assertRaisesRegex(ValueError,'modified train'):
                    p.prepare()

    def test_completed_jobs_skip_and_incomplete_jobs_request_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,cfg=self.config(Path(tmp))
            p=launcher.Pipeline(cfg,resume=True)
            directory=cfg['workspace']/'runs/mmsd2/L-low/seed_42'
            (directory/'checkpoints'/'epoch_0').mkdir(parents=True)
            with patch.object(launcher,'verify_run_config'),patch.object(launcher,'complete_run',return_value=True),patch.object(p,'execute') as execute:
                self.assertTrue(p.job('mmsd2','L-low',42))
                execute.assert_not_called()
            write_json(directory/'progress.json',dict(status='incomplete',checkpoint='checkpoints/step_3'))
            with patch.object(launcher,'verify_run_config'),patch.object(launcher,'complete_run',return_value=False),patch.object(p,'execute') as execute:
                self.assertFalse(p.job('mmsd2','L-low',42))
                command=execute.call_args.args[0]
                self.assertEqual(command[-2:],['--resume-from','latest'])
                self.assertEqual(launcher.read_json(cfg['workspace']/'pipeline-status.json')['status'],'incomplete')

    def test_crash_before_first_checkpoint_restarts_without_losing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,cfg=self.config(Path(tmp))
            p=launcher.Pipeline(cfg,resume=True)
            directory=cfg['workspace']/'runs/mmsd2/L-low/seed_42'
            directory.mkdir(parents=True)
            (directory/'config.json').write_text('{')
            with patch.object(launcher,'complete_run',return_value=True),patch.object(p,'execute') as execute:
                self.assertTrue(p.job('mmsd2','L-low',42))
                self.assertNotIn('--resume-from',execute.call_args.args[0])
                saved=list((cfg['workspace']/'recovery').iterdir())
                self.assertEqual(len(saved),1)
                self.assertEqual((saved[0]/'config.json').read_text(),'{')

    def test_offline_cache_and_cpu_analysis_do_not_download_or_require_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,cfg=self.config(Path(tmp))
            p=launcher.Pipeline(cfg,resume=True)
            with patch.dict(os.environ,{'HF_HUB_OFFLINE':'1'}):
                with self.assertRaisesRegex(ValueError,'incomplete'):
                    p.model()
            folder=cfg['cache_dir']/('models--'+launcher.DEFAULTS['model'].replace('/','--'))/'snapshots'/cfg['model_revision']
            folder.mkdir(parents=True)
            for name in ('config.json','generation_config.json','preprocessor_config.json','tokenizer_config.json','vocab.json','merges.txt','tokenizer.json','model.safetensors'):
                (folder/name).write_text('{}')
            with patch.object(p,'execute') as execute:
                p.model();execute.assert_not_called()
            (folder/'merges.txt').unlink()
            with patch.dict(os.environ,{'HF_HUB_OFFLINE':'1'}):
                with self.assertRaisesRegex(ValueError,'incomplete'):
                    p.model()
            cfg['workspace'].mkdir()
            with launcher.workspace_lock(cfg['workspace']) as fd:
                p.fd=fd;p.initialize('prepare')
            cfg['datasets']['mmsd2']['source'].unlink()
            with patch.object(p,'gpu') as gpu,patch.object(p,'analyze') as analyze:
                p.run('analyze')
                gpu.assert_not_called();analyze.assert_called_once()
