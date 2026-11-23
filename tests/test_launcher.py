"""Full-study configuration, immutable input identities, and restart behavior."""
import contextlib,importlib.util,io,json
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
from modalitygap.protocol import CONFIGS,write_json

REPO=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('campaign_launcher',REPO/'scripts/run_experiment.py')
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def config(self,root):
        source=root/'input files/source.jsonl';source.parent.mkdir(parents=True);source.write_text('{}\n')
        path=root/'settings/experiment.json'
        write_json(path,dict(workspace='../output',cache_dir='../cache',datasets=dict(mmsd2=dict(source='../input files/source.jsonl',source_revision='fixture-author-revision')),
            configs=list(CONFIGS),seeds=[42,43,44],pool_size='all',overlap_policy='error',masking_method='symmetric-attention-v2',stats_samples=2,checkpoint_every_steps=2,keep_step_checkpoints=2))
        return path,launcher.load_config(path)

    def test_full_plan_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,cfg=self.config(Path(tmp));stream=io.StringIO()
            with contextlib.redirect_stdout(stream):launcher.preview(cfg,'all',False,None)
            self.assertEqual(stream.getvalue().count('modalitygap.cli run'),18)
            self.assertNotIn('--pool-size',stream.getvalue());self.assertFalse(cfg['workspace'].exists())

    def test_reject_subsets_and_unsupported_protocols(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,cfg=self.config(Path(tmp));raw=json.loads(path.read_text())
            for edit in [dict(seeds=[42]),dict(configs=['L-low']),dict(pool_size=6000),dict(masking_method='unsupported'),dict(unknown=True)]:
                write_json(path,dict(raw,**edit))
                with self.assertRaises(ValueError):launcher.load_config(path)

    def test_frozen_inputs_and_policy_are_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,cfg=self.config(Path(tmp));p=launcher.Pipeline(cfg)
            with launcher.workspace_lock(cfg['workspace']) as p.fd:p.initialize('prepare')
            with launcher.workspace_lock(cfg['workspace']) as fd:
                q=launcher.Pipeline(cfg,resume=True);q.fd=fd;q.initialize('prepare')
                with self.assertRaisesRegex(ValueError,'--resume'):p.initialize('prepare')
                cfg['datasets']['mmsd2']['source'].write_text('modified\n')
                with self.assertRaisesRegex(ValueError,'Frozen pipeline'):q.initialize('prepare')

    def test_workspace_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            with launcher.workspace_lock(Path(tmp)):
                with self.assertRaisesRegex(ValueError,'already running'):
                    with launcher.workspace_lock(Path(tmp)):pass

    def test_completed_job_is_skipped_and_interruption_uses_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,cfg=self.config(Path(tmp));p=launcher.Pipeline(cfg,resume=True)
            directory=cfg['workspace']/'runs/mmsd2/L-low/seed_42';(directory/'checkpoints/epoch_0').mkdir(parents=True)
            with patch.object(launcher,'verify_run_config'),patch.object(launcher,'complete_run',return_value=True),patch.object(p,'execute') as execute:
                self.assertTrue(p.job('mmsd2','L-low',42));execute.assert_not_called()
            write_json(directory/'progress.json',dict(status='incomplete',checkpoint='step_3'))
            with patch.object(launcher,'verify_run_config'),patch.object(launcher,'complete_run',return_value=False),patch.object(p,'execute') as execute:
                self.assertFalse(p.job('mmsd2','L-low',42));self.assertEqual(execute.call_args.args[0][-2:],['--resume-from','latest'])

    def test_cpu_analysis_does_not_load_model_or_access_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,cfg=self.config(Path(tmp));p=launcher.Pipeline(cfg)
            with launcher.workspace_lock(cfg['workspace']) as p.fd:p.initialize('prepare')
            cfg['datasets']['mmsd2']['source'].unlink()
            with patch.object(launcher.Pipeline,'gpu') as gpu,patch.object(launcher.Pipeline,'analyze') as analyze:
                launcher.Pipeline(cfg,resume=True).run('analyze')
                gpu.assert_not_called();analyze.assert_called_once()

    def test_analysis_rejects_an_incomplete_full_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,cfg=self.config(Path(tmp));p=launcher.Pipeline(cfg)
            first=cfg['workspace']/'runs/mmsd2/L-low/seed_42'
            with patch.object(launcher,'complete_run',side_effect=lambda path:path==first),patch.object(p,'execute') as execute:
                with self.assertRaisesRegex(ValueError,'Incomplete full matrix'):p.analyze()
                execute.assert_not_called()
