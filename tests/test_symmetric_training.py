"""Production runner/launcher protocol routing and exact symmetric continuation."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from modalitygap import runner
from modalitygap.checkpoint import read,capture_rng
from modalitygap.protocol import write_json
import test_launcher


class SymmetricLauncherTests(unittest.TestCase):
    def test_config_command_identity_and_no_synthetic_noise_variants(self):
        launcher=test_launcher.launcher
        with tempfile.TemporaryDirectory() as tmp:
            path,cfg=test_launcher.LauncherTests().config(Path(tmp))
            raw=json.loads(path.read_text());raw.update(masking_method='symmetric-attention-v2',configs=['L-low'])
            write_json(path,raw);cfg=launcher.load_config(path)
            command=launcher.train_command(cfg,'mmsd2','L-low',42)
            self.assertEqual(command[command.index('--masking-method')+1],'symmetric-attention-v2')
            self.assertEqual(launcher.pilot_configs(cfg),['L-low'])
            raw['masking_method']='unknown';write_json(path,raw)
            with self.assertRaises(ValueError):launcher.load_config(path)


@unittest.skipUnless(os.environ.get('MODALITYGAP_GPU_TESTS')=='1','Enable CUDA integration tests')
class SymmetricRunnerTests(unittest.TestCase):
    def test_all_tasks_full_horizon_exact_resume_and_protocol_rejection(self):
        import torch
        from test_resume import loader,StochasticEncoder,fixtures
        from test_checkpoint import assert_tree_equal
        with tempfile.TemporaryDirectory() as tmp,patch.object(runner,'load_model',side_effect=loader),patch.object(runner,'Encoder',StochasticEncoder):
            root=Path(tmp)
            for dataset in ('mmsd2','mmimdb','newsclippings','crisismmd','hatefulmemes'):
                common=fixtures(root,dataset)
                args=dict(common,masking_method='symmetric-attention-v2',keep_step_checkpoints=2,config='L-low',output=str(root/'baseline'))
                baseline=runner.run(SimpleNamespace(**args))
                expected_weights,expected=read(baseline/'checkpoints/epoch_8');expected_rng=capture_rng()
                partial=dict(args,output=str(root/'interrupted'),stop_after_steps=3,checkpoint_every_steps=1)
                runner.run(SimpleNamespace(**partial))
                with self.assertRaisesRegex(ValueError,'masking protocol'):
                    runner.run(SimpleNamespace(**dict(partial,resume_from='latest',masking_method='mean-noise-v1',stop_after_steps=None)))
                torch.manual_seed(999);torch.cuda.manual_seed_all(999)
                resumed=runner.run(SimpleNamespace(**dict(partial,resume_from='latest',stop_after_steps=None)))
                weights,state=read(resumed/'checkpoints/epoch_8')
                assert_tree_equal(self,expected_weights,weights)
                for key in ('optimizer','scheduler','python_rng','numpy_rng','torch_rng','cuda_rng'):
                    assert_tree_equal(self,expected[key],state[key],key)
                assert_tree_equal(self,expected_rng,capture_rng())
                import csv
                with (resumed/'metrics.csv').open() as f:rows=list(csv.DictReader(f))
                self.assertEqual(len(rows),27)
                self.assertEqual({r['variant'] for r in rows},{'full','mask_image','mask_text'})
                self.assertEqual(len(state['trainer']['metrics']),24)
                predictions=[json.loads(line) for line in (resumed/'predictions/epoch_8_mask_text.jsonl').read_text().splitlines()]
                self.assertTrue(all(r['noise_seed'] is None and r['noise_scale']==0 for r in predictions))
                for path in (baseline/'predictions').glob('*.jsonl'):
                    self.assertEqual(path.read_bytes(),(resumed/'predictions'/path.name).read_bytes())
                self.assertTrue(test_launcher.launcher.complete_run(resumed))
                self.assertLessEqual(len(list((resumed/'checkpoints').glob('step_*'))),2)
                self.assertEqual(len(list((resumed/'checkpoints').glob('epoch_*'))),9)
                selection=root/f'{dataset}-selection.json'
                write_json(selection,dict(locked_at='test-prespecified',comparisons=[dict(run=str(resumed),config='L-low',epoch=2)]))
                runner.heldout(SimpleNamespace(data=common['data'],stats=common['stats'],selection=str(selection),output=str(root/'heldout'),cache_dir=None))
                held=root/'heldout'/dataset/'L-low/seed_42/epoch_2'
                with (held/'metrics.csv').open() as f:self.assertEqual(len(list(csv.DictReader(f))),3)
                held_pred=[json.loads(line) for line in (held/'predictions/epoch_2_mask_text.jsonl').read_text().splitlines()]
                self.assertTrue(all(r['noise_seed'] is None for r in held_pred))


if __name__=='__main__':unittest.main()
