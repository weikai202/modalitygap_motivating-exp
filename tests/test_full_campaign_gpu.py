"""CUDA engineering check of all 36 symmetric jobs, three-seed analysis and test reload.

Tiny random Qwen + stochastic fixture encoder, not real-data research results.
"""
import csv,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('MODALITYGAP_GPU_TESTS')=='1','Enable CUDA integration tests')
class FullCampaignGPUTests(unittest.TestCase):
    def test_complete_two_dataset_six_config_three_seed_matrix(self):
        import torch
        from modalitygap import runner
        from modalitygap.protocol import CONFIGS,write_json
        from modalitygap.analysis import analyze
        from modalitygap.checkpoint import read
        from test_resume import loader,StochasticEncoder,fixtures
        import test_launcher
        with tempfile.TemporaryDirectory() as tmp,patch.object(runner,'load_model',side_effect=loader),patch.object(runner,'Encoder',StochasticEncoder):
            root=Path(tmp);runs=0
            for dataset in ('mmsd2','mmimdb'):
                common=fixtures(root,dataset)
                for config in CONFIGS:
                    for seed in (42,43,44):
                        print('FULL MATRIX',dataset,config,seed,flush=True)
                        args=dict(common,config=config,seed=seed,output=str(root/'runs'),masking_method='symmetric-attention-v2',keep_step_checkpoints=2,checkpoint_every_steps=1)
                        path=runner.run(SimpleNamespace(**args));runs+=1
                        self.assertTrue(test_launcher.launcher.complete_run(path))
                        weights,state=read(path/'checkpoints/epoch_8')
                        self.assertEqual(state['step'],16);self.assertEqual(state['scheduler']['last_epoch'],16)
                        self.assertEqual(len(list((path/'checkpoints').glob('epoch_*'))),9)
                        self.assertLessEqual(len(list((path/'checkpoints').glob('step_*'))),2)
                        initial,_=read(path/'checkpoints/epoch_0')
                        self.assertTrue(any(not torch.equal(v,initial[k]) for k,v in weights.items()))
                        with (path/'metrics.csv').open() as f:self.assertEqual(len(list(csv.DictReader(f))),27)
                analyze(root/'runs',dataset,root/'figures',bootstrap_samples=4)
                report=json.loads((root/'figures'/dataset/'analysis.json').read_text())
                self.assertEqual(len(report['bootstrap']),3)
                self.assertEqual(report['actual_main_evaluations'],486)
                selection=root/f'{dataset}-locked.json'
                path=root/'runs'/dataset/'LM-high/seed_44'
                write_json(selection,dict(locked_at='engineering-fixed',comparisons=[dict(run=str(path),config='LM-high',epoch=e) for e in (2,8)]))
                runner.heldout(SimpleNamespace(data=common['data'],stats=common['stats'],selection=str(selection),output=str(root/'heldout'),cache_dir=None))
                for epoch in (2,8):
                    p=root/'heldout'/dataset/'LM-high/seed_44'/f'epoch_{epoch}'/'metrics.csv'
                    with p.open() as f:self.assertEqual(len(list(csv.DictReader(f))),3)
            self.assertEqual(runs,36)
