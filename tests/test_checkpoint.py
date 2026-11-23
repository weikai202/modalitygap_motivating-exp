import copy
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from modalitygap.checkpoint import capture_rng, latest, read, restore_rng, save
from modalitygap.runner import load_trainable


def assert_tree_equal(case, left, right, path='root'):
    case.assertEqual(type(left), type(right), path)
    if isinstance(left, torch.Tensor):
        case.assertTrue(torch.equal(left.cpu(), right.cpu()), path)
    elif isinstance(left, np.ndarray):
        case.assertTrue(np.array_equal(left, right), path)
    elif isinstance(left, dict):
        case.assertEqual(left.keys(), right.keys(), path)
        for key in left:
            assert_tree_equal(case,left[key],right[key],f'{path}.{key}')
    elif isinstance(left,(tuple,list)):
        case.assertEqual(len(left),len(right),path)
        for i,(a,b) in enumerate(zip(left,right)):
            assert_tree_equal(case,a,b,f'{path}[{i}]')
    else:
        case.assertEqual(left,right,path)


class Processor:
    def save_pretrained(self,path):
        Path(path).mkdir(parents=True)
        # Serialization must not disturb any global RNG stream.
        random.random(); np.random.rand(); torch.rand(1)
        if torch.cuda.is_available():
            torch.rand(1,device='cuda')
        (Path(path)/'processor.json').write_text('{}')


class CheckpointTests(unittest.TestCase):
    def test_parameter_optimizer_scheduler_rng_roundtrip_with_dropout(self):
        from transformers import get_cosine_schedule_with_warmup
        def setup():
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
            model=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.Dropout(.4),torch.nn.Linear(8,2))
            optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
            scheduler=get_cosine_schedule_with_warmup(optimizer,2,8)
            return model,optimizer,scheduler
        def advance(model,optimizer,scheduler,start,end):
            model.train()
            for _ in range(start,end):
                x=torch.randn(3,4)+random.random()+float(np.random.rand())
                if torch.cuda.is_available():
                    x=x+torch.rand((),device='cuda').item()
                model(x).square().mean().backward()
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        baseline,opt,sched=setup()
        advance(baseline,opt,sched,0,8)
        expected=(copy.deepcopy(baseline.state_dict()),copy.deepcopy(opt.state_dict()),copy.deepcopy(sched.state_dict()),capture_rng())
        model,optimizer,scheduler=setup()
        advance(model,optimizer,scheduler,0,3)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'step_3'
            before=capture_rng()
            save(model,optimizer,scheduler,path,1,3,Processor(),trainer={'next_index':48})
            assert_tree_equal(self,before,capture_rng())
            self.assertEqual(latest(Path(tmp)),path)
            weights,state=read(path)
            model,optimizer,scheduler=setup()
            load_trainable(model,weights)
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            # Simulate arbitrary RNG consumption in a restarted process.
            random.random(); np.random.rand(); torch.rand(10)
            if torch.cuda.is_available():
                torch.rand(10,device='cuda')
            restore_rng(state)
            advance(model,optimizer,scheduler,3,8)
            for left,right in zip(expected,(model.state_dict(),optimizer.state_dict(),scheduler.state_dict(),capture_rng())):
                assert_tree_equal(self,left,right)

    def test_serialization_failure_does_not_publish_or_advance_rng(self):
        class BrokenProcessor(Processor):
            def save_pretrained(self,path):
                super().save_pretrained(path)
                raise RuntimeError('simulated serialization failure')
        model=torch.nn.Linear(2,2)
        optimizer=torch.optim.AdamW(model.parameters())
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1)
        with tempfile.TemporaryDirectory() as tmp:
            before=capture_rng()
            with self.assertRaisesRegex(RuntimeError,'serialization'):
                save(model,optimizer,scheduler,Path(tmp)/'step_1',1,1,BrokenProcessor())
            self.assertEqual(list(Path(tmp).iterdir()),[])
            assert_tree_equal(self,before,capture_rng())

    def test_invalid_completion_markers_and_unknown_formats_are_rejected(self):
        from modalitygap.protocol import digest,write_json
        model=torch.nn.Linear(2,2)
        optimizer=torch.optim.AdamW(model.parameters())
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            save(model,optimizer,scheduler,root/'step_0',0,0,Processor())
            (root/'step_1').mkdir()
            (root/'step_1'/'complete.json').write_text('{')
            with self.assertRaisesRegex(ValueError,'completion marker'):
                latest(root)
            cp=root/'step_0'
            state=torch.load(cp/'state.pt',map_location='cpu',weights_only=False)
            state['format_version']=99
            torch.save(state,cp/'state.pt')
            mark=json.loads((cp/'complete.json').read_text())
            mark.update(format_version=99,state_sha256=digest(cp/'state.pt'))
            write_json(cp/'complete.json',mark)
            with self.assertRaisesRegex(ValueError,'format'):
                read(cp)

    def test_incomplete_corrupt_and_accumulating_checkpoints_are_rejected(self):
        model=torch.nn.Linear(2,2)
        optimizer=torch.optim.AdamW(model.parameters())
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'.step_99-incomplete').mkdir()
            save(model,optimizer,scheduler,root/'step_0',0,0,Processor())
            save(model,optimizer,scheduler,root/'.step_1-unpublished',1,1,Processor())
            self.assertEqual(latest(root),root/'step_0')
            with self.assertRaises(ValueError):
                save(model,optimizer,scheduler,root/'step_0',0,0,Processor())
            model(torch.ones(1,2)).sum().backward()
            with self.assertRaises(ValueError):
                save(model,optimizer,scheduler,root/'step_1',1,1,Processor())
            self.assertFalse((root/'step_1').exists())
            (root/'step_0'/'state.pt').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                read(root/'step_0')
            with self.assertRaises(ValueError):
                latest(root)
