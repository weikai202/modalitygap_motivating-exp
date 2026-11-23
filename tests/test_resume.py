"""Exercise the production runner's restore path, not a separate training loop."""
import copy
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from modalitygap.checkpoint import capture_rng, read
from modalitygap.protocol import DEFAULTS, LABELS, digest, write_json, write_jsonl
from test_checkpoint import assert_tree_equal, Processor

GPU = os.environ.get('MODALITYGAP_GPU_TESTS') == '1'


class Tokenizer:
    eos_token_id=127
    pad_token_id=0
    def decode(self,ids,**kwargs):
        return 'invalid' if ids else ''


class TestProcessor(Processor):
    tokenizer=Tokenizer()


class StochasticEncoder:
    """Test-only augmentation deliberately exercises all four RNG streams."""
    def __init__(self,processor,dataset):
        self.processor=processor
    def __call__(self,row,training=False):
        ids=torch.tensor([[1,91,92,92,92,92,93,5,6,7]])
        pixels=torch.full((16,3*2*14*14),float(int(row['id'])+1)/50)
        batch=dict(input_ids=ids,attention_mask=torch.ones_like(ids),
                   image_grid_thw=torch.tensor([[1,4,4]]),pixel_values=pixels)
        if training:
            jitter=random.random()+float(np.random.rand())+torch.rand(()).item()+torch.rand((),device='cuda').item()
            batch['pixel_values']=pixels+jitter*.01
            batch['labels']=torch.tensor([[-100]*8+[6,7]])
        return batch,[7,8,9],False


def loader(revision,cache_dir=None):
    from transformers import Qwen2_5_VLConfig,Qwen2_5_VLForConditionalGeneration
    torch.manual_seed(17)
    config=Qwen2_5_VLConfig(
        text_config=dict(vocab_size=128,hidden_size=32,intermediate_size=64,
            num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,
            rope_scaling={'type':'mrope','mrope_section':[1,1,2]}),
        vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,
            out_hidden_size=32,patch_size=14,spatial_merge_size=2,
            temporal_patch_size=2,window_size=112,fullatt_block_indexes=[0]),
        image_token_id=92,video_token_id=94,vision_start_token_id=91,vision_end_token_id=93,
        eos_token_id=127,pad_token_id=0)
    model=Qwen2_5_VLForConditionalGeneration(config).to(device='cuda',dtype=torch.bfloat16)
    model.generation_config.sequence_bias={(127,):100.}
    model.generation_config.eos_token_id=127
    return model.requires_grad_(False),TestProcessor()


def fixtures(root,dataset):
    data=root/dataset
    # 19 examples exercises accumulation=16 and the final partial batch of 3.
    for split,n in (('train',19),('dev',1),('test',1)):
        write_jsonl(data/f'{split}.jsonl',[
            dict(id=str(i),group=str(i),text=f'Example {i}',image='test encoder supplies pixels',
                 labels=[LABELS[dataset][-1]]) for i in range(n)])
    write_json(data/'manifest.json',dict(dataset=dataset,requested_pool_size='all',counts=dict(train=19,dev=1,test=1),retained_complete_counts=dict(train=19,dev=1,test=1),
        split_hashes={s:digest(data/f'{s}.jsonl') for s in ('train','dev','test')}))
    stats=root/f'{dataset}.pt'
    torch.save(dict(**{m:dict(mean=torch.zeros(32),std=torch.ones(32)) for m in ('image','text')},
        metadata=dict(revision='a'*40,manifest_sha256=digest(data/'manifest.json'),input_limits=DEFAULTS)),stats)
    return dict(data=str(data),stats=str(stats),dataset=dataset,revision='a'*40,cache_dir=None,
        seed=42,code_revision='resume-test',checkpoint_every_steps=0,keep_step_checkpoints=2,
        stop_after_steps=None,resume_from=None)


@unittest.skipUnless(GPU,'Set MODALITYGAP_GPU_TESTS=1 on a CUDA host')
class ResumeTests(unittest.TestCase):
    def test_mid_epoch_resume_is_bitwise_identical_for_both_tasks_and_all_scopes(self):
        from modalitygap import runner
        original_evaluate=runner.evaluate
        def evaluate(*args,**kwargs):
            result=original_evaluate(*args,**kwargs)
            # A pending evaluation must be replayed once, with RNG restored beforehand.
            random.random(); np.random.rand(); torch.rand(())
            torch.rand((),device='cuda')
            return result
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner,'load_model',side_effect=loader), \
             patch.object(runner,'Encoder',StochasticEncoder), patch.object(runner,'evaluate',side_effect=evaluate):
            root=Path(tmp)
            for dataset in ('mmsd2','mmimdb'):
                common=fixtures(root,dataset)
                for config in ('L-low','M-low','LM-low'):
                    with self.subTest(dataset=dataset,config=config):
                        print(f'Resume equivalence: {dataset}/{config}',flush=True)
                        args=dict(common,config=config,output=str(root/'baseline'))
                        baseline=runner.run(SimpleNamespace(**args))
                        baseline_rng=capture_rng()
                        expected_weights,expected=read(baseline/'checkpoints'/'epoch_8')
                        interrupted=dict(args,output=str(root/'interrupted'),stop_after_steps=3,checkpoint_every_steps=1)
                        path=runner.run(SimpleNamespace(**interrupted))
                        self.assertEqual(path.name,'step_3')
                        _,middle=read(path)
                        self.assertEqual(middle['trainer']['epoch'],2)
                        self.assertEqual(middle['trainer']['next_index'],16)
                        # Change every global stream before restarting model construction.
                        random.seed(999); np.random.seed(999); torch.manual_seed(999)
                        torch.cuda.manual_seed_all(999)
                        if dataset=='mmimdb' and config=='LM-low':
                            second=runner.run(SimpleNamespace(**dict(interrupted,resume_from='latest',stop_after_steps=7)))
                            self.assertEqual(second.name,'step_7')
                            random.seed(77); np.random.seed(77); torch.manual_seed(77); torch.cuda.manual_seed_all(77)
                        resumed=runner.run(SimpleNamespace(**dict(interrupted,resume_from='latest',stop_after_steps=None)))
                        weights,actual=read(resumed/'checkpoints'/'epoch_8')
                        assert_tree_equal(self,expected_weights,weights)
                        for key in ('optimizer','scheduler','python_rng','numpy_rng','torch_rng','cuda_rng'):
                            assert_tree_equal(self,expected[key],actual[key],key)
                        assert_tree_equal(self,baseline_rng,capture_rng())
                        # Training losses and step counts agree; elapsed wall time is intentionally excluded.
                        for left,right in zip(expected['trainer']['resources'],actual['trainer']['resources']):
                            for key in ('epoch','step','mean_loss','mean_length','max_length','truncation_rate','last_grad_norm'):
                                self.assertEqual(left[key],right[key],key)
                        self.assertEqual(len(actual['trainer']['metrics']),24) # epoch 8 is pending in snapshot
                        self.assertEqual(json.loads((resumed/'progress.json').read_text())['status'],'complete')
                        self.assertLessEqual(len(list((resumed/'checkpoints').glob('step_*'))),2)
                        if dataset=='mmsd2' and config=='LM-low':
                            self.check_pending_evaluation_and_fork(runner,args,baseline,expected_weights,expected,baseline_rng,root)

    def check_pending_evaluation_and_fork(self,runner,args,baseline,weights,state,rng,root):
        fork=runner.run(SimpleNamespace(**dict(args,output=str(root/'fork'),
            resume_from=str(baseline/'checkpoints'/'epoch_2'))))
        actual_weights,actual=read(fork/'checkpoints'/'epoch_8')
        assert_tree_equal(self,weights,actual_weights)
        for key in ('optimizer','scheduler','python_rng','numpy_rng','torch_rng','cuda_rng'):
            assert_tree_equal(self,state[key],actual[key],key)
        assert_tree_equal(self,rng,capture_rng())
        self.assertTrue((fork/'predictions'/'epoch_0_full.jsonl').exists())
        self.assertTrue((fork/'checkpoints'/'epoch_1'/'trainable.pt').exists())
        # Earlier checkpoint cannot overwrite an already completed branch.
        with self.assertRaisesRegex(ValueError,'Newer checkpoints'):
            runner.run(SimpleNamespace(**dict(args,resume_from=str(baseline/'checkpoints'/'epoch_2'))))
        # Mismatched run provenance is rejected before creating a destination.
        with self.assertRaisesRegex(ValueError,'configuration'):
            runner.run(SimpleNamespace(**dict(args,seed=43,output=str(root/'bad'),
                resume_from=str(baseline/'checkpoints'/'epoch_2'))))
        self.assertFalse((root/'bad').exists())
