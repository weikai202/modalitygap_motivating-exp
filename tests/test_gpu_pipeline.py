"""Actual CUDA orchestration with real processor and a small Qwen fixture.

Strong immediate EOS bias in this random-weight fixture keeps orchestration fast.
Pretrained generation is separately checked by scripts/validate_gpu.py.
"""
import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ENABLED = os.environ.get('MODALITYGAP_GPU_TESTS') == '1'


@unittest.skipUnless(ENABLED, 'Set MODALITYGAP_GPU_TESTS=1 outside sandbox on CUDA host')
class GPUPipelineTests(unittest.TestCase):
    def test_both_tasks_all_configs_full_horizon_and_heldout_reload(self):
        import torch
        from PIL import Image
        from transformers import AutoProcessor,Qwen2_5_VLConfig,Qwen2_5_VLForConditionalGeneration
        from modalitygap import runner
        from modalitygap.analysis import analyze
        from modalitygap.data import prepare
        from modalitygap.protocol import CONFIGS,DEFAULTS,write_json,write_jsonl
        self.assertTrue(torch.cuda.is_available())
        processor_path = os.environ['MODALITYGAP_PROCESSOR']
        processor = AutoProcessor.from_pretrained(processor_path,use_fast=False,local_files_only=True,
                        min_pixels=DEFAULTS['min_pixels'],max_pixels=DEFAULTS['max_pixels'])
        def loader(revision,cache_dir=None):
            torch.manual_seed(17)
            cfg = Qwen2_5_VLConfig(
                text_config=dict(vocab_size=151936,hidden_size=32,intermediate_size=64,
                    num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,
                    rope_scaling={'type':'mrope','mrope_section':[1,1,2]}),
                vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,
                    out_hidden_size=32,patch_size=14,spatial_merge_size=2,
                    temporal_patch_size=2,window_size=112,fullatt_block_indexes=[0]),
                image_token_id=151655,video_token_id=151656,vision_start_token_id=151652,
                vision_end_token_id=151653,eos_token_id=151645,pad_token_id=151643)
            model = Qwen2_5_VLForConditionalGeneration(cfg).to(device='cuda',dtype=torch.bfloat16)
            model.generation_config.sequence_bias = {(151645,):100.0}
            model.generation_config.eos_token_id = 151645
            return model.requires_grad_(False),processor
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner,'load_model',side_effect=loader):
            root = Path(tmp)
            for dataset in ('mmsd2','mmimdb'):
                rows = []
                for i in range(4):
                    image = root/f'{dataset}_{i}.png'
                    Image.new('RGB',(84+i*28,112),(30*i,40,90)).save(image)
                    rows.append(dict(id=str(i),split='train' if i<2 else 'dev' if i==2 else 'test',
                        image=str(image),text=f'Example {i}. '+('A person sees an unexpected event. '*(i+1)),
                        labels=['sarcastic' if i%2 else 'not_sarcastic'] if dataset=='mmsd2' else ['Drama','Comedy']))
                source, data = root/f'{dataset}.jsonl',root/dataset
                write_jsonl(source,rows)
                prepare(source,data,dataset,'synthetic-test-fixture')
                stats = root/f'{dataset}.pt'
                common = dict(data=str(data),dataset=dataset,revision='a'*40,cache_dir=None)
                runner.estimate(SimpleNamespace(**common,output=str(stats),samples=2))
                frozen = torch.load(stats,map_location='cpu',weights_only=True)
                baseline,_ = loader('a'*40)
                from modalitygap.model import Encoder
                from modalitygap.protocol import read_jsonl
                moments = {'text':[], 'image':[]}
                with torch.no_grad():
                    for row in read_jsonl(data/'train.jsonl'):
                        batch,positions,_ = Encoder(processor,dataset)(row)
                        batch = runner.to_device(batch)
                        text = baseline.get_input_embeddings()(batch['input_ids'])[0,positions]
                        image = torch.cat(baseline.model.get_image_features(batch['pixel_values'],batch['image_grid_thw']))
                        for modality,x in (('text',text),('image',image)):
                            x=x.double().cpu()
                            moments[modality].append((x.mean(0),x.square().mean(0)))
                for modality,values in moments.items():
                    mean=torch.stack([x[0] for x in values]).mean(0)
                    second=torch.stack([x[1] for x in values]).mean(0)
                    torch.testing.assert_close(frozen[modality]['mean'],mean.float(),rtol=1e-5,atol=1e-6)
                    torch.testing.assert_close(frozen[modality]['std'],(second-mean.square()).clamp_min(0).sqrt().float(),rtol=1e-5,atol=1e-6)
                del baseline,batch,text,image
                torch.cuda.empty_cache()
                for config in CONFIGS:
                    print(f'GPU integration: {dataset}/{config}',flush=True)
                    runner.run(SimpleNamespace(**common,stats=str(stats),output=str(root/'runs'),config=config,
                        seed=42,code_revision='integration-fixture'))
                    run = root/'runs'/dataset/config/'seed_42'
                    with (run/'metrics.csv').open() as f:
                        values = list(csv.DictReader(f))
                    cfg=json.loads((run/'config.json').read_text())
                    self.assertEqual(cfg['precision'],'bf16-mixed-fp32-trainable')
                    initial = torch.load(run/'checkpoints'/'epoch_0'/'trainable.pt',weights_only=True)
                    final = torch.load(run/'checkpoints'/'epoch_8'/'trainable.pt',weights_only=True)
                    self.assertTrue(all(v.dtype==torch.float32 for v in final.values()))
                    state=torch.load(run/'checkpoints'/'epoch_8'/'state.pt',map_location='cpu',weights_only=False)
                    self.assertEqual(state['step'],8)
                    self.assertEqual(state['scheduler']['last_epoch'],8)
                    self.assertEqual(state['optimizer']['param_groups'][0]['initial_lr'],CONFIGS[config]['lr'])
                    for param_state in state['optimizer']['state'].values():
                        self.assertEqual(param_state['exp_avg'].dtype,torch.float32)
                        self.assertEqual(param_state['exp_avg_sq'].dtype,torch.float32)
                    self.assertIn('python_rng',state)
                    self.assertIn('numpy_rng',state)
                    self.assertIn('cuda_rng',state)
                    self.assertTrue(any(not torch.equal(initial[n],final[n]) for n in initial))
                    self.assertEqual(len(values),27) # Nine epochs, three conditions each
                    self.assertEqual(len(list((run/'checkpoints').glob('epoch_*'))),9)
                    with (run/'metrics.csv').open() as f:
                        metrics = list(csv.DictReader(f))
                    self.assertTrue(all(float(r['invalid_rate'])==1 for r in metrics))
                    self.assertTrue(all((run/'checkpoints'/f'epoch_{e}'/'complete.json').exists() for e in range(9)))
                    if config=='LM-low':
                        selection = root/f'{dataset}-selection.json'
                        write_json(selection,dict(locked_at='2026-11-13T00:00:00Z',comparisons=[
                            dict(run=str(run),config=config,epoch=e) for e in (2,8)]))
                        runner.heldout(SimpleNamespace(data=str(data),stats=str(stats),selection=str(selection),
                            output=str(root/'test'),cache_dir=None))
                        for e in (2,8):
                            path = root/'test'/dataset/config/'seed_42'/f'epoch_{e}'/'metrics.csv'
                            with path.open() as f:
                                self.assertEqual(len(list(csv.DictReader(f))),3)
                analyze(root/'runs',dataset,root/'figures',bootstrap_samples=2000)
                self.assertEqual(len(list((root/'figures'/dataset).glob('*.png'))),7)
                report = json.loads((root/'figures'/dataset/'analysis.json').read_text())
                self.assertEqual(report['actual_main_evaluations'],162)
                self.assertEqual(report['actual_total_evaluations'],162)
                self.assertEqual(len(report['pairs']),30)
            torch.cuda.empty_cache()
