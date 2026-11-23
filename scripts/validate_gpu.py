"""Pretrained 3090 resource/interface smoke test. Synthetic inputs, NOT research results.

Uses unchanged protocol limits, precision, accumulation, and generation budgets.
Runs 32 training fixtures/task, two dev fixtures/task, and two optimizer steps for
L/M/LM. Full eight-epoch orchestration is tested separately with small weights.
"""
import argparse
import csv
import gc
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image,ImageDraw

from modalitygap.data import prepare
from modalitygap.protocol import DEFAULTS,CONFIGS,CONDITIONS,GENRES,digest,read_jsonl,write_json,write_jsonl
from modalitygap.runner import estimate,run,load_trainable,to_device
from modalitygap.model import Encoder,Intervention,adapt,load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache-dir',default='artifacts/hf-cache')
    p.add_argument('--output',default='artifacts/validation/resource')
    a = p.parse_args()
    root = Path(a.output)
    if root.exists():
        raise ValueError('Use a fresh validation output directory')
    root.mkdir(parents=True)
    revision = Path(a.cache_dir,'model-revision.txt').read_text().strip()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA/BF16 unavailable; run outside device-hiding sandbox')
    report = dict(kind='synthetic engineering smoke test; not dataset performance',
                  model_revision=revision,device=torch.cuda.get_device_name(),
                  torch=torch.__version__,cuda=torch.version.cuda,limits=DEFAULTS,runs=[])
    write_json(root/'report.json',report)
    for dataset in ('mmsd2','mmimdb'):
        rows = []
        for i in range(36):
            image = root/f'{dataset}_{i}.png'
            # Unique RGB pixels and unique text across splits, varied aspect ratios.
            im = Image.new('RGB',(448,448) if i%2==0 else (672,280),(20+i*4,70+i,120))
            ImageDraw.Draw(im).text((10,10),f'Synthetic {dataset} fixture {i}',fill='white')
            im.save(image)
            text = f'Example {dataset} {i}. '+('A person encounters an unexpected scene. '*100)
            rows.append(dict(id=f'{dataset}-{i}',split='train' if i<32 else 'dev' if i<34 else 'test',
                image=str(image.resolve()),text=text,
                labels=['sarcastic' if i%2 else 'not_sarcastic'] if dataset=='mmsd2' else list(GENRES)))
        source,data = root/f'{dataset}-source.jsonl',root/f'{dataset}-manifest'
        write_jsonl(source,rows)
        prepare(source,data,dataset,'generated synthetic fixtures for engineering validation')
        stats = root/f'{dataset}-stats.pt'
        common = dict(data=str(data),dataset=dataset,revision=revision,cache_dir=a.cache_dir)
        print(f'[{dataset}] frozen-checkpoint statistics on 32 examples',flush=True)
        estimate(SimpleNamespace(**common,output=str(stats),samples=32))
        gc.collect()
        torch.cuda.empty_cache()
        for config in ('L-low','M-low','LM-low'):
            print(f'[{dataset}] {config}: epoch 0, two optimizer steps, three-condition epoch 1',flush=True)
            run(SimpleNamespace(**common,stats=str(stats),output=str(root/'runs'),config=config,
                                seed=42,pilot_steps=2,code_revision='working-tree; see source-sha256.json'))
            directory=root/'runs'/dataset/config/'seed_42_pilot'
            resources=json.loads((directory/'resources.json').read_text())
            with (directory/'metrics.csv').open() as f:
                evaluation=list(csv.DictReader(f))
            assert len(evaluation)==6 and len(resources)==1 and resources[0]['step']==2
            assert resources[0]['last_grad_norm']>0
            initial=torch.load(directory/'checkpoints'/'epoch_0'/'trainable.pt',map_location='cpu',weights_only=True)
            updated=torch.load(directory/'checkpoints'/'epoch_1'/'trainable.pt',map_location='cpu',weights_only=True)
            changed=sum(not torch.equal(v,updated[n]) for n,v in initial.items())
            assert changed>0
            del initial,updated
            assert resources[0]['allocated_bytes']<torch.cuda.get_device_properties(0).total_memory
            gc.collect()
            torch.cuda.empty_cache()
            print(f'[{dataset}] {config}: reload checkpoint and reproduce all three conditions',flush=True)
            model,processor=load_model(revision,cache_dir=a.cache_dir)
            model,_=adapt(model,CONFIGS[config]['scope'],42)
            state=torch.load(directory/'checkpoints'/'epoch_1'/'trainable.pt',map_location='cpu',weights_only=True)
            load_trainable(model,state)
            model.eval()
            encoder=Encoder(processor,dataset)
            hook=Intervention(model,torch.load(stats,map_location='cpu',weights_only=True))
            row=read_jsonl(data/'dev.jsonl')[0]
            try:
                with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
                    for condition in CONDITIONS:
                        batch,positions,_=encoder(row)
                        batch=to_device(batch)
                        with hook.use(row,batch,positions,condition):
                            actual=model.generate(**batch,do_sample=False,use_cache=True,
                                max_new_tokens=8 if dataset=='mmsd2' else 128)
                        tokens=actual[0,batch['input_ids'].shape[1]:].cpu().tolist()
                        expected=read_jsonl(directory/'predictions'/f'epoch_1_{condition}.jsonl')[0]
                        if tokens != expected['generated_token_ids']:
                            raise AssertionError(f'Checkpoint reload changed {dataset}/{config}/{condition} generation')
            finally:
                hook.close()
            del model,hook,batch,actual,state
            report['runs'].append(dict(dataset=dataset,config=config,resources=resources,
                checkpoint_reload_reproduces_three_conditions=True,updated_trainable_tensors=changed,
                evaluations=len(evaluation),validation_seconds=sum(float(x['validation_seconds']) for x in evaluation)))
            write_json(root/'report.json',report)
            gc.collect()
            torch.cuda.empty_cache()
    report['status']='passed'
    report['validation_scope']='Original pretrained 3B, real processor, BF16 CUDA; synthetic fixtures only'
    write_json(root/'source-sha256.json',{str(p):digest(p) for p in Path('src').rglob('*.py')})
    write_json(root/'report.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
