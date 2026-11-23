"""Fresh-process exact continuation test with the original pretrained 3B model."""
import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from modalitygap.checkpoint import capture_rng,read
from modalitygap.protocol import write_json
from modalitygap.runner import run


def compare(left,right,path='root'):
    if type(left) is not type(right):
        raise AssertionError(f'{path}: types differ')
    if isinstance(left,torch.Tensor):
        if left.dtype!=right.dtype or not torch.equal(left.cpu(),right.cpu()):
            raise AssertionError(f'{path}: tensors are not bitwise equal')
    elif isinstance(left,np.ndarray):
        if not np.array_equal(left,right):
            raise AssertionError(f'{path}: NumPy state differs')
    elif isinstance(left,dict):
        if left.keys()!=right.keys():
            raise AssertionError(f'{path}: keys differ')
        for key in left:
            compare(left[key],right[key],f'{path}.{key}')
    elif isinstance(left,(list,tuple)):
        if len(left)!=len(right):
            raise AssertionError(f'{path}: lengths differ')
        for i,(a,b) in enumerate(zip(left,right)):
            compare(a,b,f'{path}[{i}]')
    elif left!=right:
        raise AssertionError(f'{path}: values differ')


def main():
    import subprocess
    import sys
    from modalitygap.data import prepare
    from modalitygap.protocol import read_jsonl,write_jsonl,MODEL_REVISION
    from modalitygap.runner import estimate,configure_reproducibility
    parser=argparse.ArgumentParser()
    parser.add_argument('--fixtures',default='artifacts/pretrained-fp32-master-validation')
    parser.add_argument('--cache-dir',default='artifacts/hf-cache')
    parser.add_argument('--output',default='artifacts/resume-deterministic-pretrained')
    parser.add_argument('--datasets',nargs='+',choices=('mmsd2','mmimdb'),default=['mmsd2','mmimdb'])
    parser.add_argument('--worker',choices=('stats','baseline','stop','resume'),default=None)
    args=parser.parse_args()
    root=Path(args.output).resolve()
    if args.worker:
        dataset=args.datasets[0]
        configure_reproducibility()
        data=root/f'{dataset}-manifest'
        stats=root/f'{dataset}-stats.pt'
        common=dict(data=str(data),dataset=dataset,revision=MODEL_REVISION,cache_dir=args.cache_dir)
        if args.worker=='stats':
            estimate(SimpleNamespace(**common,output=str(stats),samples=17))
            return
        output=root/('baseline' if args.worker=='baseline' else 'interrupted')
        result=run(SimpleNamespace(**common,stats=str(stats),config='LM-high',seed=42,pilot_steps=0,
            code_revision='deterministic resume validation',output=str(output),
            resume_from='latest' if args.worker=='resume' else None,
            checkpoint_every_steps=0 if args.worker=='baseline' else 1,
            stop_after_steps=3 if args.worker=='stop' else None))
        if args.worker!='stop':
            torch.save(capture_rng(),result/'final-rng.pt')
        return
    if root.exists():
        raise ValueError('Choose a fresh output root')
    report=dict(status='running',scope='fresh-process uninterrupted versus optimizer-step-3 resume; original 3B, synthetic fixtures',runs=[])
    write_json(root/'report.json',report)
    for dataset in args.datasets:
        # 17 training examples exercise a full batch of 16 and a final batch of 1.
        rows=[]
        for split,n in (('train',17),('dev',1),('test',1)):
            rows.extend(read_jsonl(Path(args.fixtures)/f'{dataset}-manifest'/f'{split}.jsonl')[:n])
        source=root/f'{dataset}-source.jsonl'
        write_jsonl(source,rows)
        prepare(source,root/f'{dataset}-manifest',dataset,'synthetic engineering resume fixtures, 17/1/1')
        for mode in ('stats','baseline','stop','resume'):
            print(f'{dataset}: fresh process {mode}',flush=True)
            command=[sys.executable,__file__,'--worker',mode,'--datasets',dataset,
                     '--output',str(root),'--cache-dir',args.cache_dir]
            with (root/f'{dataset}-{mode}.log').open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        baseline=root/'baseline'/dataset/'LM-high'/'seed_42'
        resumed=root/'interrupted'/dataset/'LM-high'/'seed_42'
        expected_weights,expected=read(baseline/'checkpoints'/'epoch_8')
        weights,actual=read(resumed/'checkpoints'/'epoch_8')
        compare(expected_weights,weights,'parameters')
        for key in ('optimizer','scheduler','python_rng','numpy_rng','torch_rng','cuda_rng'):
            compare(expected[key],actual[key],key)
        compare(torch.load(baseline/'final-rng.pt',map_location='cpu',weights_only=False),
                torch.load(resumed/'final-rng.pt',map_location='cpu',weights_only=False),'post_evaluation_rng')
        report['runs'].append(dict(dataset=dataset,config='LM-high',training_examples=17,
            interruption_step=3,interruption_epoch=2,next_index=16,final_epoch=8,
            final_step=actual['step'],parameters='bitwise_equal',optimizer='bitwise_equal',
            scheduler='equal',python_rng='equal',numpy_rng='equal',torch_rng='equal',cuda_rng='equal',
            separate_processes=True))
        write_json(root/'report.json',report)
        del expected_weights,expected,weights,actual
        gc.collect()
    report['status']='passed'
    write_json(root/'report.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
