"""Run one prespecified L-low pilot per real dataset; no test-set evaluation."""
import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from modalitygap.protocol import MODEL_REVISION,write_json
from modalitygap.runner import configure_reproducibility,estimate,run


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=('mmsd2','mmimdb'),required=True)
    p.add_argument('--root',default='artifacts/real-data-pilot-v1')
    p.add_argument('--pilot-steps',type=int,default=8)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    root=Path(a.root)
    data=Path('data/real-pilot')/a.dataset/'manifest'
    if not (data/'manifest.json').is_file():raise ValueError('Prepare and audit the real-data manifest first')
    manifest=json.loads((data/'manifest.json').read_text())
    train_count=manifest['counts']['train'];dev_count=manifest['counts']['dev']
    configure_reproducibility()
    stats=root/'stats'/f'{a.dataset}.pt'
    common=dict(data=str(data),dataset=a.dataset,revision=MODEL_REVISION,cache_dir='artifacts/hf-cache')
    if not stats.exists():
        print(f'Frozen-original-checkpoint moments on all {train_count} fixed real training examples',flush=True)
        estimate(SimpleNamespace(**common,output=str(stats),samples=train_count))
        gc.collect();torch.cuda.empty_cache()
    directory=root/'runs'/a.dataset/'L-low'/'seed_42_pilot'
    args=SimpleNamespace(**common,stats=str(stats),output=str(root/'runs'),config='L-low',seed=42,
        pilot_steps=a.pilot_steps,code_revision='prespecified-real-data-pilot-L-low-seed42',
        checkpoint_every_steps=4,stop_after_steps=None,resume_from='latest' if a.resume else None)
    print(f'Real {a.dataset}: L-low, seed42, max {a.pilot_steps} optimizer steps; three validation conditions',flush=True)
    run(args)
    write_json(directory/'pilot-purpose.json',dict(kind='real-data small resource/metric sanity pilot',
        config='L-low',training_seed=42,pilot_steps=a.pilot_steps,full_screening=False,
        train_count=train_count,dev_count=dev_count,test_used=False,
        note='F1 on fixed dev subset is exploratory; no ranking-reversal or causal contribution claims'))
    print('REAL PILOT DONE',directory,flush=True)


if __name__=='__main__':main()
