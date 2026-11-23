"""Eight-epoch pretrained end-to-end engineering validation on synthetic fixtures.

Runs LM-high (largest adaptation scope and rate) for each task, all main and
sensitivity evaluations, locked epoch-2/8 held-out evaluations, and analysis.
This is a software check; it is not an MMSD2/MM-IMDb research experiment.
"""
import argparse
import csv
import gc
import json
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace

import torch
from modalitygap.analysis import analyze
from modalitygap.protocol import digest,write_json
from modalitygap.runner import run,heldout


def csv_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fixtures',default='artifacts/validation/resource')
    p.add_argument('--cache-dir',default='artifacts/hf-cache')
    p.add_argument('--output',default='artifacts/validation/full')
    a=p.parse_args()
    root,fixtures=Path(a.output),Path(a.fixtures)
    if root.exists():
        raise ValueError('Use a fresh output directory')
    fixture_report=json.loads((fixtures/'report.json').read_text())
    if fixture_report.get('status')!='passed':
        raise ValueError('Complete the max-budget resource pilot first')
    root.mkdir(parents=True)
    report=dict(kind='pretrained synthetic engineering validation, not research results',
                scope='LM-high, both tasks, eight epochs, full/sensitivity/held-out/analysis',runs=[])
    write_json(root/'report.json',report)
    for dataset in ('mmsd2','mmimdb'):
        data,stats=fixtures/f'{dataset}-manifest',fixtures/f'{dataset}-stats.pt'
        args=dict(dataset=dataset,data=str(data),stats=str(stats),cache_dir=a.cache_dir,
                  revision=fixture_report['model_revision'],config='LM-high',seed=42,
                  pilot_steps=0,code_revision='engineering validation; exact hashes in config.json',
                  output=str(root/'runs'))
        print(f'[{dataset}] pretrained LM-high full eight-epoch run',flush=True)
        run(SimpleNamespace(**args))
        gc.collect()
        torch.cuda.empty_cache()
        directory=root/'runs'/dataset/'LM-high'/'seed_42'
        values=csv_rows(directory/'metrics.csv')
        assert len(values)==45
        resources=json.loads((directory/'resources.json').read_text())
        assert len(resources)==8 and resources[-1]['step']==16
        assert len(list((directory/'checkpoints').glob('epoch_*')))==9
        selection=root/f'{dataset}-locked-selection.json'
        write_json(selection,dict(locked_at=datetime.now(timezone.utc).isoformat(),
            purpose='prespecified software check with synthetic test fixtures',comparisons=[
                dict(run=str(directory),config='LM-high',epoch=epoch) for epoch in (2,8)]))
        print(f'[{dataset}] locked epoch-2/8 held-out generation',flush=True)
        heldout(SimpleNamespace(data=str(data),stats=str(stats),cache_dir=a.cache_dir,
                                selection=str(selection),output=str(root/'heldout')))
        gc.collect()
        torch.cuda.empty_cache()
        print(f'[{dataset}] paired 2000-resample bootstrap and seven figures',flush=True)
        analyze(root/'runs',dataset,root/'figures',bootstrap_samples=2000)
        assert len(list((root/'figures'/dataset).glob('*.png')))==7
        for epoch in (2,8):
            assert len(csv_rows(root/'heldout'/dataset/'LM-high'/'seed_42'/f'epoch_{epoch}'/'metrics.csv'))==3
        report['runs'].append(dict(dataset=dataset,config='LM-high',epochs=8,optimizer_steps=16,
            main_evaluations=27,sensitivity_evaluations=18,heldout_evaluations=6,figures=7,
            max_allocated_bytes=max(r['allocated_bytes'] for r in resources),
            max_reserved_bytes=max(r['reserved_bytes'] for r in resources),
            training_seconds=sum(r['seconds'] for r in resources),
            validation_seconds=sum(float(r['validation_seconds']) for r in values)))
        write_json(root/'report.json',report)
    report['status']='passed'
    write_json(root/'report.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
