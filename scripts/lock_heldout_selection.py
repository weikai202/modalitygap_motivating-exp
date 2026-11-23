"""Explicitly lock chosen comparisons and print per-dataset test commands; never run test."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shlex,sys
from modalitygap.protocol import CONFIGS,write_json,digest
from run_experiment import load_config,complete_run,verify_checkpoint,verify_run_config


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--configs',nargs='+',choices=tuple(CONFIGS),required=True);p.add_argument('--seeds',nargs='+',type=int,choices=(42,43,44),required=True);p.add_argument('--output',required=True);a=p.parse_args()
    cfg=load_config(a.config);out=Path(a.output).resolve()
    if out.exists():raise ValueError('Selection already exists; do not replace a test lock')
    selections={};time=datetime.now(timezone.utc).isoformat()
    for name in cfg['datasets']:
        comparisons=[]
        for config in a.configs:
            for seed in a.seeds:
                run=cfg['workspace']/'runs'/name/config/f'seed_{seed}'
                if not complete_run(run):raise ValueError(f'Incomplete run: {run}')
                verify_run_config(cfg,name,config,seed,run,False)
                for epoch in (2,8):
                    verify_checkpoint(run/'checkpoints'/f'epoch_{epoch}')
                    comparisons.append(dict(run=str(run),config=config,epoch=epoch))
        selections[name]=dict(locked_at=time,purpose='Explicit development-selected comparisons; test epoch2 and8 only',pipeline_sha256=digest(cfg['workspace']/'pipeline.json'),comparisons=comparisons)
    for name,value in selections.items():
        path=out/f'{name}.json';write_json(path,value)
        command=[sys.executable,'-m','modalitygap.cli','test','--data',str(cfg['workspace']/'manifests'/name),'--stats',str(cfg['workspace']/'stats'/f'{name}.pt'),'--selection',str(path),'--output',str(cfg['workspace']/'heldout-v1'),'--cache-dir',str(cfg['cache_dir'])]
        print(shlex.join(command))


if __name__=='__main__':main()
