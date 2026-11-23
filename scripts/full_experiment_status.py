"""Read-only campaign progress and pilot-based checkpoint disk estimate."""
import argparse,json
from pathlib import Path
from run_experiment import load_config,pilot_configs
from modalitygap.protocol import CONFIGS


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=load_config(a.config)
    done=[];pending=[];sizes={};counts={}
    for name in cfg['datasets']:
        manifest=cfg['workspace']/'manifests'/name/'manifest.json'
        if manifest.exists():counts[name]=json.loads(manifest.read_text())['counts']
        for candidate in pilot_configs(cfg):
            folder=cfg['workspace']/'pilots'/name/candidate/'seed_42_pilot/checkpoints'
            checkpoints=[sum(x.stat().st_size for x in d.rglob('*') if x.is_file()) for d in folder.glob('*') if d.is_dir()]
            if checkpoints:sizes[name,CONFIGS[candidate]['scope']]=max(checkpoints)
        for candidate in cfg['configs']:
            for seed in cfg['seeds']:
                run=cfg['workspace']/'runs'/name/candidate/f'seed_{seed}';status=run/'progress.json'
                row=dict(dataset=name,config=candidate,seed=seed)
                if status.exists():row.update(json.loads(status.read_text()))
                (done if row.get('status')=='complete' else pending).append(row)
    estimated=None
    if len(sizes)==len(cfg['datasets'])*len(pilot_configs(cfg)) and cfg['keep_step_checkpoints']:
        estimated=sum(sizes[name,CONFIGS[c]['scope']]*(9+cfg['keep_step_checkpoints'])*len(cfg['seeds']) for name in cfg['datasets'] for c in cfg['configs'])/1024**3
    print(json.dumps(dict(workspace=str(cfg['workspace']),counts=counts,completed=len(done),total=len(done)+len(pending),pending=pending,
        estimated_checkpoint_GiB_from_pilots=estimated,disk_estimate_note='Approximation for nine epoch plus retained step checkpoints; excludes raw data/model/cache/pilots and transient save space.'),indent=2))


if __name__=='__main__':main()
