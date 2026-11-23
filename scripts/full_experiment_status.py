"""Read-only progress for the full training matrix."""
import argparse,json
from run_experiment import load_config


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=load_config(a.config)
    done=[];pending=[];counts={}
    for name in cfg['datasets']:
        manifest=cfg['workspace']/'manifests'/name/'manifest.json'
        if manifest.exists():counts[name]=json.loads(manifest.read_text())['counts']
        for candidate in cfg['configs']:
            for seed in cfg['seeds']:
                status=cfg['workspace']/'runs'/name/candidate/f'seed_{seed}'/'progress.json'
                row=dict(dataset=name,config=candidate,seed=seed)
                if status.exists():row.update(json.loads(status.read_text()))
                (done if row.get('status')=='complete' else pending).append(row)
    print(json.dumps(dict(workspace=str(cfg['workspace']),counts=counts,completed=len(done),total=len(done)+len(pending),pending=pending),indent=2))


if __name__=='__main__':main()
