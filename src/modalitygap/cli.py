import argparse

from .protocol import CONFIGS, DEFAULTS, MODEL_REVISION, LABELS


def main():
    parser = argparse.ArgumentParser(description='Protocol v2 modality reliance screening')
    commands = parser.add_subparsers(dest='command',required=True)
    p = commands.add_parser('prepare',help='Audit an explicit author-data JSONL export and freeze manifests')
    p.add_argument('--source',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--dataset',choices=tuple(LABELS),required=True)
    p.add_argument('--source-revision',required=True)
    p.add_argument('--pool-size',type=lambda x: 'all' if x=='all' else int(x),default=6000)
    p.add_argument('--overlap-policy',choices=('error','prefer-heldout'),default='error')
    p = commands.add_parser('matrix',help='Print the fixed candidate matrix (no training)')
    p.add_argument('--confirmation',action='store_true')
    for name in ('stats','run'):
        p = commands.add_parser(name)
        p.add_argument('--cache-dir',default=None)
        p.add_argument('--data',required=True)
        p.add_argument('--dataset',choices=tuple(LABELS),required=True)
        p.add_argument('--revision',default=MODEL_REVISION,help='Immutable 40-character model commit SHA (default: pinned study revision)')
        p.add_argument('--output',required=True)
        if name=='stats':
            p.add_argument('--samples',type=int,default=256)
        else:
            p.add_argument('--allow-legacy-resume',action='store_true',help='Allow old epoch checkpoints without an exact-continuation guarantee')
            p.add_argument('--resume-from',default=None,help='Checkpoint directory or latest in this run')
            p.add_argument('--keep-step-checkpoints',type=int,default=0,help='Retain newest N step snapshots; 0 keeps all; epoch snapshots always retained')
            p.add_argument('--checkpoint-every-steps',type=int,default=0,help='Also save after every N optimizer steps; 0 means epoch boundaries only')
            p.add_argument('--stop-after-steps',type=int,default=None,help='Graceful stop at this global optimizer step; eight-epoch schedule is unchanged')
            p.add_argument('--masking-method',choices=('mean-noise-v1','symmetric-attention-v2'),default='mean-noise-v1')
            p.add_argument('--stats',required=True)
            p.add_argument('--config',choices=tuple(CONFIGS),required=True)
            p.add_argument('--seed',type=int,choices=(42,43,44),default=42)
            p.add_argument('--code-revision',required=True)
            p.add_argument('--pilot-steps',type=int,default=0,help='Separate pilot output; screening always uses 8 epochs')
    p = commands.add_parser('test',help='Evaluate locked test comparisons at epochs 2 and 8 only')
    p.add_argument('--cache-dir',default=None)
    for arg in ('data','stats','selection','output'):
        p.add_argument('--'+arg,required=True)
    p = commands.add_parser('analyze')
    p.add_argument('--root',required=True)
    p.add_argument('--dataset',choices=tuple(LABELS),required=True)
    p.add_argument('--output',default='figures')
    p.add_argument('--bootstrap-samples',type=int,default=2000)
    args = parser.parse_args()
    for field in ('keep_step_checkpoints',):
        if getattr(args,field,0)<0:parser.error(f'{field} must be nonnegative')
    for field in ('samples','pool_size','bootstrap_samples'):
        if hasattr(args,field) and getattr(args,field)!='all' and getattr(args,field)<=0:
            parser.error(f'{field} must be positive')
    if hasattr(args,'pilot_steps') and not 0<=args.pilot_steps<=100:
        parser.error('pilot-steps must be 0 (screening) or 1–100')
    if getattr(args,'checkpoint_every_steps',0)<0 or (getattr(args,'stop_after_steps',None) is not None and args.stop_after_steps<=0):
        parser.error('checkpoint interval must be nonnegative and stop step must be positive')
    if args.command=='prepare':
        from .data import prepare
        prepare(args.source,args.output,args.dataset,args.source_revision,args.pool_size,overlap_policy=args.overlap_policy)
    elif args.command=='matrix':
        import json
        print(json.dumps([dict(dataset=d,config=c,seed=s,**DEFAULTS,**v)
            for d in ('mmsd2','mmimdb') for c,v in CONFIGS.items()
            for s in ((42,43,44) if args.confirmation else (42,))],indent=2))
    elif args.command=='stats':
        from .runner import estimate
        estimate(args)
    elif args.command=='run':
        from .runner import run
        run(args)
    elif args.command=='test':
        from .runner import heldout
        heldout(args)
    elif args.command=='analyze':
        from .analysis import analyze
        analyze(args.root,args.dataset,args.output,args.bootstrap_samples)


if __name__=='__main__':
    main()
