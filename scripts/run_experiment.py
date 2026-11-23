"""Sequential, resumable campaign launcher. Never invents research input data."""
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from modalitygap.protocol import CONFIGS, CONDITIONS, DEFAULTS, MODEL_REVISION, LABELS, digest

REPO = Path(__file__).resolve().parents[1]
STAGES = ('all', 'prepare', 'stats', 'train', 'analyze')


def timestamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def core_hashes():
    return {p.name:digest(p) for p in (REPO/'src'/'modalitygap').glob('*.py')}


def load_config(path):
    path = Path(path).resolve()
    cfg = read_json(path)
    allowed = {'workspace','cache_dir','model_revision','datasets','configs','seeds',
               'stats_samples','checkpoint_every_steps','masking_method','pool_size','overlap_policy','keep_step_checkpoints'}
    if not isinstance(cfg, dict) or set(cfg)-allowed:
        raise ValueError('Configuration must be an object with only documented keys')
    def resolve(value):
        if not isinstance(value, str) or not value:
            raise ValueError('Paths must be nonempty strings')
        return (path.parent/value).resolve()
    workspace = resolve(cfg['workspace'])
    cache = resolve(cfg.get('cache_dir','../artifacts/hf-cache'))
    revision = cfg.get('model_revision',MODEL_REVISION)
    if not isinstance(revision,str) or len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('model_revision must be a full 40-character commit SHA')
    datasets = cfg.get('datasets')
    if not isinstance(datasets,dict) or not datasets or set(datasets)-set(LABELS):
        raise ValueError('datasets must contain supported task names: mmsd2 and mmimdb')
    prepared = {}
    for name, entry in datasets.items():
        if not isinstance(entry,dict) or set(entry)!={'source','source_revision'}:
            raise ValueError(f'{name}: provide source and source_revision')
        description = entry['source_revision']
        if not isinstance(description,str) or not description.strip() or 'REPLACE_WITH' in description:
            raise ValueError(f'{name}: replace source_revision with the actual author release/export rule')
        source = resolve(entry['source'])
        if source.is_relative_to(workspace):
            raise ValueError('Keep original source JSONL/images outside the generated workspace')
        prepared[name] = dict(source=source, source_revision=description)
    configs = cfg.get('configs',list(CONFIGS))
    seeds = cfg.get('seeds',[42,43,44])
    if not isinstance(configs,list) or not configs or any(not isinstance(x,str) or x not in CONFIGS for x in configs) or len(set(configs))!=len(configs):
        raise ValueError('configs must list each of the six candidates exactly once')
    if not isinstance(seeds,list) or not seeds or any(type(x)!=int or x not in (42,43,44) for x in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError('seeds must list 42,43,44 exactly once')
    if set(configs)!=set(CONFIGS) or set(seeds)!={42,43,44}:
        raise ValueError('Full experiments require all six configurations and seeds 42,43,44')
    values={k:cfg.get(k,v) for k,v in dict(stats_samples=256,checkpoint_every_steps=50,keep_step_checkpoints=2).items()}
    if any(type(v) is not int for v in values.values()) or values['stats_samples']<=0 or values['checkpoint_every_steps']<0 or values['keep_step_checkpoints']<0:
        raise ValueError('Invalid statistics/checkpoint settings')
    pool=cfg.get('pool_size','all')
    if pool!='all':raise ValueError('Full experiments require pool_size=all')
    overlap=cfg.get('overlap_policy','prefer-heldout')
    if overlap not in ('error','prefer-heldout'):raise ValueError('Unknown overlap_policy')
    method=cfg.get('masking_method','symmetric-attention-v2')
    if method!='symmetric-attention-v2':raise ValueError('Only symmetric-attention-v2 is supported')
    return dict(pool_size=pool,overlap_policy=overlap,masking_method=method,config_file=path,workspace=workspace,cache_dir=cache,model_revision=revision,
                datasets=prepared,configs=configs,seeds=seeds,**values)


@contextmanager
def workspace_lock(workspace):
    workspace.mkdir(parents=True,exist_ok=True)
    with (workspace/'.pipeline.lock').open('a+') as handle:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('This workspace is already running; do not start a second trainer') from exc
        handle.seek(0); handle.truncate()
        handle.write(json.dumps(dict(pid=os.getpid(),started=timestamp()))+'\n'); handle.flush()
        # Children inherit this FD so a killed launcher cannot unlock a live trainer.
        yield handle.fileno()


def cli(*args):
    return [sys.executable,'-u','-m','modalitygap.cli',*map(str,args)]


def common(cfg,dataset):
    return ['--dataset',dataset,'--data',str(cfg['workspace']/'manifests'/dataset),
            '--revision',cfg['model_revision'],'--cache-dir',str(cfg['cache_dir'])]


def train_command(cfg,dataset,config,seed):
    return cli('run',*common(cfg,dataset),'--stats',cfg['workspace']/'stats'/f'{dataset}.pt',
        '--config',config,'--seed',seed,'--output',cfg['workspace']/'runs',
        '--code-revision','core-sha256:'+hashlib.sha256(json.dumps(core_hashes(),sort_keys=True).encode()).hexdigest(),
        '--checkpoint-every-steps',cfg['checkpoint_every_steps'],'--keep-step-checkpoints',cfg['keep_step_checkpoints'],
        '--masking-method','symmetric-attention-v2')


def preview(cfg,stage,resume,stop):
    print('DRY RUN: no files changed, no commands executed.')
    print('Workspace:',cfg['workspace'])
    if stage in ('all','prepare','stats','train'):
        for name,entry in cfg['datasets'].items():
            print(shlex.join(cli('prepare','--dataset',name,'--source',entry['source'],
                '--source-revision',entry['source_revision'],'--output',cfg['workspace']/'manifests'/name,
                '--overlap-policy',cfg['overlap_policy'])))
    if stage in ('all','stats','train'):
        print(shlex.join([sys.executable,str(REPO/'scripts/fetch_model.py'),'--cache',str(cfg['cache_dir']),'--revision',cfg['model_revision']]))
        for name in cfg['datasets']:
            print(shlex.join(cli('stats',*common(cfg,name),'--samples',cfg['stats_samples'],'--output',cfg['workspace']/'stats'/f'{name}.pt')))
    if stage in ('all','train'):
        for name in cfg['datasets']:
            for config in cfg['configs']:
                for seed in cfg['seeds']:
                    command=train_command(cfg,name,config,seed)
                    if stop is not None:command+=['--stop-after-steps',str(stop)]
                    print(shlex.join(command))
    if stage in ('all','analyze'):
        for name in cfg['datasets']:
            print(shlex.join(cli('analyze','--root',cfg['workspace']/'runs','--dataset',name,
                '--output',cfg['workspace']/'figures','--bootstrap-samples',2000)))


def manifest_valid(cfg,name):
    folder=cfg['workspace']/'manifests'/name
    meta=read_json(folder/'manifest.json')
    source=cfg['datasets'][name]
    if meta['dataset']!=name or meta['source_sha256']!=digest(source['source']) or meta['source_revision']!=source['source_revision']:
        raise ValueError(f'{name}: source differs from the frozen manifest')
    if meta.get('requested_pool_size')!=cfg['pool_size'] or meta.get('overlap_policy','error')!=cfg['overlap_policy']:
        raise ValueError(f'{name}: pool or overlap policy differs from frozen manifest')
    for split,sha in meta['split_hashes'].items():
        if digest(folder/f'{split}.jsonl')!=sha:
            raise ValueError(f'{name}: modified {split} manifest')
    return meta


def stats_valid(cfg,name):
    import torch
    path=cfg['workspace']/'stats'/f'{name}.pt'
    stats=torch.load(path,map_location='cpu',weights_only=True)
    meta=stats['metadata']
    manifest=manifest_valid(cfg,name)
    if meta['revision']!=cfg['model_revision'] or meta['dataset']!=name or meta['manifest_sha256']!=digest(cfg['workspace']/'manifests'/name/'manifest.json') or meta['input_limits']!=DEFAULTS:
        raise ValueError(f'{name}: masking statistics provenance mismatch')
    if len(meta['ids'])!=min(cfg['stats_samples'],manifest['counts']['train']):
        raise ValueError(f'{name}: masking statistics sample size mismatch')
    for modality in ('image','text'):
        mean,std=stats[modality]['mean'],stats[modality]['std']
        if mean.ndim!=1 or mean.shape!=std.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std<0).any():
            raise ValueError(f'{name}: invalid masking moments')
    return meta


def verify_checkpoint(path):
    marker=read_json(path/'complete.json')
    for name in ('trainable','state'):
        if digest(path/f'{name}.pt')!=marker[f'{name}_sha256']:
            raise ValueError(f'Corrupt checkpoint: {path}/{name}.pt')
    return marker


def verify_run_config(cfg,name,config,seed,directory):
    saved=read_json(directory/'config.json')
    expected=dict(masking_method=cfg.get('masking_method','symmetric-attention-v2'),**DEFAULTS,**CONFIGS[config],dataset=name,seed=seed,revision=cfg['model_revision'],
        data_manifest_sha256=digest(cfg['workspace']/'manifests'/name/'manifest.json'),
        stats_sha256=digest(cfg['workspace']/'stats'/f'{name}.pt'),source_sha256=core_hashes())
    if any(saved.get(k)!=v for k,v in expected.items()):
        raise ValueError(f'Run provenance differs: {directory}; use a fresh workspace')
    return saved


def complete_run(directory):
    if not (directory/'progress.json').is_file() or read_json(directory/'progress.json').get('status')!='complete':
        return False
    marker=verify_checkpoint(directory/'checkpoints'/'epoch_8')
    if marker['epoch']!=8 or marker['step']!=read_json(directory/'progress.json')['step']:
        raise ValueError('Final checkpoint and progress marker disagree')
    with (directory/'metrics.csv').open() as f:
        rows=list(csv.DictReader(f))
    expected={(e,c) for e in range(9) for c in CONDITIONS}
    actual={(int(r['epoch']),r['variant']) for r in rows if r['variant'] in CONDITIONS}
    n=27
    if len(rows)!=n or actual!=expected or len({(r['epoch'],r['variant']) for r in rows})!=n:
        raise ValueError(f'Incomplete/duplicate evaluation rows: {directory}')
    for row in rows:
        if not (directory/'predictions'/f'epoch_{row["epoch"]}_{row["variant"]}.jsonl').is_file():
            raise ValueError(f'Missing earlier predictions: {directory}; restore artifacts or branch from an earlier checkpoint')
    with (directory/'contributions.csv').open() as f:
        if {int(r['epoch']) for r in csv.DictReader(f)}!=set(range(9)):
            raise ValueError(f'Incomplete contributions: {directory}')
    return True


class Pipeline:
    def __init__(self,cfg,resume=False,stop=None):
        self.cfg,self.resume,self.stop=cfg,resume,stop
        self.root=cfg['workspace']
        self.fd=None
        self.profile=None

    def execute(self,command,label,offline=False):
        log=self.root/'logs'/f'{label}.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        atomic_json(self.root/'pipeline-status.json',dict(status='running',task=label,command=command,log=str(log)))
        print(f'\n[{label}] {shlex.join(command)}',flush=True)
        env=os.environ.copy()
        env['PYTHONUNBUFFERED']='1'
        env.setdefault('OMP_NUM_THREADS','1')
        if offline:
            env['HF_HUB_OFFLINE']='1'
        with log.open('a') as stream:
            stream.write(f'\n=== {timestamp()} ===\n{shlex.join(command)}\n');stream.flush()
            with subprocess.Popen(command,cwd=REPO,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                  text=True,bufsize=1,pass_fds=(self.fd,)) as child:
                for line in child.stdout:
                    print(line,end='',flush=True)
                    stream.write(line);stream.flush()
                code=child.wait()
        if code:
            raise RuntimeError(f'{label} failed with exit code {code}; see {log}')

    def initialize(self,stage):
        path=self.root/'pipeline.json'
        previous=read_json(path) if path.exists() else None
        def source_digest(name,entry):
            if entry['source'].is_file():
                return digest(entry['source'])
            if stage=='analyze' and previous is not None:
                return previous['datasets'][name]['source_sha256']
            raise ValueError(f'Missing source JSONL: {entry["source"]}')
        identity=dict(pool_size=self.cfg['pool_size'],overlap_policy=self.cfg['overlap_policy'],masking_method=self.cfg.get('masking_method','symmetric-attention-v2'),model_revision=self.cfg['model_revision'],stats_samples=self.cfg['stats_samples'],
            source_sha256=core_hashes(),
            datasets={name:dict(source=str(entry['source']),source_revision=entry['source_revision'],
                source_sha256=source_digest(name,entry)) for name,entry in self.cfg['datasets'].items()})
        if path.exists():
            if not self.resume:
                raise ValueError('Workspace already initialized; add --resume')
            if previous!=identity:
                raise ValueError('Frozen pipeline source/data/statistics settings changed; use a fresh workspace')
        else:
            if any(p.name!='.pipeline.lock' for p in self.root.iterdir()):
                raise ValueError('Refusing to adopt a nonempty workspace without pipeline.json')
            atomic_json(path,identity)
        with (self.root/'invocations.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=timestamp(),config=str(self.cfg['config_file']),
                configs=self.cfg['configs'],seeds=self.cfg['seeds'],resume=self.resume,
                checkpoint_every_steps=self.cfg['checkpoint_every_steps'],stop_after_steps=self.stop))+'\n')

    def quarantine(self,path):
        target=self.root/'recovery'/f'{timestamp()}-{path.name}'
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(path),target)
        print('Preserved interrupted preparation artifact:',target,flush=True)

    def dependent_runs_exist(self,name):
        return (self.root/'runs'/name).exists()

    def prepare(self):
        for name,entry in self.cfg['datasets'].items():
            folder=self.root/'manifests'/name
            if (folder/'manifest.json').is_file():
                try:
                    read_json(folder/'manifest.json')
                except (ValueError,OSError):
                    if self.dependent_runs_exist(name) or (self.root/'stats'/f'{name}.pt').exists():
                        raise ValueError('Broken manifest with dependent artifacts; restore the original manifest')
                    self.quarantine(folder)
                else:
                    manifest_valid(self.cfg,name)
                    print('Reuse frozen manifest:',folder,flush=True)
                    continue
            if folder.exists():
                if self.dependent_runs_exist(name) or (self.root/'stats'/f'{name}.pt').exists():
                    raise ValueError('Manifest is incomplete but dependent artifacts exist; restore original files')
                self.quarantine(folder)
            self.execute(cli('prepare','--dataset',name,'--source',entry['source'],
                '--source-revision',entry['source_revision'],'--output',folder,
                '--overlap-policy',self.cfg['overlap_policy']),f'prepare_{name}')
            manifest_valid(self.cfg,name)

    def gpu(self):
        command=[sys.executable,'-c',
            "import json,torch; assert torch.cuda.is_available(), 'CUDA unavailable: use CUDA PyTorch outside a device-hiding sandbox'; assert torch.cuda.is_bf16_supported(), 'BF16 unavailable'; print(json.dumps(dict(device=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda)))"]
        result=subprocess.run(command,cwd=REPO,text=True,capture_output=True)
        if result.returncode:
            raise RuntimeError('GPU preflight failed:\n'+result.stderr)
        self.profile=json.loads(result.stdout.strip().splitlines()[-1])
        print('GPU:',self.profile,flush=True)

    def model(self):
        folder=self.cfg['cache_dir']/('models--'+DEFAULTS['model'].replace('/','--'))/'snapshots'/self.cfg['model_revision']
        valid=False
        try:
            # The pinned loader uses the slow tokenizer; fast tokenizer.json alone is insufficient.
            required=['config.json','generation_config.json','preprocessor_config.json',
                      'tokenizer_config.json','vocab.json','merges.txt']
            if (folder/'model.safetensors.index.json').exists():
                required += list(set(read_json(folder/'model.safetensors.index.json')['weight_map'].values()))
            else:
                required += ['model.safetensors']
            valid=all((folder/name).is_file() and (folder/name).stat().st_size>0 for name in required)
        except (OSError,ValueError,KeyError,TypeError):
            pass
        if valid:
            print('Reuse cached immutable model:',folder,flush=True)
            return
        if os.environ.get('HF_HUB_OFFLINE','').lower() in ('1','true','yes'):
            raise ValueError('Model cache is incomplete while HF_HUB_OFFLINE is set; download first or unset it')
        self.execute([sys.executable,str(REPO/'scripts/fetch_model.py'),'--cache',str(self.cfg['cache_dir']),
            '--revision',self.cfg['model_revision']],'download_model')

    def stats(self):
        for name in self.cfg['datasets']:
            path=self.root/'stats'/f'{name}.pt'
            if path.exists():
                try:
                    metadata=stats_valid(self.cfg,name)
                    if not path.with_suffix('.json').is_file():
                        atomic_json(path.with_suffix('.json'),metadata)
                    print('Reuse frozen statistics:',path,flush=True)
                    continue
                except Exception as exc:
                    if self.dependent_runs_exist(name):
                        raise ValueError('Cannot replace statistics used by existing runs; restore original statistics') from exc
                    self.quarantine(path)
                    if path.with_suffix('.json').exists():
                        self.quarantine(path.with_suffix('.json'))
            self.execute(cli('stats',*common(self.cfg,name),'--samples',self.cfg['stats_samples'],'--output',path),f'stats_{name}',offline=True)
            stats_valid(self.cfg,name)

    def job(self,name,config,seed):
        directory=self.root/'runs'/name/config/f'seed_{seed}'
        command=train_command(self.cfg,name,config,seed)
        if directory.exists():
            checkpoints=directory/'checkpoints'
            published=checkpoints.exists() and any(not p.name.startswith('.') for p in checkpoints.iterdir())
            history=any((directory/n).exists() for n in ('metrics.csv','resources.json','progress.json'))
            if not published and not history:self.quarantine(directory)
        if directory.exists():
            verify_run_config(self.cfg,name,config,seed,directory)
            if complete_run(directory):
                print('Skip completed training:',directory,flush=True);return True
            command+=['--resume-from','latest']
        if self.stop is not None:command+=['--stop-after-steps',str(self.stop)]
        self.execute(command,f'train_{name}_{config}_seed_{seed}',offline=True)
        if not complete_run(directory):
            status=read_json(directory/'progress.json')
            atomic_json(self.root/'pipeline-status.json',dict(status='incomplete',task=f'{name}/{config}/seed_{seed}',checkpoint=status['checkpoint']))
            print('Saved checkpoint. Repeat with --resume and without --stop-after-steps.',flush=True);return False
        return True

    def train(self):
        for name in self.cfg['datasets']:
            for config in self.cfg['configs']:
                for seed in self.cfg['seeds']:
                    if not self.job(name,config,seed):return False
        return True

    def analyze(self):
        for name in self.cfg['datasets']:
            for config in self.cfg['configs']:
                for seed in self.cfg['seeds']:
                    path=self.root/'runs'/name/config/f'seed_{seed}'
                    if not complete_run(path):raise ValueError(f'Incomplete full matrix: {path}')
            self.execute(cli('analyze','--root',self.root/'runs','--dataset',name,
                '--output',self.root/'figures','--bootstrap-samples',2000),f'analyze_{name}')

    def run(self,stage):
        with workspace_lock(self.root) as self.fd:
            self.initialize(stage)
            try:
                if stage!='analyze':
                    if stage!='prepare':
                        self.gpu()
                    self.prepare()
                if stage in ('all','stats','train'):
                    self.model(); self.stats()
                if stage in ('all','train') and not self.train():
                    return
                if stage in ('all','analyze'):
                    self.analyze()
                atomic_json(self.root/'pipeline-status.json',dict(status='complete',stage=stage,time=timestamp()))
                print('Requested stage complete. Outputs:',self.root,flush=True)
            except BaseException as exc:
                atomic_json(self.root/'pipeline-status.json',dict(status='failed',stage=stage,error=str(exc),time=timestamp()))
                raise


def main():
    p=argparse.ArgumentParser(description='Full-dataset training, exact resume, and analysis')
    p.add_argument('--config',required=True,help='JSON configuration; paths resolve relative to this file')
    p.add_argument('--stage',choices=STAGES,default='all')
    p.add_argument('--resume',action='store_true',help='Reuse the frozen workspace; skip completed runs and resume incomplete ones')
    p.add_argument('--dry-run',action='store_true',help='Print the plan without writing files, downloading or accessing CUDA')
    p.add_argument('--stop-after-steps',type=int,default=None,help='Gracefully stop the first unfinished full-experiment run at this absolute optimizer step')
    args=p.parse_args()
    try:
        cfg=load_config(args.config)
        if args.stop_after_steps is not None and args.stop_after_steps<=0:
            raise ValueError('stop-after-steps must be positive')
        if args.dry_run:
            preview(cfg,args.stage,args.resume,args.stop_after_steps)
        else:
            Pipeline(cfg,args.resume,args.stop_after_steps).run(args.stage)
    except (ValueError,RuntimeError,OSError,KeyError) as exc:
        p.exit(1,f'ERROR: {exc}\n')


if __name__=='__main__':
    main()
