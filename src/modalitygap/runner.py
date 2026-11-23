from .protocol import generation_limit
import csv
import importlib.metadata
import json
import math
import os
import platform
import random
import time
from pathlib import Path

# Must be configured before the first CUDA BLAS operation in this process.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from .model import Encoder, Intervention, adapt, load_model
from .protocol import (CONDITIONS, CONFIGS, DEFAULTS, contributions, digest,
                       metrics, parse, read_jsonl, write_json, write_jsonl)


def to_device(batch):
    return {k:v.cuda() for k,v in batch.items()}


def sync_time():
    import torch
    torch.cuda.synchronize()
    return time.perf_counter()


def verified_data(path, dataset):
    path = Path(path)
    metadata = json.loads((path/'manifest.json').read_text())
    if metadata['dataset'] != dataset:
        raise ValueError('Dataset does not match manifest')
    for split, sha in metadata['split_hashes'].items():
        if digest(path/f'{split}.jsonl') != sha:
            raise ValueError(f'Manifest changed: {split}')
    return metadata


def estimate(args):
    import torch
    verified_data(args.data, args.dataset)
    output = Path(args.output)
    if output.exists():
        raise ValueError('Refusing to overwrite frozen masking statistics')
    model, processor = load_model(args.revision, cache_dir=getattr(args, 'cache_dir', None))
    model.eval()
    encoder, hook = Encoder(processor, args.dataset), Intervention(model)
    rows = read_jsonl(Path(args.data)/'train.jsonl')
    rows = random.Random(2026).sample(rows, min(args.samples, len(rows)))
    moments = {m:[] for m in ('image', 'text')}
    try:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for row in rows:
                inputs, positions, _ = encoder(row)
                inputs = to_device(inputs)
                with hook.use(row, inputs, positions, capture=True):
                    model(**inputs, use_cache=False)
                for modality, x in hook.observed.items():
                    if not len(x):
                        raise ValueError('Empty body embeddings cannot estimate text moments')
                    moments[modality].append((x.double().mean(0), x.double().square().mean(0)))
    finally:
        hook.close()
    stats = {}
    for modality, values in moments.items():
        mean = torch.stack([v[0] for v in values]).mean(0)
        second = torch.stack([v[1] for v in values]).mean(0)
        stats[modality] = dict(mean=mean.float(), std=(second-mean.square()).clamp_min(0).sqrt().float())
    stats['metadata'] = dict(ids=[r['id'] for r in rows], revision=args.revision,
                             dataset=args.dataset, manifest_sha256=digest(Path(args.data)/'manifest.json'),
                             seed=2026, estimator='E_example[E_token[x]], sqrt(E_example[E_token[x^2]] - mu^2); population std',
                             input_limits=DEFAULTS)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, output)
    write_json(output.with_suffix('.json'), stats['metadata'])


def decode_prediction(tokenizer, dataset, generated):
    """Count unmapped vocabulary IDs as invalid outputs instead of crashing a run."""
    ids = generated.detach().cpu().tolist() if hasattr(generated, 'detach') else list(generated)
    content = list(ids)
    terminators = {getattr(tokenizer, 'eos_token_id', None), getattr(tokenizer, 'pad_token_id', None)} - {None}
    while content and content[-1] in terminators:
        content.pop()
    try:
        # Unexpected structure tokens must remain visible to strict parsing.
        raw = tokenizer.decode(content, skip_special_tokens=False)
    except (TypeError, ValueError, OverflowError) as exc:
        return '<decoding_error:' + json.dumps(ids) + '>', [], True, type(exc).__name__
    prediction, invalid = parse(dataset, raw)
    return raw, prediction, invalid, None


def make_intervention(model,stats,method):
    if method=='symmetric-attention-v2':
        from .symmetric_mask import SymmetricMask
        hook=SymmetricMask(model,stats)
        hook.symmetric_protocol=True
        return hook
    if method!='mean-noise-v1':raise ValueError('Unknown masking method')
    return Intervention(model,stats)


def evaluate(model, encoder, hook, rows, dataset, directory, epoch, seed, config,
             step, training_seconds, sensitivity=True):
    import torch
    model.eval()
    results, main_scores = [], {}
    variants = [(c, 2026, .05, c) for c in CONDITIONS]
    symmetric = getattr(hook, "symmetric_protocol", False)
    if sensitivity and not symmetric and epoch in (0,4,8):
        variants += [(c, ns, scale, f'{c}_noise_{ns}_scale_{scale}')
                     for c in CONDITIONS[1:] for ns,scale in ((2027,.05),(2028,.05),(2026,0.))]
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for condition, noise_seed, scale, suffix in variants:
            predictions = []
            start = sync_time()
            for row in rows:
                inputs, positions, truncated = encoder(row)
                inputs = to_device(inputs)
                if symmetric:
                    from transformers import LogitsProcessorList
                    with hook.use(inputs, positions, condition) as penalty:
                        output = model.generate(**inputs, do_sample=False, use_cache=True,
                            repetition_penalty=1.,logits_processor=LogitsProcessorList([penalty]),
                            max_new_tokens=generation_limit(dataset))
                else:
                    with hook.use(row, inputs, positions, condition, noise_seed, scale):
                        output = model.generate(**inputs, do_sample=False, use_cache=True,
                                                max_new_tokens=generation_limit(dataset))
                generated = output[0, inputs['input_ids'].shape[1]:]
                raw, prediction, invalid, decode_error = decode_prediction(encoder.processor.tokenizer,dataset,generated)
                predictions.append(dict(dataset=dataset, config=config, training_seed=seed, epoch=epoch,
                    step=step, id=row['id'], group=row.get('group',row['id']), condition=condition,
                    noise_seed=None if symmetric else noise_seed, noise_scale=0. if symmetric else scale, reference=row['labels'], raw_output=raw,
                    prediction=prediction, invalid=invalid, decode_error=decode_error, generated_token_ids=generated.detach().cpu().tolist(),
                    generated_length=len(generated), truncated=truncated))
            elapsed = sync_time()-start
            score = metrics(dataset, predictions)
            record = dict(epoch=epoch, condition=condition, variant=suffix, seed=seed, config=config,
                          training_seconds=training_seconds, validation_seconds=elapsed, **score)
            results.append(record)
            write_jsonl(directory/'predictions'/f'epoch_{epoch}_{suffix}.jsonl', predictions)
            if suffix==condition:
                main_scores[condition] = score['score']
    write_jsonl(directory/'interventions'/f'epoch_{epoch}.jsonl', hook.logs)
    hook.logs.clear()
    return results, dict(epoch=epoch, seed=seed, config=config, **contributions(main_scores))


def save_checkpoint(model, optimizer, scheduler, output, epoch, step, processor,
                    trainer=None, signature=None):
    from .checkpoint import save
    save(model, optimizer, scheduler, output, epoch, step, processor, trainer, signature)


def load_trainable(model, state):
    """Reject missing, extra, malformed, or nonfinite trainable checkpoint tensors."""
    import torch
    expected = {n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(state) != set(expected):
        raise ValueError('Checkpoint trainable keys do not match current adaptation scope')
    for name, value in state.items():
        if value.shape != expected[name].shape or not torch.isfinite(value).all():
            raise ValueError(f'Invalid checkpoint tensor: {name}')
    model.load_state_dict(state, strict=False)


def write_csv(path, records):
    if not records:
        return
    keys = list(dict.fromkeys(k for r in records for k in r))
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in records)


def configure_reproducibility():
    import torch
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
        raise ValueError('Exact resume requires a deterministic CUBLAS_WORKSPACE_CONFIG')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')


def _csv_records(path):
    if not path.exists():
        return []
    with path.open() as f:
        rows = list(csv.DictReader(f))
    def decode(value):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return [{k:decode(v) for k,v in row.items()} for row in rows]


def _resume_signature(args, trainable):
    import os
    import torch
    return dict(**DEFAULTS, **CONFIGS[args.config], dataset=args.dataset, seed=args.seed,
        revision=args.revision, pilot_steps=args.pilot_steps,
        masking_method=getattr(args,'masking_method','mean-noise-v1'),
        manifest_sha256=digest(Path(args.data)/'manifest.json'), stats_sha256=digest(args.stats),
        trainable=[list(row) for row in trainable],
        source_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
        device_count=torch.cuda.device_count(), cudnn=torch.backends.cudnn.version(),
        software={p:importlib.metadata.version(p) for p in ('transformers','peft','numpy','Pillow','torchvision')},
        deterministic=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
        tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
        matmul_precision=torch.get_float32_matmul_precision(),
        cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


def _legacy_progress(state, donor, train_count):
    epoch = state['epoch']
    if not 0 <= epoch <= 8 or state['step'] != epoch*math.ceil(train_count/16):
        raise ValueError('Legacy checkpoint is not a complete epoch boundary')
    resources = [r for r in json.loads((donor/'resources.json').read_text()) if r['epoch']<=epoch]
    if len(resources) != epoch:
        raise ValueError('Legacy checkpoint needs its epoch resource history')
    return dict(epoch=epoch, next_index=train_count if epoch else 0, tracker=None,
        training_seconds=sum(r['seconds'] for r in resources), resources=resources,
        metrics=[r for r in _csv_records(donor/'metrics.csv') if r['epoch']<epoch],
        contributions=[r for r in _csv_records(donor/'contributions.csv') if r['epoch']<epoch])


def _copy_resume_prefix(donor, directory, state, progress):
    import shutil
    directory.mkdir(parents=True, exist_ok=False)
    for folder in ('predictions', 'interventions'):
        if not (donor/folder).exists():
            continue
        for source in (donor/folder).iterdir():
            if source.is_file() and source.name.startswith('epoch_'):
                epoch = int(source.name.split('_')[1].split('.')[0])
                if epoch < progress['epoch']:
                    target = directory/folder/source.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
    for source in (donor/'checkpoints').iterdir():
        if not source.name.startswith('.') and (source/'complete.json').is_file():
            marker = json.loads((source/'complete.json').read_text())
            if marker['step'] <= state['step']:
                shutil.copytree(source, directory/'checkpoints'/source.name)


def run(args):
    import torch
    from transformers import get_cosine_schedule_with_warmup, set_seed
    from .checkpoint import latest, read, restore_rng
    configure_reproducibility()
    meta = verified_data(args.data, args.dataset)
    stats = torch.load(args.stats, map_location='cpu', weights_only=True)
    if stats['metadata']['revision'] != args.revision or stats['metadata']['manifest_sha256'] != digest(Path(args.data)/'manifest.json') or stats['metadata']['input_limits'] != DEFAULTS:
        raise ValueError('Statistics must match model revision and immutable training manifest')
    keep=getattr(args,'keep_step_checkpoints',0)
    if type(keep) is not int or keep<0:raise ValueError('Invalid step checkpoint retention')
    interval = getattr(args, 'checkpoint_every_steps', 0)
    stop = getattr(args, 'stop_after_steps', None)
    resume = getattr(args, 'resume_from', None)
    if interval < 0 or (stop is not None and stop <= 0):
        raise ValueError('Invalid checkpoint interval or stop step')
    method=getattr(args,'masking_method','mean-noise-v1')
    if method not in ('mean-noise-v1','symmetric-attention-v2'):raise ValueError('Unknown masking method')
    config = CONFIGS[args.config]
    directory = Path(args.output)/args.dataset/args.config/f'seed_{args.seed}'
    if args.pilot_steps:
        directory = directory.parent/f'seed_{args.seed}_pilot'
    checkpoint, state, weights, donor, original = None, None, None, None, None
    if resume:
        checkpoint = latest(directory/'checkpoints') if resume == 'latest' else Path(resume).resolve()
        weights, state = read(checkpoint)
        donor = checkpoint.parent.parent
        original = json.loads((donor/'config.json').read_text())
        if state.get('format_version',1) != 2 and not getattr(args,'allow_legacy_resume',False):
            raise ValueError('Legacy checkpoint lacks exact-resume metadata; --allow-legacy-resume permits continuation without a bitwise-equivalence guarantee')
        expected = dict(**DEFAULTS, **config, dataset=args.dataset, seed=args.seed,
            revision=args.revision, pilot_steps=args.pilot_steps,
            data_manifest_sha256=digest(Path(args.data)/'manifest.json'), stats_sha256=digest(args.stats))
        if original.get('masking_method','mean-noise-v1') != method:
            raise ValueError('Resume masking protocol differs')
        if any(original.get(k) != v for k,v in expected.items()):
            raise ValueError('Resume configuration, data, statistics or precision policy differs')
        if stop is not None and stop <= state['step']:
            raise ValueError('stop-after-steps must exceed the restored global step')
        if directory.exists():
            if directory.resolve() != donor.resolve():
                raise ValueError('Resume destination already exists and is a different run')
            for marker in (directory/'checkpoints').glob('*/complete.json'):
                if marker.parent.name.startswith('.'):
                    continue
                if json.loads(marker.read_text())['step'] > state['step']:
                    raise ValueError('Newer checkpoints exist; resume latest or use a new output root')
    elif directory.exists():
        raise ValueError('Output already exists; use --resume-from latest or a new output root')
    set_seed(args.seed)
    model, processor = load_model(args.revision, cache_dir=getattr(args, 'cache_dir', None))
    model, trainable = adapt(model, config['scope'], args.seed)
    encoder = Encoder(processor,args.dataset)
    hook = make_intervention(model,stats,method)
    try:
        signature = _resume_signature(args, trainable)
        train = read_jsonl(Path(args.data)/'train.jsonl')
        dev = read_jsonl(Path(args.data)/'dev.jsonl')
        if args.pilot_steps:
            train, dev = train[:64], dev[:32]
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=config['lr'],weight_decay=.01)
        steps_per_epoch = math.ceil(len(train)/16)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
            num_warmup_steps=math.ceil(steps_per_epoch*8*.05),num_training_steps=steps_per_epoch*8)
        progress = dict(epoch=0, next_index=0, tracker=None, training_seconds=0.,
                        metrics=[], contributions=[], resources=[])
        step = 0
        if state is not None:
            if state.get('format_version') == 2:
                if state.get('signature') != signature or not isinstance(state.get('trainer'), dict):
                    raise ValueError('Resume source code, model layout or runtime differs from checkpoint')
                progress = state['trainer']
            else:
                old_software = {k.lower().replace('_','-'):v for k,v in original.get('software',{}).items()}
                incompatible = any(old_software.get(k.lower().replace('_','-')) != v for k,v in signature['software'].items())
                if incompatible or original.get('cuda_runtime') != torch.version.cuda or original.get('torch_runtime') != torch.__version__ or original.get('device') != torch.cuda.get_device_name():
                    raise ValueError('Legacy resume requires the original PyTorch and GPU type')
                progress = _legacy_progress(state, donor, len(train))
            epoch, cursor = progress['epoch'], progress['next_index']
            if not 0 <= epoch <= 8 or not 0 <= cursor <= len(train) or (epoch==0 and cursor!=0):
                raise ValueError('Invalid checkpoint epoch or data cursor')
            if cursor != len(train) and cursor % 16:
                raise ValueError('Checkpoint is not at an optimizer-step boundary')
            expected_step = 0 if epoch==0 else (epoch-1)*steps_per_epoch + math.ceil(cursor/16)
            if state['step'] != expected_step:
                raise ValueError('Checkpoint step disagrees with data cursor')
            if 0 < cursor < len(train) and (not progress['tracker'] or progress['tracker']['examples'] != cursor):
                raise ValueError('Checkpoint is missing partial-epoch accounting')
            load_trainable(model, weights)
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            step = state['step']
            if scheduler.last_epoch != step:
                raise ValueError('Scheduler step disagrees with checkpoint global step')
            if not directory.exists():
                _copy_resume_prefix(donor, directory, state, progress)
        record = dict(**DEFAULTS, **config, masking_method=method, seed=args.seed, dataset=args.dataset,
            revision=args.revision, data_manifest_sha256=digest(Path(args.data)/'manifest.json'),
            trainable=trainable, trainable_count=sum(n for _,n in trainable), pilot_steps=args.pilot_steps,
            data_manifest=meta, stats_sha256=digest(args.stats),
            software={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
            device=torch.cuda.get_device_name(), torch_runtime=torch.__version__, cuda_runtime=torch.version.cuda,
            source_revision=args.code_revision, source_sha256=signature['source_sha256'],
            checkpoint_format=2, deterministic_resume=True, resume_history=(original or {}).get('resume_history', []))
        if checkpoint:
            record['resume_history'] = record['resume_history'] + [dict(checkpoint=str(checkpoint),step=step,legacy=state.get('format_version',1)!=2)]
        write_json(directory/'config.json', record)
        model.eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            inp, positions, _ = encoder(dev[0])
            inp = to_device(inp)
            plain = model(**inp, use_cache=False).logits
            if method=='symmetric-attention-v2':
                with hook.use(inp, positions, 'full'):
                    active = model(**inp, use_cache=False).logits
            else:
                with hook.use(dev[0], inp, positions):
                    active = model(**inp, use_cache=False).logits
            if not torch.equal(plain, active):
                raise RuntimeError('Inactive hook changed full-input logits')
            del plain, active, inp
        optimizer.zero_grad(set_to_none=True)
        # Constructors and sanity checks may consume randomness. Restore LAST.
        if state is not None:
            restore_rng(state)
        def commit(path):
            save_checkpoint(model, optimizer, scheduler, path, progress['epoch'], step,
                            processor, trainer=progress, signature=signature)
            from .checkpoint import prune_steps
            prune_steps(directory/'checkpoints',getattr(args,'keep_step_checkpoints',0))
            return path
        def stopped(path):
            write_json(directory/'progress.json', dict(status='incomplete',step=step,
                epoch=progress['epoch'],next_index=progress['next_index'],checkpoint=str(path)))
            return path
        while progress['epoch'] <= 8:
            epoch = progress['epoch']
            if epoch and progress['next_index'] < len(train):
                model.train()
                torch.cuda.reset_peak_memory_stats()
                start = sync_time()
                order = list(range(len(train)))
                random.Random(args.seed+epoch).shuffle(order)
                tracker = progress['tracker'] or dict(loss_sum=0.,examples=0,length_sum=0,max_length=0,
                    truncated_count=0,elapsed_seconds=0.,allocated_bytes=0,reserved_bytes=0,last_grad_norm=0.)
                progress['tracker'] = tracker
                for index in range(progress['next_index'],len(order)):
                    inputs, _, truncated = encoder(train[order[index]], training=True)
                    inputs = to_device(inputs)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        output = model(**inputs, use_cache=False)
                    loss = output.loss
                    if not torch.isfinite(loss):
                        raise RuntimeError('Non-finite training loss')
                    group_size = min(16, len(order)-(index//16)*16)
                    (loss/group_size).backward()
                    tracker['loss_sum'] += loss.detach().item()
                    tracker['examples'] += 1
                    tracker['length_sum'] += inputs['input_ids'].shape[1]
                    tracker['max_length'] = max(tracker['max_length'],inputs['input_ids'].shape[1])
                    tracker['truncated_count'] += truncated
                    if (index+1)%16==0 or index+1==len(order):
                        grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        step += 1
                        progress['next_index'] = index+1
                        tracker['last_grad_norm'] = grad.item()
                        tracker['elapsed_seconds'] += sync_time()-start
                        tracker['allocated_bytes'] = max(tracker['allocated_bytes'],torch.cuda.max_memory_allocated())
                        tracker['reserved_bytes'] = max(tracker['reserved_bytes'],torch.cuda.max_memory_reserved())
                        must_stop = (stop is not None and step>=stop) or (args.pilot_steps and step>=args.pilot_steps)
                        if progress['next_index'] < len(train) and (must_stop or (interval and step%interval==0)):
                            path = commit(directory/'checkpoints'/f'step_{step}')
                            if must_stop:
                                return stopped(path)
                        start = sync_time()
                seconds = tracker['elapsed_seconds']
                progress['training_seconds'] += seconds
                progress['resources'].append(dict(epoch=epoch,step=step,seconds=seconds,
                    examples_per_second=tracker['examples']/seconds,mean_loss=tracker['loss_sum']/tracker['examples'],
                    mean_length=tracker['length_sum']/tracker['examples'],max_length=tracker['max_length'],
                    truncation_rate=tracker['truncated_count']/tracker['examples'],last_grad_norm=tracker['last_grad_norm'],
                    allocated_bytes=tracker['allocated_bytes'],reserved_bytes=tracker['reserved_bytes']))
                progress['tracker'] = None
            path = directory/'checkpoints'/f'epoch_{epoch}'
            if not path.exists():
                commit(path)
            # A deliberate boundary stop leaves evaluation pending in the snapshot.
            if stop is not None and step>=stop:
                return stopped(path)
            values, derived = evaluate(model,encoder,hook,dev,args.dataset,directory,epoch,args.seed,
                args.config,step,progress['training_seconds'],sensitivity=not args.pilot_steps)
            progress['metrics'].extend(values)
            progress['contributions'].append(derived)
            write_csv(directory/'metrics.csv',progress['metrics'])
            write_csv(directory/'contributions.csv',progress['contributions'])
            write_json(directory/'resources.json',progress['resources'])
            if args.pilot_steps and step>=args.pilot_steps:
                return stopped(path)
            progress['epoch'] += 1
            progress['next_index'] = 0
        write_json(directory/'progress.json',dict(status='complete',epoch=8,step=step,checkpoint=str(path)))
        return directory
    finally:
        hook.close()

def heldout(args):
    """Evaluate a locked comparison list, never sweep test epochs."""
    import torch
    selection = json.loads(Path(args.selection).read_text())
    if not selection.get('locked_at') or not selection.get('comparisons'):
        raise ValueError('Prespecify locked_at and comparisons before test evaluation')
    for item in selection['comparisons']:
        if item['epoch'] not in (2,8):
            raise ValueError('Held-out evaluation allows only epoch 2 or 8')
        run_dir = Path(item['run'])
        saved = json.loads((run_dir/'config.json').read_text())
        verified_data(args.data,saved['dataset'])
        if item['config'] not in CONFIGS or CONFIGS[item['config']]['scope'] != saved['scope'] or CONFIGS[item['config']]['lr'] != saved['lr']:
            raise ValueError('Selection config does not match run')
        if digest(Path(args.data)/'manifest.json') != saved['data_manifest_sha256']:
            raise ValueError('Held-out data manifest differs from training')
        destination = Path(args.output)/saved['dataset']/item['config']/f'seed_{saved["seed"]}'/f'epoch_{item["epoch"]}'
        if destination.exists():
            raise ValueError('Held-out output exists')
        model, processor = load_model(saved['revision'], cache_dir=getattr(args, 'cache_dir', None))
        model,_ = adapt(model,saved['scope'],saved['seed'])
        checkpoint = run_dir/'checkpoints'/f'epoch_{item["epoch"]}'
        check = json.loads((checkpoint/'complete.json').read_text())
        if digest(checkpoint/'trainable.pt') != check['trainable_sha256']:
            raise ValueError('Corrupt checkpoint')
        state = torch.load(checkpoint/'trainable.pt',map_location='cpu',weights_only=True)
        expected = {n for n,p in model.named_parameters() if p.requires_grad}
        if set(state) != expected:
            raise ValueError('Checkpoint trainable keys do not match')
        load_trainable(model, state)
        stats = torch.load(args.stats,map_location='cpu',weights_only=True)
        if digest(args.stats) != saved['stats_sha256']:
            raise ValueError('Masking statistics differ')
        hook = make_intervention(model,stats,saved.get('masking_method','mean-noise-v1'))
        try:
            values,derived = evaluate(model,Encoder(processor,saved['dataset']),hook,
                read_jsonl(Path(args.data)/'test.jsonl'),saved['dataset'],destination,item['epoch'],
                saved['seed'],item['config'],0,0,sensitivity=False)
            write_csv(destination/'metrics.csv',values)
            write_csv(destination/'contributions.csv',[derived])
            write_json(destination/'selection.json',selection)
        finally:
            hook.close()
        del model
        torch.cuda.empty_cache()
