"""Atomic optimizer-boundary snapshots and complete RNG restoration."""
import os
import random
import shutil
import tempfile
from pathlib import Path

from .protocol import digest, write_json


def capture_rng():
    import numpy as np
    import torch
    return dict(python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    import numpy as np
    import torch
    cuda = state['cuda_rng']
    if len(cuda) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError('CUDA device count differs from checkpoint RNG state')
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'].cpu())
    if cuda:
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda])


def save(model, optimizer, scheduler, output, epoch, step, processor,
         trainer=None, signature=None):
    import torch
    output = Path(output)
    if output.exists():
        raise ValueError(f'Checkpoint already exists: {output}')
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('Save only after optimizer.step and zero_grad(set_to_none=True)')
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = capture_rng()
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
    try:
        torch.save({n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},
                   staging/'trainable.pt')
        torch.save(dict(format_version=2, epoch=epoch, step=step,
                        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        trainer=trainer, signature=signature, **rng), staging/'state.pt')
        processor.save_pretrained(staging/'processor')
        write_json(staging/'complete.json', dict(format_version=2, epoch=epoch, step=step,
            trainable_sha256=digest(staging/'trainable.pt'), state_sha256=digest(staging/'state.pt')))
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        # Saving, including processor serialization, cannot perturb dropout streams.
        restore_rng(rng)


def read(path):
    import json
    import torch
    path = Path(path)
    complete = json.loads((path/'complete.json').read_text())
    for name in ('trainable', 'state'):
        if digest(path/f'{name}.pt') != complete[f'{name}_sha256']:
            raise ValueError(f'Corrupt checkpoint {name}: {path}')
    weights = torch.load(path/'trainable.pt', map_location='cpu', weights_only=True)
    # Local training states contain NumPy RNG objects as well as tensors.
    state = torch.load(path/'state.pt', map_location='cpu', weights_only=False)
    version = state.get('format_version',1)
    if version not in (1,2) or complete.get('format_version',1) != version:
        raise ValueError('Unsupported or inconsistent checkpoint format')
    if (state['epoch'], state['step']) != (complete['epoch'], complete['step']):
        raise ValueError('Checkpoint completion marker disagrees with saved state')
    return weights, state


def latest(directory):
    import json
    candidates = []
    for path in Path(directory).glob('*'):
        if path.is_dir() and not path.name.startswith('.') and (path/'complete.json').is_file():
            try:
                mark = json.loads((path/'complete.json').read_text())
                candidates.append((int(mark['step']), int(mark['epoch']), path.name, path))
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f'Invalid checkpoint completion marker: {path}') from exc
    if not candidates:
        raise ValueError('No completed checkpoint found')
    # Verify the newest committed snapshot; never silently roll back a corrupt one.
    path = max(candidates)[-1]
    read(path)
    return path


def prune_steps(directory, keep):
    """Retain newest complete step snapshots; never prune epoch snapshots."""
    import json
    if type(keep) is not int or keep < 0:raise ValueError('keep must be nonnegative')
    if not keep:return
    steps=[]
    for path in Path(directory).glob('step_*'):
        if not path.is_dir() or not (path/'complete.json').is_file():continue
        marker=json.loads((path/'complete.json').read_text())
        if path.name != f"step_{int(marker['step'])}":raise ValueError('Step checkpoint marker/name mismatch')
        steps.append((int(marker['step']),path))
    for _,path in sorted(steps)[:-keep]:shutil.rmtree(path)
