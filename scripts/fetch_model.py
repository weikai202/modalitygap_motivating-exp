"""Resolve and download an immutable snapshot into a workspace-owned cache."""
import argparse
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

p = argparse.ArgumentParser()
p.add_argument('--cache',default='artifacts/hf-cache')
p.add_argument('--processor-only',action='store_true')
p.add_argument('--revision',default='66285546d2b821cf421d4f5eb2576359d3770cd3')
a = p.parse_args()
repo = 'Qwen/Qwen2.5-VL-3B-Instruct'
if len(a.revision) != 40 or any(c not in '0123456789abcdef' for c in a.revision):
    raise ValueError('Use an immutable 40-character model commit SHA')
sha = HfApi().model_info(repo,revision=a.revision).sha
path = snapshot_download(repo,revision=sha,cache_dir=a.cache,
                         allow_patterns=['*.json','*.txt','*.jinja'] if a.processor_only else None)
Path(a.cache).mkdir(parents=True,exist_ok=True)
Path(a.cache,'model-revision.txt').write_text(sha+'\n')
print('REVISION='+sha,flush=True)
print('SNAPSHOT='+path,flush=True)
