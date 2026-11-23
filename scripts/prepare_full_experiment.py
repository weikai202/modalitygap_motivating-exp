"""Export full releases and generate ready-to-run screening/confirmation configurations."""
import argparse,json
from pathlib import Path
from modalitygap.full_data import mmsd2,mmimdb
from modalitygap.protocol import CONFIGS,MODEL_REVISION,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True,help='inputs.json from download_full_data.py');p.add_argument('--output',default='data/full-export-v1');p.add_argument('--workspace',default='artifacts/full-study-v1');p.add_argument('--pool-size',type=lambda x:'all' if x=='all' else int(x),default=6000);a=p.parse_args()
    if a.pool_size!='all' and a.pool_size<=0:p.error('pool-size must be positive or all')
    inputs=json.loads(Path(a.inputs).read_text());output=Path(a.output).resolve()
    if output.exists():raise ValueError('Output already exists; use a new immutable export directory')
    if set(inputs)!={'mmsd2','mmimdb'}:raise ValueError('Full experiment requires both dataset inputs')
    mmsd2(inputs['mmsd2']['annotations'],output/'mmsd2',parquet=inputs['mmsd2']['parquet'])
    mmimdb(inputs['mmimdb']['archive'],output/'mmimdb')
    sources={}
    for task in ('mmsd2','mmimdb'):
        provenance=json.loads((output/task/'provenance.json').read_text())
        sources[task]=dict(source=str(output/task/'source.jsonl'),source_revision=provenance['source_revision'])
    config=dict(workspace=str(Path(a.workspace).resolve()),cache_dir=str(Path('artifacts/hf-cache').resolve()),model_revision=MODEL_REVISION,
        masking_method='symmetric-attention-v2',pool_size=a.pool_size,overlap_policy='prefer-heldout',
        configs=list(CONFIGS),stats_samples=256,pilot_steps=4,checkpoint_every_steps=50,keep_step_checkpoints=2,datasets=sources)
    for name,seeds in [('screen',[42]),('confirm',[42,43,44])]:write_json(output/f'{name}.json',dict(config,seeds=seeds))
    print('Screen:',output/'screen.json');print('Confirm (same workspace, --resume):',output/'confirm.json')


if __name__=='__main__':main()
