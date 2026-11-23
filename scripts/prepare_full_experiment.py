"""Export both full releases and generate the single full-study configuration."""
import argparse,json
from pathlib import Path
from modalitygap.full_data import mmsd2,mmimdb
from modalitygap.protocol import CONFIGS,MODEL_REVISION,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True,help='inputs.json from download_full_data.py');p.add_argument('--output',default='data/full-export-v1');p.add_argument('--workspace',default='artifacts/full-study-v1');a=p.parse_args()
    inputs=json.loads(Path(a.inputs).read_text());output=Path(a.output).resolve()
    if output.exists():raise ValueError('Output exists; choose a new immutable export directory')
    if set(inputs)!={'mmsd2','mmimdb'}:raise ValueError('Both dataset inputs are required')
    mmsd2(inputs['mmsd2']['annotations'],output/'mmsd2',parquet=inputs['mmsd2']['parquet'])
    mmimdb(inputs['mmimdb']['archive'],output/'mmimdb')
    sources={task:dict(source=str(output/task/'source.jsonl'),source_revision=json.loads((output/task/'provenance.json').read_text())['source_revision']) for task in ('mmsd2','mmimdb')}
    config=dict(workspace=str(Path(a.workspace).resolve()),cache_dir=str(Path('artifacts/hf-cache').resolve()),model_revision=MODEL_REVISION,
        masking_method='symmetric-attention-v2',pool_size='all',overlap_policy='prefer-heldout',configs=list(CONFIGS),seeds=[42,43,44],
        stats_samples=256,checkpoint_every_steps=50,keep_step_checkpoints=2,datasets=sources)
    write_json(output/'experiment.json',config);print(output/'experiment.json')


if __name__=='__main__':main()
