"""Replay strict decoding and scoring against saved token IDs without GPU inference."""
import argparse
import csv
import json
import math
from pathlib import Path

from transformers import AutoTokenizer
from modalitygap.protocol import read_jsonl,metrics,write_json
from modalitygap.runner import decode_prediction


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--processor',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    tokenizer=AutoTokenizer.from_pretrained(a.processor,use_fast=False,local_files_only=True)
    count,files=0,0
    metric_cache={}
    for path in sorted(Path(a.root).rglob('predictions/*.jsonl')):
        rows=read_jsonl(path)
        if not rows:
            raise AssertionError(f'Empty prediction file: {path}')
        for row in rows:
            raw,prediction,invalid,error=decode_prediction(tokenizer,row['dataset'],row['generated_token_ids'])
            if (raw,prediction,invalid,error)!=(row['raw_output'],row['prediction'],row['invalid'],row.get('decode_error')):
                raise AssertionError(f'Strict decoding differs: {path}, {row["id"]}')
            if len(row['generated_token_ids'])!=row['generated_length']:
                raise AssertionError(f'Generated length mismatch: {path}')
            count+=1
        metric_path=path.parent.parent/'metrics.csv'
        if metric_path not in metric_cache:
            with metric_path.open() as f:
                metric_cache[metric_path]={(int(r['epoch']),r['variant']):r for r in csv.DictReader(f)}
        epoch=rows[0]['epoch']
        variant=path.stem[len(f'epoch_{epoch}_'):]
        saved=metric_cache[metric_path][epoch,variant]
        recomputed=metrics(rows[0]['dataset'],rows)
        for field,value in recomputed.items():
            if field=='per_label':
                if json.loads(saved[field])!=value:
                    raise AssertionError(f'Per-label metric mismatch: {path}')
            elif not math.isclose(float(saved[field]),value,rel_tol=1e-9,abs_tol=1e-12):
                raise AssertionError(f'Metric mismatch: {path}, {field}')
        files+=1
    if not files:
        raise ValueError('No prediction artifacts found')
    report=dict(status='passed',root=a.root,prediction_files=files,examples=count,
                strict_special_token_decoding=True,all_raw_predictions_and_aggregate_metrics_match=True)
    write_json(a.output,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
