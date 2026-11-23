"""Real-data processor audit: references cannot alter any inference tensor or span."""
import json
from pathlib import Path
import torch
from transformers import AutoProcessor
from modalitygap.model import Encoder
from modalitygap.protocol import DEFAULTS,LABELS,read_jsonl,write_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    processor=AutoProcessor.from_pretrained(ROOT/'artifacts/hf-cache/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3',
        use_fast=False,local_files_only=True,min_pixels=DEFAULTS['min_pixels'],max_pixels=DEFAULTS['max_pixels'])
    result={}
    for task in ('newsclippings','crisismmd','hatefulmemes'):
        rows=read_jsonl(ROOT/'data/real-expansion-v1'/task/'source.jsonl');encoder=Encoder(processor,task)
        lengths={label:[] for label in LABELS[task]};truncated=0
        for row in rows:
            original,span,cut=encoder(row)
            other_label=next(label for label in LABELS[task] if label not in row['labels'])
            changed,other_span,other_cut=encoder(dict(row,labels=[other_label]))
            assert span==other_span and cut==other_cut and set(original)==set(changed)
            for key in original:assert torch.equal(original[key],changed[key]),(task,row['id'],key)
            body=processor.tokenizer.encode(row['text'],add_special_tokens=False)
            assert original['input_ids'][0,span].tolist()==body[:len(span)]
            lengths[row['labels'][0]].append(len(span));truncated+=cut
        result[task]=dict(status='passed',examples=len(rows),all_inference_tensors_and_spans_reference_independent=True,
                         body_token_spans_exact=True,truncated=truncated,
                         retained_body_token_lengths_by_label=lengths,
                         structural_caveat='Original body length and image grid/positions are intentionally retained; this audit proves reference independence, not independence from retained structure or visible OCR.')
        print(task,'real processor audit passed',len(rows),flush=True)
    write_json(ROOT/'artifacts/real-expansion-symmetric-8epoch-v1/real-input-audit.json',result)


if __name__=='__main__':main()
