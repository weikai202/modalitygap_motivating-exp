"""Real-data evaluation and exact same-structure leakage probes, isolated v2 outputs."""
import argparse,json,random
from pathlib import Path
import torch
from transformers import LogitsProcessorList
from modalitygap.model import load_model,adapt,Encoder
from modalitygap.runner import configure_reproducibility,to_device,load_trainable,decode_prediction,verified_data
from modalitygap.checkpoint import read
from modalitygap.protocol import MODEL_REVISION,read_jsonl,write_json,write_jsonl,metrics,digest
from symmetric_mask import SymmetricMask

ROOT=Path('artifacts/real-data-pilot-v1')
OUT=ROOT/'symmetric-attention-mask-v2'


def generate(model,encoder,hook,row,condition):
    batch,body,truncated=encoder(row);batch=to_device(batch)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),hook.use(batch,body,condition) as penalty:
        output=model.generate(**batch,do_sample=False,use_cache=True,repetition_penalty=1.,
            logits_processor=LogitsProcessorList([penalty]),return_dict_in_generate=True,
            output_logits=True,output_scores=True,max_new_tokens=8 if encoder.dataset=='mmsd2' else 128)
    ids=output.sequences[0,batch['input_ids'].shape[1]:].tolist()
    raw,pred,invalid,error=decode_prediction(encoder.processor.tokenizer,encoder.dataset,ids)
    logits=output.logits[0][0].float();scores=output.scores[0][0].float();eos=encoder.processor.tokenizer.eos_token_id
    return dict(dataset=encoder.dataset,id=row['id'],reference=row['labels'],prediction=pred,
        invalid=invalid,decode_error=error,raw_output=raw,generated_token_ids=ids,generated_length=len(ids),
        eos_probability=logits.softmax(-1)[eos].item(),processed_eos_probability=scores.softmax(-1)[eos].item(),
        first_eos=ids[0] in (151645,151643),truncated=truncated,
        decoder_calls=list(hook.calls),attention_layer_checks=len(hook.layer_checks))


def probe(model,encoder,hook,row,condition):
    batch,body,_=encoder(row);batch=to_device(batch)
    altered={k:v.clone() for k,v in batch.items()}
    if condition=='mask_text':
        # Same number of tokens, entirely different token IDs, without special IDs.
        ids=encoder.processor.tokenizer.encode(' random unrelated different words movie story ',add_special_tokens=False)
        altered['input_ids'][0,body]=torch.tensor([ids[i%len(ids)] for i in range(len(body))],device='cuda')
    else:
        altered['pixel_values']=torch.flip(batch['pixel_values'],dims=[0])*.37+.19
    def run(inputs,active):
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),hook.use(inputs,body,active) as penalty:
            out=model(**inputs,use_cache=True,logits_to_keep=1)
            raw=out.logits[0,-1].float().clone()
            processed=penalty(inputs['input_ids'],raw[None].clone())[0]
            # Force same continuation tokens to test all cached steps, even if model prefers EOS.
            cache=out.past_key_values;track=[raw,processed]
            for token in (17,42,91):
                seq=torch.cat((inputs['input_ids'],torch.tensor([[token]],device='cuda')),dim=1)
                out=model(input_ids=seq[:,-1:],attention_mask=torch.ones((1,cache.get_seq_length()+1),device='cuda',dtype=torch.long),
                    past_key_values=cache,use_cache=True,logits_to_keep=1)
                cache=out.past_key_values
                track.append(out.logits[0,-1].float().clone())
            return track,len(hook.layer_checks)
    a,checks=run(batch,condition);b,checks2=run(altered,condition)
    delta=[(x-y).abs().max().item() for x,y in zip(a,b)]
    if any(delta):raise AssertionError(f'Content leakage: {row["id"]} {condition} {delta}')
    c,_=run(batch,'full');d,_=run(altered,'full')
    return dict(id=row['id'],condition=condition,masked_max_absolute_deltas=delta,
        full_first_logit_max_delta=(c[0]-d[0]).abs().max().item(),attention_checks=checks+checks2,
        exact_prefill_processed_and_three_cached_steps=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=('mmsd2','mmimdb'));args=p.parse_args()
    configure_reproducibility()
    directory=OUT/args.dataset
    if directory.exists():raise ValueError('Use a fresh diagnostic output; no overwrite')
    data=Path('data/real-pilot')/args.dataset/'manifest';verified_data(data,args.dataset)
    rows=read_jsonl(data/'dev.jsonl');audit=random.Random(2026).sample(rows,8)
    directory.mkdir(parents=True)
    write_json(directory/'plan.json',dict(dataset=args.dataset,dev_ids=[r['id'] for r in rows],probe_ids=[r['id'] for r in audit],
        epochs=[0,2],conditions=['full','mask_image','mask_text'],method='fixed training means + all-layer hidden-key exclusion',
        repetition_penalty='1.05 on visible prompt/generated tokens; masked prompt excluded',
        position_policy='Outer model receives original attention mask and grids; decoder key mask applied after original mRoPE construction',
        source_sha256=digest('scripts/symmetric_mask.py'),manifest_sha256=digest(data/'manifest.json')))
    stats=torch.load(ROOT/'stats'/f'{args.dataset}.pt',weights_only=True)
    model,processor=load_model(MODEL_REVISION,cache_dir='artifacts/hf-cache');model,_=adapt(model,'L',42)
    model.gradient_checkpointing_disable();model.eval();encoder=Encoder(processor,args.dataset);hook=SymmetricMask(model,stats)
    summaries=[];leakage=[]
    try:
        for epoch in (0,2):
            weights,state=read(ROOT/'runs'/args.dataset/'L-low/seed_42_pilot/checkpoints'/f'epoch_{epoch}')
            load_trainable(model,weights);del weights,state
            for condition in ('full','mask_image','mask_text'):
                saved=[generate(model,encoder,hook,row,condition) for row in rows]
                if condition=='full':
                    old={r['id']:r for r in read_jsonl(ROOT/'runs'/args.dataset/'L-low/seed_42_pilot/predictions'/f'epoch_{epoch}_full.jsonl')}
                    if any(r['generated_token_ids']!=old[r['id']]['generated_token_ids'] for r in saved):raise AssertionError('Full generation changed')
                write_jsonl(directory/f'epoch_{epoch}_{condition}.jsonl',saved)
                summary=dict(epoch=epoch,condition=condition,**metrics(args.dataset,saved),
                    mean_eos_probability=sum(r['eos_probability'] for r in saved)/len(saved),
                    mean_processed_eos_probability=sum(r['processed_eos_probability'] for r in saved)/len(saved),
                    first_eos_count=sum(r['first_eos'] for r in saved))
                summaries.append(summary);write_json(directory/'summary.json',summaries);print(json.dumps(summary),flush=True)
            for row in audit:
                for condition in ('mask_text','mask_image'):
                    record=dict(epoch=epoch,**probe(model,encoder,hook,row,condition));leakage.append(record)
                    write_json(directory/'leakage.json',leakage)
            print('LEAKAGE PASSED',epoch,len(leakage),flush=True)
    finally:hook.close()
    write_json(directory/'complete.json',dict(status='passed',predictions=192,leakage_probes=len(leakage),full_tokens_exact=True))
    print('SYMMETRIC DIAGNOSTICS COMPLETE',args.dataset,flush=True)


if __name__=='__main__':main()
