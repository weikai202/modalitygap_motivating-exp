"""Frozen-checkpoint diagnostics. Never changes production masking or training.

Main conditions use all 32 fixed dev examples. Mechanism probes use a fixed
seed2026 subset of eight examples, selected before their outcomes are observed.
"""
import argparse
import gc
import json
import random
from pathlib import Path

import torch
from modalitygap.checkpoint import read
from modalitygap.model import Encoder, Intervention, adapt, load_model
from modalitygap.protocol import MODEL_REVISION, digest, metrics, read_jsonl, write_json, write_jsonl
from modalitygap.runner import configure_reproducibility, decode_prediction, load_trainable, to_device, verified_data

ROOT = Path('artifacts/real-data-pilot-v1')
OUT = ROOT/'mask-diagnostics-v1'


class AuditIntervention(Intervention):
    """Observe production replacements and optionally rescale for mechanism probes."""
    def __init__(self, model, stats):
        self.gain = 1.
        self.audit = None
        super().__init__(model, stats)

    def _hook(self, module, args, kwargs):
        s = self.state
        if s is None:
            return
        cache = kwargs.get('cache_position')
        if cache is not None and cache.numel() and cache[0].item() > 0:
            return super()._hook(module,args,kwargs)
        before = kwargs['inputs_embeds']
        result = super()._hook(module,args,kwargs)
        after = kwargs['inputs_embeds']
        image = (s['ids'][0] == self.image_token_id).nonzero().flatten().tolist()
        spans = dict(image=image, text=s['positions'])
        modality = {'mask_image':'image','mask_text':'text'}.get(s['condition'])
        changed = spans[modality] if modality else []
        if self.gain != 1.:
            if not modality:
                raise ValueError('Gain requires a mask')
            after = after.clone()
            after[0,changed] *= self.gain
            kwargs['inputs_embeds'] = after
            result = args,kwargs
        protected = [i for i in range(before.shape[1]) if i not in set(changed)]
        if not torch.equal(before[:,protected], after[:,protected]):
            raise AssertionError('Mask changed protected tokens')
        self.audit = dict(protected_tokens_identical=True,
                          sequence_length=before.shape[1], body_tokens=len(spans['text']),
                          image_tokens=len(image), gain=self.gain)
        for name, positions in spans.items():
            a,b = before[0,positions].float(), after[0,positions].float()
            def describe(x):
                norms=x.norm(dim=-1)
                return dict(mean_token_norm=norms.mean().item(),
                            rms_token_norm=x.square().sum(-1).mean().sqrt().item(),
                            mean_vector_norm=x.mean(0).norm().item(),
                            centered_rms=(x-x.mean(0)).square().sum(-1).mean().sqrt().item())
            self.audit[name] = dict(original=describe(a),replacement=describe(b),
                mean_cosine=torch.nn.functional.cosine_similarity(a,b,dim=-1).mean().item())
        return result


def conditions():
    items=[dict(name='full',mask='full',seed=2026,scale=.05)]
    for m in ('image','text'):
        items += [dict(name=f'{m}_mean_only',mask=f'mask_{m}',seed=2026,scale=0.)]
        items += [dict(name=f'{m}_noise_{s}',mask=f'mask_{m}',seed=s,scale=.05) for s in (2026,2027,2028)]
    items += [dict(name=f'shuffle_{m}',mask='full',shuffle=m,seed=2026,scale=.05) for m in ('image','text')]
    items += [dict(name='text_noise_std1',mask='mask_text',seed=2026,scale=1.,probe=True),
              dict(name='text_noise_normmatched',mask='mask_text',seed=2026,scale=.05,normalize=True,probe=True),
              dict(name='text_mean_normmatched',mask='mask_text',seed=2026,scale=0.,normalize=True,probe=True)]
    return items


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True,choices=('mmsd2','mmimdb'))
    args=parser.parse_args()
    configure_reproducibility()
    data=Path('data/real-pilot')/args.dataset/'manifest'
    verified_data(data,args.dataset)
    rows=read_jsonl(data/'dev.jsonl')
    if len(rows)!=32:
        raise ValueError('Expected frozen 32-example development subset')
    probe_ids=set(r['id'] for r in random.Random(2026).sample(rows,8))
    # A fixed cyclic derangement has no self pairs, and keeps all samples dev-only.
    shuffled=list(rows);random.Random(2026).shuffle(shuffled)
    donors={r['id']:shuffled[(i+1)%len(shuffled)] for i,r in enumerate(shuffled)}
    directory=OUT/args.dataset
    directory.mkdir(parents=True,exist_ok=True)
    plan=dict(dataset=args.dataset,epochs=[0,2],conditions=conditions(),
              dev_ids=[r['id'] for r in rows],probe_ids=sorted(probe_ids),
              donors={k:v['id'] for k,v in donors.items()},
              shuffle_reference='Original recipient labels; shuffle_text changes plot/post only, shuffle_image changes image only',
              manifest_sha256=digest(data/'manifest.json'),stats_sha256=digest(ROOT/'stats'/f'{args.dataset}.pt'),
              model_revision=MODEL_REVISION,
              norm_matching='Fixed population RMS from frozen train moments; no recipient contents/labels used',
              first_token='Actual generation prefill logits and post-processor scores; report raw EOS probability and post-processor EOS probability separately',
              decoding='Original deterministic greedy generation, inherited checkpoint repetition_penalty=1.05, output_logits and output_scores enabled',
              note='Exploratory diagnostic, not protocol tuning or a replacement for primary scores')
    path=directory/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:
        raise ValueError('Frozen diagnostic plan differs')
    write_json(path,plan)
    stats=torch.load(ROOT/'stats'/f'{args.dataset}.pt',map_location='cpu',weights_only=True)
    model,processor=load_model(MODEL_REVISION,cache_dir='artifacts/hf-cache')
    model,_=adapt(model,'L',42)
    model.gradient_checkpointing_disable()
    model.eval()
    encoder=Encoder(processor,args.dataset)
    hook=AuditIntervention(model,stats)
    eos=processor.tokenizer.eos_token_id
    summaries=[]
    try:
        for epoch in (0,2):
            checkpoint=ROOT/'runs'/args.dataset/'L-low'/'seed_42_pilot'/'checkpoints'/f'epoch_{epoch}'
            weights,state=read(checkpoint)
            load_trainable(model,weights)
            del weights,state
            for condition in conditions():
                name=condition['name']
                output=directory/f'epoch_{epoch}_{name}.jsonl'
                if output.exists():
                    saved=read_jsonl(output)
                    if len(saved)!=(8 if condition.get('probe') else 32):
                        raise ValueError('Incomplete saved condition')
                else:
                    saved=[]
                    hook.gain=1.
                    if condition.get('normalize'):
                        mu,sd=stats['text']['mean'],stats['text']['std']
                        hook.gain=((mu.square().sum()+sd.square().sum()) /
                            (mu.square().sum()+condition['scale']**2*sd.square().sum())).sqrt().item()
                    for original in rows:
                        if condition.get('probe') and original['id'] not in probe_ids:
                            continue
                        row=dict(original)
                        donor=donors[row['id']]
                        if condition.get('shuffle')=='image':
                            row['image'],row['image_hash']=donor['image'],donor['image_hash']
                        elif condition.get('shuffle')=='text':
                            row['text']=donor['text']
                        batch,positions,truncated=encoder(row)
                        batch=to_device(batch)
                        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                            with hook.use(row,batch,positions,condition['mask'],condition['seed'],condition['scale']):
                                generation=model.generate(**batch,do_sample=False,use_cache=True,
                                    max_new_tokens=8 if args.dataset=='mmsd2' else 128,
                                    return_dict_in_generate=True,output_logits=True,output_scores=True)
                            audit=hook.audit
                            logits=generation.logits[0][0].float()
                            processed=generation.scores[0][0].float()
                            prob=logits.softmax(-1)
                            processed_prob=processed.softmax(-1)
                            generated=generation.sequences[0,batch['input_ids'].shape[1]:].tolist()
                            greedy=int(processed.argmax())
                            if generated[0]!=greedy:
                                raise AssertionError('First generation token differs from processed score argmax')
                            top=torch.topk(prob,5)
                            first=dict(eos_probability=prob[eos].item(),
                                processed_eos_probability=processed_prob[eos].item(),
                                eos_logit=logits[eos].item(),
                                eos_rank=1+int((logits>logits[eos]).sum()),
                                processed_eos_rank=1+int((processed>processed[eos]).sum()),
                                raw_argmax_id=int(logits.argmax()),argmax_id=greedy,
                                argmax_text=processor.tokenizer.decode([greedy],skip_special_tokens=False),
                                top5=[dict(id=int(t),text=processor.tokenizer.decode([int(t)],skip_special_tokens=False),probability=float(p)) for p,t in zip(top.values,top.indices)])
                        raw,pred,invalid,error=decode_prediction(processor.tokenizer,args.dataset,generated)
                        saved.append(dict(dataset=args.dataset,epoch=epoch,condition=name,id=row['id'],
                            donor_id=donor['id'] if condition.get('shuffle') else None,
                            reference=original['labels'],prediction=pred,invalid=invalid,decode_error=error,
                            raw_output=raw,generated_token_ids=generated,generated_length=len(generated),
                            first_token=first,embeddings=audit,truncated=truncated))
                        hook.logs.clear()
                        del batch,logits,prob,top,generation,processed,processed_prob
                    write_jsonl(output,saved)
                score=metrics(args.dataset,saved)
                summary=dict(epoch=epoch,condition=name,probe=bool(condition.get('probe')),**score,
                    mean_eos_probability=sum(r['first_token']['eos_probability'] for r in saved)/len(saved),
                    mean_processed_eos_probability=sum(r['first_token']['processed_eos_probability'] for r in saved)/len(saved),
                    greedy_eos_count=sum(r['first_token']['argmax_id']==eos for r in saved),
                    mean_text_original_norm=sum(r['embeddings']['text']['original']['mean_token_norm'] for r in saved)/len(saved),
                    mean_text_replacement_norm=sum(r['embeddings']['text']['replacement']['mean_token_norm'] for r in saved)/len(saved),
                    mean_image_original_norm=sum(r['embeddings']['image']['original']['mean_token_norm'] for r in saved)/len(saved),
                    mean_image_replacement_norm=sum(r['embeddings']['image']['replacement']['mean_token_norm'] for r in saved)/len(saved))
                summaries.append(summary)
                write_json(directory/'summary.json',summaries)
                print(json.dumps({k:v for k,v in summary.items() if k in ('epoch','condition','score','invalid_rate','mean_eos_probability','greedy_eos_count')}),flush=True)
    finally:
        hook.close()
    print('DIAGNOSTICS COMPLETE',args.dataset,flush=True)


if __name__=='__main__':
    main()
