"""Isolated real-checkpoint diagnosis: replay, content isolation, structure and OCR controls."""
from collections import Counter
from contextlib import contextmanager
import argparse,json,random,re
from pathlib import Path
import torch
from transformers import LogitsProcessorList
from PIL import Image
from modalitygap.model import Encoder,load_model,adapt
from modalitygap.runner import configure_reproducibility,to_device,load_trainable,decode_prediction
from modalitygap.checkpoint import read
from modalitygap.protocol import MODEL_REVISION,read_jsonl,write_json,write_jsonl,metrics,digest
from modalitygap.symmetric_mask import SymmetricMask,VisibleHistoryPenalty

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/contribution-cause-diagnostics-v1'
OLD=ROOT/'artifacts/real-expansion-symmetric-8epoch-v1'


@contextmanager
def intervention(hook,batch,body,condition):
    if condition!='mask_both':
        with hook.use(batch,body,condition) as penalty:yield penalty
    else:
        # Diagnostic union of the SAME blocked key sets. All values are unreachable.
        with hook.use(batch,body,'mask_text'):
            image=(batch['input_ids'][0]==hook.image_id).nonzero().flatten().tolist()
            hidden=sorted(set(body)|set(image));hook.state['hidden']=hidden
            yield VisibleHistoryPenalty(hidden)


def generate(model,encoder,hook,row,condition):
    batch,body,truncated=encoder(row);batch=to_device(batch)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),intervention(hook,batch,body,condition) as penalty:
        output=model.generate(**batch,do_sample=False,use_cache=True,repetition_penalty=1.,
            logits_processor=LogitsProcessorList([penalty]),max_new_tokens=8,
            return_dict_in_generate=True,output_logits=True,output_scores=True)
    ids=output.sequences[0,batch['input_ids'].shape[1]:].tolist()
    raw,pred,invalid,error=decode_prediction(encoder.processor.tokenizer,encoder.dataset,ids)
    eos=encoder.processor.tokenizer.eos_token_id
    return dict(dataset=encoder.dataset,id=row['id'],reference=row['labels'],prediction=pred,invalid=invalid,
        raw_output=raw,generated_token_ids=ids,generated_length=len(ids),decode_error=error,truncated=truncated,
        first_eos=bool(ids) and ids[0] in (151645,151643),
        eos_probability=output.logits[0][0].float().softmax(-1)[eos].item(),
        attention_layer_checks=len(hook.layer_checks),body_length=len(body))


def probe(model,encoder,hook,row,condition):
    batch,body,_=encoder(row);batch=to_device(batch)
    changed={k:v.clone() for k,v in batch.items()}
    if condition=='mask_text':
        donor=encoder.processor.tokenizer.encode(' unrelated neutral replacement words ',add_special_tokens=False)
        changed['input_ids'][0,body]=torch.tensor([donor[i%len(donor)] for i in range(len(body))],device='cuda')
    else:changed['pixel_values']=torch.flip(batch['pixel_values'],dims=[0])*.37+.19
    def forward(inputs,active):
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),hook.use(inputs,body,active) as penalty:
            output=model(**inputs,use_cache=True,logits_to_keep=1);cache=output.past_key_values
            history=inputs['input_ids'];raw=output.logits[0,-1].float().clone()
            values=[raw,penalty(history,raw[None].clone())[0]]
            for token in (17,42,91):
                history=torch.cat((history,torch.tensor([[token]],device='cuda')),dim=1)
                output=model(input_ids=history[:,-1:],attention_mask=torch.ones((1,cache.get_seq_length()+1),device='cuda',dtype=torch.long),past_key_values=cache,use_cache=True,logits_to_keep=1)
                cache=output.past_key_values;raw=output.logits[0,-1].float().clone()
                values.extend([raw,penalty(history,raw[None].clone())[0]])
            return values,len(hook.layer_checks)
    a,ca=forward(batch,condition);b,cb=forward(changed,condition)
    deltas=[(x-y).abs().max().item() for x,y in zip(a,b)]
    assert all(d==0 for d in deltas),(row['id'],condition,deltas)
    c,_=forward(batch,'full');d,_=forward(changed,'full')
    control=(c[0]-d[0]).abs().max().item();assert control>0
    return dict(id=row['id'],condition=condition,prefill_and_three_cached_steps_raw_and_processed_max_deltas=deltas,
                full_control_max_logit_delta=control,attention_layer_checks=ca+cb,status='passed')


def ocr(model,processor,row):
    with Image.open(row['image']) as im:image=im.convert('RGB')
    messages=[dict(role='user',content=[dict(type='image'),dict(type='text',text='Transcribe the visible written text in this image. Output only the text, without explanation.')])]
    prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    batch=to_device(processor(text=[prompt],images=[image],return_tensors='pt'))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        out=model.generate(**batch,do_sample=False,use_cache=True,max_new_tokens=128,repetition_penalty=1.05)
    text=processor.tokenizer.decode(out[0,batch['input_ids'].shape[1]:],skip_special_tokens=True)
    ref=Counter(re.findall(r'\w+',row['text'].casefold()));got=Counter(re.findall(r'\w+',text.casefold()))
    return dict(id=row['id'],reference_body=row['text'],image_only_transcription=text,
                reference_word_recall=sum((ref&got).values())/sum(ref.values()),
                exact_normalized_words=list(re.findall(r'\w+',row['text'].casefold()))==list(re.findall(r'\w+',text.casefold())))


def main(task):
    configure_reproducibility();folder=OUT/task
    if (folder/'complete.json').exists():raise ValueError('Diagnostics already complete; no overwrite')
    if (folder/'results').exists():raise ValueError('Use new diagnostic output after an interrupted run')
    results=folder/'results';results.mkdir()
    rows=read_jsonl(folder/'dev128.jsonl');old=rows[:32]
    stats=torch.load(OLD/task/'stats'/f'{task}.pt',weights_only=True)
    model,processor=load_model(MODEL_REVISION,cache_dir=str(ROOT/'artifacts/hf-cache'));model,_=adapt(model,'L',42)
    weights,state=read(OLD/task/'runs'/task/'L-low/seed_42/checkpoints/epoch_8');load_trainable(model,weights);del weights,state
    model.gradient_checkpointing_disable();model.eval();encoder=Encoder(processor,task);hook=SymmetricMask(model,stats)
    summary={};probes=[]
    try:
        for condition in ('full','mask_image','mask_text'):
            records=[generate(model,encoder,hook,r,condition) for r in rows]
            prior={r['id']:r for r in read_jsonl(OLD/task/'runs'/task/'L-low/seed_42/predictions'/f'epoch_8_{condition}.jsonl')}
            assert all(r['generated_token_ids']==prior[r['id']]['generated_token_ids'] for r in records[:32]),condition
            write_jsonl(results/f'dev128_{condition}.jsonl',records)
            summary[condition]={key:metrics(task,subset) for key,subset in [('original32',records[:32]),('additional96',records[32:]),('combined128',records)]}
            print(task,condition,json.dumps({k:v['score'] for k,v in summary[condition].items()}),flush=True)
        for r in random.Random(2026).sample(old,8):
            for condition in ('mask_text','mask_image'):
                probes.append(probe(model,encoder,hook,r,condition))
                write_json(results/'leakage-probes.json',probes)
        print(task,'content isolation probes passed',len(probes),flush=True)
        for length in (1,64):
            records=[generate(model,encoder,hook,dict(r,text=' '.join(['x']*length)),'mask_text') for r in old]
            assert {r['body_length'] for r in records}=={length}
            write_jsonl(results/f'fixed_body_{length}_mask_text.jsonl',records)
            summary[f'fixed_body_{length}']=metrics(task,records)
        records=[generate(model,encoder,hook,r,'mask_both') for r in old]
        write_jsonl(results/'both_modalities_blocked.jsonl',records);summary['mask_both']=metrics(task,records)
        if task=='crisismmd':
            cleaned=[]
            for r in old:
                text=re.sub(r'https?://\S+','',r['text']);text=re.sub(r'(^RT\s+)?@\w+:?','',text).strip()
                cleaned.append(dict(r,text=text or r['text']))
            records=[generate(model,encoder,hook,r,'full') for r in cleaned]
            write_jsonl(results/'url_mentions_removed_full.jsonl',records);summary['url_mentions_removed']=metrics(task,records)
        else:
            ids={'63987','67208'};chosen=[r for r in old if r['id'] in ids]
            chosen+=random.Random(2026).sample([r for r in old if r['id'] not in ids],6)
            records=[ocr(model,processor,r) for r in chosen]
            write_json(results/'image-only-ocr.json',records)
            summary['ocr']=dict(count=len(records),mean_reference_word_recall=sum(r['reference_word_recall'] for r in records)/len(records),exact_normalized_words=sum(r['exact_normalized_words'] for r in records))
        write_json(results/'summary.json',summary)
        write_json(folder/'complete.json',dict(status='passed',checkpoint_epoch=8,original32_tokens_reproduced_all_three_conditions=True,
            expanded_dev=128,leakage_probes=len(probes),manifest_sha256=digest(folder/'dev128.jsonl')))
    finally:hook.close()
    print(task,'DIAGNOSTICS COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('crisismmd','hatefulmemes'),required=True);main(p.parse_args().dataset)
