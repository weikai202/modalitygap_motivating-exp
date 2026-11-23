"""Independent mean-placeholder + key exclusion intervention; no training changes."""
from contextlib import contextmanager
import torch
from transformers import LogitsProcessor, RepetitionPenaltyLogitsProcessor


class VisibleHistoryPenalty(LogitsProcessor):
    """Apply the original penalty only to visible prompt and generated tokens."""
    def __init__(self, hidden_positions, penalty=1.05):
        self.hidden=set(hidden_positions)
        self.penalty=RepetitionPenaltyLogitsProcessor(penalty)

    def __call__(self, input_ids, scores):
        keep=[i for i in range(input_ids.shape[1]) if i not in self.hidden]
        return self.penalty(input_ids[:,keep],scores)


class SymmetricMask:
    def __init__(self, model, stats):
        self.stats=stats;self.state=None;self.calls=[];self.layer_checks=[];self.logs=[]
        decoders=[m for n,m in model.named_modules() if n.endswith('language_model')]
        if len(decoders)!=1:raise ValueError('Expected one language decoder')
        self.decoder=decoders[0]
        self.image_id=model.config.image_token_id
        self.handles=[self.decoder.register_forward_pre_hook(self._hook,with_kwargs=True)]
        for layer in self.decoder.layers:
            self.handles.append(layer.self_attn.register_forward_pre_hook(self._attention,with_kwargs=True))

    @contextmanager
    def use(self,batch,body,condition):
        if self.state is not None:raise ValueError('Nested intervention')
        if condition not in ('full','mask_image','mask_text'):raise ValueError('Unknown condition')
        if batch['input_ids'].shape[0]!=1:raise ValueError('Batch size must be one')
        images=(batch['input_ids'][0]==self.image_id).nonzero().flatten().tolist()
        if not images or not body or set(images)&set(body):raise ValueError('Invalid content spans')
        hidden={'full':[],'mask_image':images,'mask_text':list(body)}[condition]
        self.state=dict(hidden=hidden,condition=condition,length=batch['input_ids'].shape[1])
        self.calls=[];self.layer_checks=[]
        try:
            yield VisibleHistoryPenalty(hidden)
        finally:
            self.logs.append(dict(condition=condition,hidden_positions=hidden,decoder_calls=list(self.calls),attention_layer_checks=len(self.layer_checks)))
            self.state=None

    def _hook(self,module,args,kwargs):
        s=self.state
        if s is None or not s['hidden']:return
        x=kwargs['inputs_embeds'];cp=kwargs.get('cache_position')
        past=kwargs.get('past_key_values')
        start=int(cp[0]) if cp is not None else (past.get_seq_length() if past is not None else 0)
        if start==0:
            if x.shape[1]!=s['length']:raise ValueError('Unexpected prefill length')
            changed=x.clone()
            modality={'mask_image':'image','mask_text':'text'}[s['condition']]
            mean=self.stats[modality]['mean'].to(x.device,x.dtype)
            if mean.shape!=(x.shape[-1],) or not torch.isfinite(mean).all():raise ValueError('Invalid mean')
            changed[0,s['hidden']]=mean
            keep=[i for i in range(x.shape[1]) if i not in s['hidden']]
            assert torch.equal(changed[:,keep],x[:,keep])
            kwargs['inputs_embeds']=changed
        mask=kwargs.get('attention_mask')
        if mask is None:mask=torch.ones((1,start+x.shape[1]),device=x.device,dtype=torch.long)
        if not isinstance(mask,torch.Tensor) or mask.ndim!=2:raise ValueError('Expected original 2D attention mask')
        if max(s['hidden'])>=mask.shape[-1]:raise ValueError('Hidden KV positions missing')
        mask=mask.clone();mask[:,s['hidden']]=0
        kwargs['attention_mask']=mask
        self.calls.append(dict(cache_start=start,query_length=x.shape[1],key_length=mask.shape[1],blocked=len(s['hidden'])))
        return args,kwargs

    def _attention(self,module,args,kwargs):
        s=self.state
        if s is None or not s['hidden']:return
        mask=kwargs.get('attention_mask')
        if not isinstance(mask,torch.Tensor) or mask.ndim!=4:raise ValueError('Expected materialized causal mask')
        blocked=mask[...,s['hidden']]
        valid=(~blocked).all() if blocked.dtype==torch.bool else (blocked<=torch.finfo(blocked.dtype).min/2).all()
        if not valid:raise AssertionError('Hidden keys reachable in attention')
        self.layer_checks.append(True)

    def close(self):
        for h in self.handles:h.remove()
