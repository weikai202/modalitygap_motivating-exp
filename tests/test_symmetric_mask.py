"""Actual pinned Qwen attention, KV cache and decoding-history isolation."""
import unittest
import torch
from transformers import Qwen2_5_VLForConditionalGeneration,RepetitionPenaltyLogitsProcessor
import test_qwen
from modalitygap.symmetric_mask import SymmetricMask,VisibleHistoryPenalty


class SymmetricMaskTests(unittest.TestCase):
    def test_qwen_all_layers_prefill_cache_and_protected_positions(self):
        fixture=test_qwen.QwenTests();config=fixture.model().config
        config.text_config.num_hidden_layers=3
        config.text_config.layer_types=['full_attention']*3
        model=Qwen2_5_VLForConditionalGeneration(config).eval().requires_grad_(False)
        inputs=fixture.inputs();body=[7,8]
        stats={m:dict(mean=torch.randn(32)) for m in ('image','text')}
        with torch.no_grad():plain=model(**inputs,use_cache=False).logits
        hook=SymmetricMask(model,stats)
        try:
            with torch.no_grad(),hook.use(inputs,body,'full'):
                self.assertTrue(torch.equal(plain,model(**inputs,use_cache=False).logits))
            for modality in ('text','image'):
                altered={k:v.clone() for k,v in inputs.items()}
                if modality=='text':altered['input_ids'][0,body]=torch.tensor([35,36])
                else:altered['pixel_values']=inputs['pixel_values']*.37+.19
                paths=[];positions=[]
                observer=model.model.language_model.register_forward_pre_hook(lambda m,a,k:positions.append(k['position_ids'].clone()),with_kwargs=True)
                for batch in (inputs,altered):
                    with torch.no_grad(),hook.use(batch,body,'mask_'+modality):
                        out=model(**batch,use_cache=True)
                        values=[out.logits[:,-1].clone()];cache=out.past_key_values
                        for tok in (11,12,13):
                            out=model(input_ids=torch.tensor([[tok]]),past_key_values=cache,
                                attention_mask=torch.ones(1,cache.get_seq_length()+1,dtype=torch.long),use_cache=True)
                            values.append(out.logits[:,-1].clone());cache=out.past_key_values
                        self.assertEqual(len(hook.layer_checks),12)
                        self.assertEqual([r['cache_start'] for r in hook.calls],[0,10,11,12])
                    paths.append(values)
                observer.remove()
                for a,b in zip(*paths):self.assertTrue(torch.equal(a,b))
                for a,b in zip(positions[:4],positions[4:]):self.assertTrue(torch.equal(a,b))
                with torch.no_grad():
                    self.assertGreater((model(**inputs).logits[:,-1]-model(**altered).logits[:,-1]).abs().max().item(),0)
                self.assertTrue(torch.equal(inputs['attention_mask'],torch.ones(1,10,dtype=torch.long)))
        finally:hook.close()

    def test_repetition_penalty_excludes_hidden_prompt_but_keeps_generated_history(self):
        a=torch.tensor([[1,2,3,4,5]]);b=torch.tensor([[1,27,28,4,5]])
        scores=torch.arange(32,dtype=torch.float)[None]-12
        penalty=VisibleHistoryPenalty([1,2])
        self.assertTrue(torch.equal(penalty(a,scores.clone()),penalty(b,scores.clone())))
        full=VisibleHistoryPenalty([])
        self.assertTrue(torch.equal(full(a,scores.clone()),RepetitionPenaltyLogitsProcessor(1.05)(a,scores.clone())))
        self.assertFalse(torch.equal(penalty(a,scores.clone()),penalty(torch.cat((a,torch.tensor([[29]])),1),scores.clone())))


if __name__=='__main__':unittest.main()
