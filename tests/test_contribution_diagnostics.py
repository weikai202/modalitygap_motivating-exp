"""The added joint-blocking control must isolate both content streams through cache."""
import importlib.util
from pathlib import Path
import unittest
import torch
from modalitygap.symmetric_mask import SymmetricMask
import test_qwen

spec=importlib.util.spec_from_file_location('contribution_diagnostics',Path(__file__).resolve().parents[1]/'scripts/diagnose_contribution_causes.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class JointMaskControlTests(unittest.TestCase):
    def test_joint_mask_isolates_both_contents_prefill_cache_and_penalty(self):
        fixture=test_qwen.QwenTests();model=fixture.model().eval();inputs=fixture.inputs();body=[7,8,9]
        changed={k:v.clone() for k,v in inputs.items()};changed['input_ids'][0,body]=torch.tensor([31,32,33]);changed['pixel_values']*=.21
        hook=SymmetricMask(model,{m:dict(mean=torch.randn(32)) for m in ('image','text')})
        def forward(batch):
            with torch.no_grad(),module.intervention(hook,batch,body,'mask_both') as penalty:
                self.assertEqual(set(hook.state['hidden']),{2,3,4,5,7,8,9})
                out=model(**batch,use_cache=True);cache=out.past_key_values;history=batch['input_ids']
                raw=out.logits[:,-1].clone();values=[raw,penalty(history,raw.clone())]
                for tok in (17,42,91):
                    history=torch.cat((history,torch.tensor([[tok]])),dim=1)
                    out=model(input_ids=history[:,-1:],past_key_values=cache,attention_mask=torch.ones(1,cache.get_seq_length()+1,dtype=torch.long),use_cache=True)
                    cache=out.past_key_values;raw=out.logits[:,-1].clone();values.extend([raw,penalty(history,raw.clone())])
                self.assertTrue(hook.layer_checks)
                self.assertTrue(all(c['blocked']==7 for c in hook.calls))
                return values
        try:
            for a,b in zip(forward(inputs),forward(changed)):self.assertTrue(torch.equal(a,b))
            with torch.no_grad():self.assertGreater((model(**inputs).logits-model(**changed).logits).abs().max().item(),0)
        finally:hook.close()
