"""Audit instrumentation must preserve the production intervention."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from modalitygap.model import Intervention

spec=importlib.util.spec_from_file_location('diagnose_real_masks',Path(__file__).resolve().parents[1]/'scripts/diagnose_real_masks.py')
diag=importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


class DiagnosticTests(unittest.TestCase):
    def test_audit_matches_production_and_rescaling_preserves_prompt_and_cache(self):
        class Decoder(torch.nn.Module):
            def forward(self,inputs_embeds,cache_position=None):
                return inputs_embeds
        def model():
            m=torch.nn.Module()
            m.config=SimpleNamespace(image_token_id=91)
            m.language_model=Decoder()
            return m
        stats={m:dict(mean=torch.ones(4)*.25,std=torch.ones(4)) for m in ('image','text')}
        batch=dict(input_ids=torch.tensor([[1,91,91,4,5,6]]))
        x=torch.arange(24,dtype=torch.float).reshape(1,6,4)
        a,b=model(),model()
        production=Intervention(a,stats);audit=diag.AuditIntervention(b,stats)
        try:
            for condition in ('full','mask_image','mask_text'):
                with production.use(dict(id='example'),batch,[4],condition):
                    expected=a.language_model(inputs_embeds=x)
                with audit.use(dict(id='example'),batch,[4],condition):
                    actual=b.language_model(inputs_embeds=x)
                self.assertTrue(torch.equal(expected,actual))
                self.assertTrue(audit.audit['protected_tokens_identical'])
            audit.gain=4.
            with audit.use(dict(id='example'),batch,[4],'mask_text'):
                scaled=b.language_model(inputs_embeds=x)
                cached=b.language_model(inputs_embeds=x[:,:1],cache_position=torch.tensor([6]))
            self.assertTrue(torch.equal(cached,x[:,:1]))
            self.assertTrue(torch.equal(scaled[:,[0,1,2,3,5]],x[:,[0,1,2,3,5]]))
            self.assertTrue(torch.equal(scaled[:,4],expected[:,4]*4))
        finally:
            production.close();audit.close()

class DiagnosticOrchestrationTests(unittest.TestCase):
    def test_complete_diagnostic_writes_every_condition_and_can_resume(self):
        import contextlib
        import io
        import tempfile
        import warnings
        from unittest.mock import patch
        import json
        class Decoder(torch.nn.Module):
            def forward(self,inputs_embeds,cache_position=None):
                return inputs_embeds
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config=SimpleNamespace(image_token_id=91)
                self.language_model=Decoder()
            def gradient_checkpointing_disable(self):
                pass
            def generate(self,**kwargs):
                self.language_model(inputs_embeds=torch.arange(24,dtype=torch.float).reshape(1,6,4))
                logits=torch.zeros(1,8);logits[0,7]=10
                return SimpleNamespace(logits=(logits,),scores=(logits,),
                    sequences=torch.cat((kwargs['input_ids'],torch.tensor([[7]])),dim=1))
        rows=[dict(id=str(i),text=f'text {i}',image=f'{i}.jpg',image_hash=str(i),
                   labels=['not_sarcastic']) for i in range(32)]
        processor=SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=7,pad_token_id=0,
                                                            decode=lambda *a,**k:''))
        encoder=lambda row:(dict(input_ids=torch.tensor([[1,91,91,4,5,6]])),[4],False)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'stats').mkdir()
            torch.save({m:dict(mean=torch.ones(4),std=torch.ones(4)) for m in ('image','text')},root/'stats/mmsd2.pt')
            with contextlib.ExitStack() as stack:
                for name,value in dict(ROOT=root,OUT=root/'diagnostics',
                    configure_reproducibility=lambda:None,
                    verified_data=lambda *a:None,
                    digest=lambda *a:'frozen-hash',
                    load_model=lambda *a,**k:(Model(),processor),
                    adapt=lambda m,*a:(m,[]),Encoder=lambda *a:encoder,
                    to_device=lambda x:x,read=lambda *a:({},{}),
                    load_trainable=lambda *a:None).items():
                    stack.enter_context(patch.object(diag,name,value))
                real_read=diag.read_jsonl
                stack.enter_context(patch.object(diag,'read_jsonl',side_effect=lambda path:rows if str(path).endswith('manifest/dev.jsonl') else real_read(path)))
                stack.enter_context(patch('sys.argv',['diagnose_real_masks.py','--dataset','mmsd2']))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    diag.main()
                    first=(root/'diagnostics/mmsd2/summary.json').read_bytes()
                    diag.main()
            summary=json.loads(first)
            self.assertEqual(len(summary),28)
            self.assertEqual(first,(root/'diagnostics/mmsd2/summary.json').read_bytes())
            self.assertEqual(len(list((root/'diagnostics/mmsd2').glob('epoch_*.jsonl'))),28)
            self.assertEqual(sum(r['count'] for r in summary),752)


if __name__=='__main__':
    unittest.main()
