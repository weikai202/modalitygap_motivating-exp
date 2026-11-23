"""No model download: exercise the pinned architecture with random tiny weights."""
import unittest

try:
    import torch
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
except ImportError:
    torch = None


@unittest.skipIf(torch is None,'Requires pinned training libraries')
class QwenTests(unittest.TestCase):
    def model(self):
        config = Qwen2_5_VLConfig(
            text_config=dict(vocab_size=128,hidden_size=32,intermediate_size=64,
                num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,
                rope_scaling={'type':'mrope','mrope_section':[1,1,2]}),
            vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,
                out_hidden_size=32,patch_size=14,spatial_merge_size=2,
                temporal_patch_size=2,window_size=112,fullatt_block_indexes=[0]),
            image_token_id=92,video_token_id=94,vision_start_token_id=91,vision_end_token_id=93)
        model = Qwen2_5_VLForConditionalGeneration(config)
        return model.requires_grad_(False)

    def inputs(self):
        return dict(input_ids=torch.tensor([[1,91,92,92,92,92,93,5,6,7]]),
                    attention_mask=torch.ones(1,10,dtype=torch.long),
                    image_grid_thw=torch.tensor([[1,4,4]]),
                    pixel_values=torch.randn(16,3*2*14*14))

    def test_checkpoint_roundtrip_preserves_frozen_backbone(self):
        import copy
        from modalitygap.model import adapt
        from modalitygap.runner import load_trainable
        for scope in ('L','M','LM'):
            with self.subTest(scope=scope):
                backbone = self.model()
                original = copy.deepcopy(backbone)
                model,_ = adapt(backbone,scope,42)
                frozen = {n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
                before = {n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
                inputs = self.inputs()
                model.train()
                model(**inputs,labels=inputs['input_ids'],use_cache=False).loss.backward()
                self.assertTrue(any(p.grad is not None and p.grad.norm()>0 for p in model.parameters() if p.requires_grad))
                optimizer.step()
                state = {n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
                self.assertTrue(any(not torch.equal(state[n],before[n]) for n in state))
                for n,p in model.named_parameters():
                    if n in frozen:
                        self.assertTrue(torch.equal(p,frozen[n]))
                restored,_ = adapt(original,scope,42)
                load_trainable(restored,state)
                model.eval()
                restored.eval()
                with torch.no_grad():
                    self.assertTrue(torch.equal(model(**inputs,use_cache=False).logits,restored(**inputs,use_cache=False).logits))
                with self.assertRaises(ValueError):
                    load_trainable(restored,{})
                broken = {n:v.clone() for n,v in state.items()}
                broken[next(iter(broken))].fill_(float('nan'))
                with self.assertRaises(ValueError):
                    load_trainable(restored,broken)

    def test_scopes_backward_and_cached_interventions(self):
        from modalitygap.symmetric_mask import SymmetricMask
        from modalitygap.model import adapt
        for scope in ('L','M','LM'):
            with self.subTest(scope=scope):
                model,parameters = adapt(self.model(),scope,42)
                self.assertEqual(any('lora_' in n for n,_ in parameters),'L' in scope)
                self.assertEqual(any('.visual.merger.' in n for n,_ in parameters),'M' in scope)
                self.assertFalse(any('lm_head' in n or 'embed_tokens' in n for n,_ in parameters))
                inp = self.inputs()
                labels = torch.tensor([[-100]*8+[6,7]])
                model.train()
                loss = model(**inp,labels=labels,use_cache=False).loss
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(any(p.grad is not None for p in model.parameters() if p.requires_grad))
                model.eval()
                stats = {m:dict(mean=torch.zeros(32),std=torch.ones(32)) for m in ('image','text')}
                hook = SymmetricMask(model,stats)
                try:
                    with torch.no_grad():
                        plain = model(**inp,use_cache=False).logits
                        with hook.use(inp,[7,8,9],'full'):
                            full = model(**inp,use_cache=False).logits
                        self.assertTrue(torch.equal(plain,full))
                        for condition in ('mask_image','mask_text'):
                            count = len(hook.logs)
                            with hook.use(inp,[7,8,9],condition):
                                out = model.generate(**inp,do_sample=False,use_cache=True,max_new_tokens=3,
                                                     eos_token_id=None,pad_token_id=0)
                            self.assertEqual(out.shape[1],13)
                            self.assertEqual(len(hook.logs),count+1)
                finally:
                    hook.close()
