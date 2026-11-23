import unittest
from types import SimpleNamespace

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None,'Install torch for intervention tensor checks')
class InterventionTests(unittest.TestCase):
    def test_protected_positions_noise_and_cached_decoding(self):
        from modalitygap.model import Intervention
        class Decoder(torch.nn.Module):
            config = SimpleNamespace(image_token_id=151655)
            def forward(self,inputs_embeds,cache_position=None):
                return inputs_embeds
        model = torch.nn.Module()
        model.language_model = Decoder()
        stats = {m:dict(mean=torch.ones(4),std=torch.ones(4)) for m in ('image','text')}
        hook = Intervention(model,stats)
        x = torch.arange(24,dtype=torch.float).reshape(1,6,4)
        inputs = dict(input_ids=torch.tensor([[1,151655,151655,4,5,6]]))
        try:
            with hook.use(dict(id='a'),inputs,[4],'full'):
                self.assertTrue(torch.equal(model.language_model(inputs_embeds=x),x))
            masked = []
            for _ in range(2):
                with hook.use(dict(id='a'),inputs,[4],'mask_image'):
                    masked.append(model.language_model(inputs_embeds=x))
                    cached = model.language_model(inputs_embeds=x[:,:1],cache_position=torch.tensor([6]))
                    self.assertTrue(torch.equal(cached,x[:,:1]))
            self.assertTrue(torch.equal(*masked))
            self.assertTrue(torch.equal(masked[0][:,[0,3,4,5]],x[:,[0,3,4,5]]))
            self.assertFalse(torch.equal(masked[0][:,1:3],x[:,1:3]))
            with hook.use(dict(id='b'),inputs,[4],'mask_image'):
                other = model.language_model(inputs_embeds=x)
            self.assertFalse(torch.equal(other[:,1:3],masked[0][:,1:3]))
            with hook.use(dict(id='a'),inputs,[4],'mask_text',scale=0):
                text = model.language_model(inputs_embeds=x)
            self.assertTrue(torch.equal(text[0,4],torch.ones(4)))
            self.assertTrue(torch.equal(text[:,:4],x[:,:4]))
        finally:
            hook.close()
