"""Training-only embedding statistics must not alter assembled model inputs."""
from types import SimpleNamespace
import unittest
import torch
from modalitygap.model import MomentCapture


class MomentCaptureTests(unittest.TestCase):
    def test_capture_observes_only_content_and_leaves_inputs_unchanged(self):
        class Decoder(torch.nn.Module):
            def forward(self,inputs_embeds):return inputs_embeds
        model=torch.nn.Module();model.config=SimpleNamespace(image_token_id=99);model.language_model=Decoder()
        inputs=dict(input_ids=torch.tensor([[1,99,99,4,5,6]]));x=torch.arange(24.).reshape(1,6,4)
        hook=MomentCapture(model)
        try:
            with hook.use(inputs,[4,5]):
                self.assertTrue(torch.equal(model.language_model(inputs_embeds=x),x))
                self.assertTrue(torch.equal(hook.observed['image'],x[0,[1,2]]))
                self.assertTrue(torch.equal(hook.observed['text'],x[0,[4,5]]))
                with self.assertRaises(ValueError):
                    with hook.use(inputs,[4,5]):pass
            self.assertTrue(torch.equal(model.language_model(inputs_embeds=x),x))
        finally:hook.close()
