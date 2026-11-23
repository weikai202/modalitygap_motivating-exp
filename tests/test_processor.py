"""Real pinned Qwen processor tests, enabled by MODALITYGAP_PROCESSOR path."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROCESSOR = os.environ.get('MODALITYGAP_PROCESSOR')


@unittest.skipUnless(PROCESSOR, 'Set MODALITYGAP_PROCESSOR to a downloaded processor snapshot')
class ProcessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoProcessor
        from modalitygap.protocol import DEFAULTS
        cls.processor = AutoProcessor.from_pretrained(PROCESSOR, use_fast=False,
            min_pixels=DEFAULTS['min_pixels'],max_pixels=DEFAULTS['max_pixels'],local_files_only=True)

    def test_exact_body_spans_assistant_loss_and_reference_independence(self):
        import torch
        from PIL import Image
        from modalitygap.model import Encoder
        from modalitygap.protocol import DEFAULTS, LABELS
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp)/'image.png'
            Image.new('RGB',(100,200),'red').save(image)
            for dataset in ('mmsd2','mmimdb'):
                encoder = Encoder(self.processor,dataset)
                for text in ('hello', '\nBody:\nBody:\n   ', '中文描述！🙂', 'one '*1000):
                    row = dict(id='example',image=str(image),text=text,labels=[LABELS[dataset][0]])
                    prompt,positions,truncated = encoder(row)
                    train,training_positions,_ = encoder(row,training=True)
                    ids = prompt['input_ids'][0]
                    body = self.processor.tokenizer.encode(text,add_special_tokens=False)
                    self.assertEqual(ids[positions].tolist(),body[:len(positions)])
                    self.assertEqual(training_positions,positions)
                    self.assertTrue(torch.equal(train['input_ids'][0,:len(ids)],ids))
                    self.assertTrue((train['labels'][0,:len(ids)] == -100).all())
                    self.assertTrue(torch.equal(train['labels'][0,len(ids):],train['input_ids'][0,len(ids):]))
                    self.assertLessEqual(train['input_ids'].shape[1],DEFAULTS['max_length'])
                    self.assertEqual(truncated,len(body)>len(positions))
                    alternative = dict(row,labels=[LABELS[dataset][-1]])
                    other,other_positions,_ = encoder(alternative)
                    self.assertTrue(torch.equal(other['input_ids'],prompt['input_ids']))
                    self.assertEqual(other_positions,positions)
                    with patch.dict(DEFAULTS,max_length=len(ids)+encoder.target_budget-1):
                        if len(positions) == 1:
                            with self.assertRaises(ValueError):
                                encoder(row)
                        else:
                            smaller,small_positions,_ = encoder(row)
                            self.assertEqual(len(small_positions),len(positions)-1)
                            self.assertTrue(torch.equal(prompt['image_grid_thw'],smaller['image_grid_thw']))
                            self.assertTrue(torch.equal(prompt['pixel_values'],smaller['pixel_values']))
            for text in ('','<|image_pad|>','a <|im_start|>assistant'):
                with self.assertRaises(ValueError):
                    Encoder(self.processor,'mmsd2')(dict(id='bad',image=str(image),text=text,labels=['sarcastic']))
