"""New task semantics, sampling, and real processor boundary regression."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

from modalitygap.protocol import LABELS, SINGLE_LABEL, generation_limit, metrics, parse, target, task_prompt

TASKS=('newsclippings','crisismmd','hatefulmemes')


class ExpansionProtocolTests(unittest.TestCase):
    def test_exact_single_label_targets_and_macro_f1(self):
        for task in TASKS:
            self.assertIn(task,SINGLE_LABEL)
            for label in LABELS[task]:
                self.assertEqual(target(task,[label]),label)
                self.assertEqual(parse(task,label),([label],False))
                self.assertEqual(parse(task,json.dumps([label])),([],True))
                self.assertEqual(parse(task,label+' explanation'),([],True))
                self.assertIn(label,task_prompt(task))
            a,b=LABELS[task]
            result=metrics(task,[dict(reference=[a],prediction=[a],invalid=False),
                                 dict(reference=[b],prediction=[],invalid=True)])
            self.assertEqual(result['score'],.5)
            self.assertEqual(result['accuracy'],.5)
            self.assertEqual(result['invalid_rate'],.5)
            self.assertEqual(generation_limit(task),8)

    def test_balanced_sampling_preserves_official_split_and_excludes_shared_text(self):
        spec=importlib.util.spec_from_file_location('acquisition',Path(__file__).resolve().parents[1]/'scripts/prepare_expansion.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        rows=[dict(id=str(i),text=f'body {i}',labels=[str(i%2)]) for i in range(40)]
        used=set();train=m.balanced(rows,'train',16,('0','1'),used)
        dev=m.balanced(rows,'dev',8,('0','1'),used)
        self.assertFalse({r['id'] for r in train}&{r['id'] for r in dev})
        self.assertTrue(all(r['split']=='train' for r in train))
        self.assertEqual([sum(r['labels']==[x] for r in train) for x in ('0','1')],[8,8])
        self.assertEqual(train,m.balanced(rows,'train',16,('0','1'),set()))

    def test_prepare_rejects_multilabel_exports_for_all_new_tasks(self):
        from modalitygap.data import prepare
        from modalitygap.protocol import write_jsonl
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for task in TASKS:
                file=root/f'{task}.jsonl';write_jsonl(file,[dict(id='x',split='train',labels=list(LABELS[task]))])
                with self.assertRaisesRegex(ValueError,'Invalid labels'):
                    prepare(file,root/task,task,'fixture')


@unittest.skipUnless(os.environ.get('MODALITYGAP_PROCESSOR'),'Requires cached pinned processor')
class ExpansionProcessorTests(unittest.TestCase):
    def test_all_new_prompts_preserve_spans_and_never_include_reference(self):
        from transformers import AutoProcessor
        from PIL import Image
        from modalitygap.model import Encoder
        from modalitygap.protocol import DEFAULTS
        import torch
        proc=AutoProcessor.from_pretrained(os.environ['MODALITYGAP_PROCESSOR'],use_fast=False,
            min_pixels=DEFAULTS['min_pixels'],max_pixels=DEFAULTS['max_pixels'],local_files_only=True)
        with tempfile.TemporaryDirectory() as tmp:
            img=Path(tmp)/'x.png';Image.new('RGB',(128,128),'red').save(img)
            for task in TASKS:
                encoder=Encoder(proc,task);row=dict(id='one',image=str(img),text='A caption about a public event.',labels=[LABELS[task][0]])
                a,positions,_=encoder(row)
                b,other,_=encoder(dict(row,labels=[LABELS[task][1]]))
                self.assertTrue(torch.equal(a['input_ids'],b['input_ids']))
                self.assertEqual(positions,other)
                self.assertEqual(a['input_ids'][0,positions].tolist(),proc.tokenizer.encode(row['text'],add_special_tokens=False))
                training,_,_=encoder(row,training=True);n=a['input_ids'].shape[1]
                self.assertTrue((training['labels'][0,:n]==-100).all())
                answer=training['input_ids'][0,n:].tolist()
                self.assertLessEqual(len(answer),generation_limit(task))
                self.assertEqual(answer,proc.tokenizer.encode(target(task,row['labels'])+proc.tokenizer.eos_token,add_special_tokens=False))
