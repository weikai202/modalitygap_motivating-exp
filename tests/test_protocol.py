import json
import tempfile
import unittest
from pathlib import Path

from modalitygap.analysis import paired_bootstrap
from modalitygap.protocol import CONFIGS, GENRES, contributions, metrics, parse, write_jsonl


class ProtocolTests(unittest.TestCase):
    def test_matrix(self):
        self.assertEqual(len(CONFIGS),6)
        self.assertEqual(len(GENRES),23)
        self.assertEqual(len(set(GENRES)),23)
        self.assertIn('Western',GENRES)

    def test_invalid_binary_is_false_negative(self):
        rows = [dict(reference=['sarcastic'],prediction=[],invalid=True),
                dict(reference=['not_sarcastic'],prediction=['not_sarcastic'],invalid=False)]
        result = metrics('mmsd2',rows)
        self.assertEqual(result['macro_f1'],.5)
        self.assertEqual(result['accuracy'],.5)
        self.assertEqual(result['sarcastic_f1'],0)
        self.assertEqual(result['invalid_rate'],.5)

    def test_decoding_preserves_unexpected_structure_tokens(self):
        from modalitygap.runner import decode_prediction
        class Tokenizer:
            eos_token_id=0
            pad_token_id=9
            def decode(self,ids,skip_special_tokens=False):
                vocab={1:'sarcastic',7:'<|image_pad|>',0:'<|im_end|>',9:'<|endoftext|>'}
                return ''.join(vocab[i] for i in ids if not skip_special_tokens or i==1)
        tokenizer=Tokenizer()
        self.assertEqual(decode_prediction(tokenizer,'mmsd2',[1,0,9]),('sarcastic',['sarcastic'],False,None))
        raw,parsed,invalid,_=decode_prediction(tokenizer,'mmsd2',[7,1,0])
        self.assertIn('<|image_pad|>',raw)
        self.assertEqual(parsed,[])
        self.assertTrue(invalid)

    def test_decode_errors_are_recorded_as_invalid(self):
        from modalitygap.runner import decode_prediction
        class UnknownTokenTokenizer:
            def decode(self,*args,**kwargs):
                raise TypeError('unknown token ID')
        for dataset in ('mmsd2','mmimdb'):
            raw,prediction,invalid,error = decode_prediction(UnknownTokenTokenizer(),dataset,[151900])
            self.assertEqual(prediction,[])
            self.assertTrue(invalid)
            self.assertEqual(error,'TypeError')
            self.assertIn('151900',raw)

    def test_strict_parsing(self):
        self.assertEqual(parse('mmsd2','sarcastic explanation'),([],True))
        self.assertEqual(parse('mmsd2','Sarcastic'),([],True))
        self.assertEqual(parse('mmimdb','["drama", " Drama ", "COMEDY"]'),(['Comedy','Drama'],False))
        for raw in ('{}','[1]','["unknown"]','```json\n[]\n```'):
            self.assertTrue(parse('mmimdb',raw)[1])

    def test_negative_contributions(self):
        value = contributions(dict(full=.6,mask_image=.7,mask_text=.8))
        self.assertAlmostEqual(value['image_contribution'],-.1)
        self.assertAlmostEqual(value['gain_over_best_ablation'],-.2)

    def test_bootstrap_preserves_pairing_and_groups(self):
        rows = [dict(id=str(i),group=str(i//2),reference=['sarcastic'],prediction=['sarcastic'],invalid=False) for i in range(4)]
        predictions = {(c,e,m):rows for c in ('L-low','L-high') for e in (2,8)
                       for m in ('full','mask_image','mask_text')}
        result = paired_bootstrap('mmsd2',predictions,samples=20)
        self.assertTrue(all(v==dict(low=0.,high=0.) for v in result['pair_difference_intervals'].values()))
        self.assertTrue(all(v==dict(low=0.,high=0.) for v in result['gain_intervals'].values()))
        predictions['L-low',2,'full'] = rows[:-1]
        with self.assertRaises(ValueError):
            paired_bootstrap('mmsd2',predictions,samples=2)

    def test_vectorized_bootstrap_matches_naive_recomputation(self):
        import random
        from modalitygap.analysis import percentile
        for dataset in ('mmsd2','mmimdb'):
            labels=('sarcastic','not_sarcastic') if dataset=='mmsd2' else ('Drama','Comedy')
            rows=[dict(id=str(i),group=str(i//2),reference=[labels[i%2]],
                       prediction=[] if i%3==0 else [labels[(i+1)%2]] if i%3==1 else [labels[i%2]],
                       invalid=i%3==0) for i in range(7)]
            key=('L-low',2,'full')
            result=paired_bootstrap(dataset,{key:rows},samples=50,seed=123)
            groups=sorted({r['group'] for r in rows})
            rng=random.Random(123)
            expected=[]
            for _ in range(50):
                sample=[r for g in rng.choices(groups,k=len(groups)) for r in rows if r['group']==g]
                expected.append(metrics(dataset,sample)['score'])
            interval=result['score_intervals'][str(key)]
            self.assertAlmostEqual(interval['low'],percentile(expected,.025))
            self.assertAlmostEqual(interval['high'],percentile(expected,.975))


class DataTests(unittest.TestCase):
    def test_all_training_examples_are_kept_without_subsampling(self):
        from PIL import Image
        from modalitygap.data import prepare
        from modalitygap.protocol import read_jsonl
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for dataset in ('mmsd2','mmimdb'):
                rows=[]
                for i in range(32):
                    image=root/f'{i}.png'
                    Image.new('RGB',(8,8),(i*7,20,30)).save(image)
                    labels=['sarcastic' if i%2 else 'not_sarcastic'] if dataset=='mmsd2' else ['Drama','Comedy'] if i%2 else ['Drama','Action']
                    rows.append(dict(id=str(i),split='train' if i<30 else 'dev' if i==30 else 'test',
                                     image=str(image),text=f'Unique post {i}',labels=labels))
                source=root/f'{dataset}.jsonl'
                write_jsonl(source,rows)
                for suffix in ('a','b'):
                    prepare(source,root/f'{dataset}-{suffix}',dataset,'fixture')
                a,b=(root/f'{dataset}-{s}' for s in ('a','b'))
                self.assertEqual(read_jsonl(a/'train.jsonl'),read_jsonl(b/'train.jsonl'))
                meta=json.loads((a/'manifest.json').read_text())
                self.assertEqual(meta['pool_size'],meta['counts']['train'])
                self.assertEqual(meta['counts']['dev'],1)
                self.assertEqual(meta['counts']['test'],1)
                self.assertEqual(meta['counts']['train'],30)
                with self.assertRaises(ValueError):prepare(source,root/'subset',dataset,'fixture',pool_size=10)
                if dataset=='mmsd2':
                    self.assertEqual(meta['label_counts']['train'],{'not_sarcastic':15,'sarcastic':15})

    def test_freeze_filter_overlap_and_hash(self):
        from PIL import Image
        from modalitygap.data import prepare
        from modalitygap.runner import verified_data
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            for i in range(4):
                Image.new('RGB',(8,8),(i*30,0,0)).save(p/f'{i}.png')
            rows = [dict(id=str(i),split='train' if i<2 else 'dev' if i==2 else 'test',
                         image=f'{i}.png',text=f'post {i}',labels=['sarcastic' if i%2 else 'not_sarcastic']) for i in range(4)]
            broken=p/'broken.jpg'
            Image.new('RGB',(100,100),'red').save(broken)
            broken.write_bytes(broken.read_bytes()[:-20])
            with Image.open(broken) as im:
                im.verify() # passes even though decoding the JPEG payload fails
            rows.append(dict(id='broken',split='train',image='broken.jpg',text='broken image',labels=['sarcastic']))
            rows.append(dict(id='missing',split='train',image='absent.png',text='missing',labels=['sarcastic']))
            write_jsonl(p/'source.jsonl',rows)
            prepare(p/'source.jsonl',p/'manifest','mmsd2','author-sha')
            meta = verified_data(p/'manifest','mmsd2')
            self.assertEqual(meta['counts'],dict(train=2,dev=1,test=1))
            self.assertEqual(len(meta['excluded']),2)
            changed=dict(meta,requested_pool_size=2)
            from modalitygap.protocol import write_json
            write_json(p/'manifest/manifest.json',changed)
            with self.assertRaisesRegex(ValueError,'every retained'):verified_data(p/'manifest','mmsd2')
            write_json(p/'manifest/manifest.json',meta)
            with self.assertRaises(ValueError):
                prepare(p/'source.jsonl',p/'manifest','mmsd2','author-sha')
            with (p/'manifest'/'dev.jsonl').open('a') as f:
                f.write('\n')
            with self.assertRaises(ValueError):
                verified_data(p/'manifest','mmsd2')
            rows[2]['text'] = rows[0]['text']
            write_jsonl(p/'source.jsonl',rows)
            with self.assertRaises(ValueError):
                prepare(p/'source.jsonl',p/'overlap','mmsd2','author-sha')


if __name__=='__main__':
    unittest.main()
