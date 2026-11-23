import csv
import json
import tempfile
import unittest
from pathlib import Path

from modalitygap.analysis import analyze
from modalitygap.protocol import CONDITIONS, metrics, write_jsonl
from modalitygap.runner import write_csv


class AnalysisTests(unittest.TestCase):
    def test_reversal_confirmation_and_early_top_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for seed in (42,43,44):
                for config in ('L-low','L-high'):
                    run=root/'runs'/'mmsd2'/config/f'seed_{seed}'
                    run.mkdir(parents=True)
                    records=[]
                    for epoch in range(9):
                        for condition in CONDITIONS:
                            k=(2 if epoch<=2 or seed==44 else 4) if config=='L-low' else 3
                            if condition!='full':
                                k=2
                            rows=[]
                            labels=('sarcastic','not_sarcastic')
                            for i in range(10):
                                label=labels[i%2]
                                prediction=label if i//2<k else labels[1-i%2]
                                rows.append(dict(id=str(i),reference=[label],prediction=[prediction],invalid=False))
                            records.append(dict(epoch=epoch,condition=condition,variant=condition,
                                                training_seconds=epoch*100,validation_seconds=1,**metrics('mmsd2',rows)))
                            if epoch in (2,8):
                                write_jsonl(run/'predictions'/f'epoch_{epoch}_{condition}.jsonl',rows)
                    write_csv(run/'metrics.csv',records)
            analyze(root/'runs','mmsd2',root/'figures',bootstrap_samples=20)
            report=json.loads((root/'figures'/'mmsd2'/'analysis.json').read_text())
            self.assertFalse(report['negative_result'])
            pair=next(r for r in report['confirmation'] if r['a']=='L-low' and r['b']=='L-high')
            self.assertEqual(pair['reversal_seeds'],[42,43])
            self.assertTrue(pair['confirmed_two_of_three'])
            rank=next(r for r in report['rankings'] if r['seed']==42)
            self.assertEqual(rank['pairwise_rank_agreement'],0)
            self.assertFalse(rank['early_top_k_retains_final_winner']['1'])
            self.assertTrue(rank['early_top_k_retains_final_winner']['2'])

    def test_end_to_end_figures_and_negative_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for config in ('L-low','L-high'):
                run = root/'artifacts'/'mmsd2'/config/'seed_42'
                run.mkdir(parents=True)
                records = []
                for epoch in range(9):
                    for condition in CONDITIONS:
                        rows = [dict(id=str(i),reference=[label],prediction=[label],invalid=False)
                                for i,label in enumerate(('sarcastic','not_sarcastic'))]
                        records.append(dict(epoch=epoch,condition=condition,variant=condition,
                                            training_seconds=epoch*100,validation_seconds=1,**metrics('mmsd2',rows)))
                        if epoch in (2,8):
                            write_jsonl(run/'predictions'/f'epoch_{epoch}_{condition}.jsonl',rows)
                write_csv(run/'metrics.csv',records)
            analyze(root/'artifacts','mmsd2',root/'figures',bootstrap_samples=10)
            out = root/'figures'/'mmsd2'
            self.assertEqual(len(list(out.glob('*.png'))),7)
            report = json.loads((out/'analysis.json').read_text())
            self.assertTrue(report['negative_result'])
            self.assertEqual(report['actual_main_evaluations'],54)
            self.assertEqual(len(report['pairs']),2)
