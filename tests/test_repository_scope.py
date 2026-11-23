"""Guard the English full-experiment-only public surface."""
import json,re,unittest
from pathlib import Path
from modalitygap.protocol import LABELS,CONFIGS

ROOT=Path(__file__).resolve().parents[1]


class RepositoryScopeTests(unittest.TestCase):
    def test_single_full_experiment_configuration(self):
        files=list((ROOT/'configs').glob('*.json'))
        self.assertEqual([p.name for p in files],['full-experiment.example.json'])
        cfg=json.loads(files[0].read_text())
        self.assertEqual(set(cfg['datasets']),{'mmsd2','mmimdb'})
        self.assertEqual(set(LABELS),{'mmsd2','mmimdb'})
        self.assertEqual(cfg['pool_size'],'all');self.assertEqual(set(cfg['configs']),set(CONFIGS))
        self.assertEqual(cfg['seeds'],[42,43,44]);self.assertEqual(cfg['masking_method'],'symmetric-attention-v2')

    def test_english_text_and_no_bundled_results(self):
        files=[ROOT/'README.md']
        for folder in ('docs','src','scripts','tests','configs','.github'):
            files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.md','.json','.py','.sh','.yml'))
        for p in files:
            self.assertIsNone(re.search(r'[\u3400-\u9fff]',p.read_text()),str(p.relative_to(ROOT)))
        self.assertFalse(list((ROOT/'docs').glob('*.json')))
        self.assertFalse(list((ROOT/'docs').glob('*.png')))
        self.assertTrue(all('result' not in p.name.lower() for p in (ROOT/'docs').iterdir()))

    def test_only_full_workflow_scripts_are_distributed(self):
        expected={'audit_predictions.py','download_full_data.py','download_mmimdb_raw.py','fetch_model.py','prepare_full_experiment.py','run_experiment.py','run_experiment.sh','full_experiment_status.py','lock_heldout_selection.py'}
        self.assertEqual({p.name for p in (ROOT/'scripts').iterdir() if p.is_file()},expected)
