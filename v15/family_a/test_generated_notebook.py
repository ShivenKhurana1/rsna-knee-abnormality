import hashlib
import json
import unittest
from pathlib import Path

from build_notebook import FAMILY_A_MODULES


HERE = Path(__file__).resolve().parent


class GeneratedNotebookTests(unittest.TestCase):
    def test_training_notebook_embeds_current_sources_and_safe_config(self):
        path = HERE / 'rsna-knee-v15-family-a-train.ipynb'
        notebook = json.loads(path.read_text(encoding='utf-8'))
        code = '\n'.join(''.join(cell['source']) for cell in notebook['cells']
                         if cell['cell_type'] == 'code')
        for name in FAMILY_A_MODULES:
            source = (HERE / name).read_text(encoding='utf-8')
            self.assertIn(repr(source), code, f'generated notebook has stale {name}')
        transfer_source = (HERE.parent / 'transfer_audit.py').read_text(encoding='utf-8')
        self.assertIn(repr(transfer_source), code)
        self.assertIn("device='cuda'", code)
        self.assertIn('backbone_lr=8e-6', code)
        self.assertIn("crossfit_policy=(arm_name == 'auxiliary')", code)
        self.assertIn("os.environ.get('FAMILY_A_SHARD', 'all')", code)
        self.assertIn('expert_fraction=0.10', code)
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])

    def test_manifest_hashes_the_checked_notebook(self):
        path = HERE / 'rsna-knee-v15-family-a-train.ipynb'
        manifest = json.loads((HERE / 'family_a_train_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['notebook_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(manifest['device_required'], 'cuda')
        self.assertEqual(manifest['auxiliary_policy_selection'],
                         'cross-fitted within each outer fold')


if __name__ == '__main__':
    unittest.main()
