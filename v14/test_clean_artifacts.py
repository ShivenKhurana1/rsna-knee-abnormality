"""Synthetic end-to-end saved-model verifier regression; not MRI evidence."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import clean_specialist as clean
import diagnostics
import verify_clean_specialist as verifier


class CleanArtifactTests(unittest.TestCase):
    def test_saved_models_replay_refit_and_corruption_rejection(self):
        rng = np.random.default_rng(1401)
        x = rng.normal(size=(40, 10)).astype(np.float32)
        y = rng.integers(0, 2, size=(40, 12)).astype(np.float32)
        ids = np.array([f'synthetic-study-{i}' for i in range(40)])
        groups = np.array([f'synthetic-group-{i//2}' for i in range(40)])
        original = diagnostics.compare
        def fast_compare(*args, **kwargs):
            kwargs['bootstrap'] = 100
            return original(*args, **kwargs)
        with tempfile.TemporaryDirectory() as folder, threadpool_limits(limits=2):
            root = Path(folder)
            (root/'features').mkdir()
            feature = root/'features/clean_features.npz'
            np.savez_compressed(feature, StudyInstanceUID=ids, GroupID=groups, labels=y, base=x, regional=np.c_[x,x**2])
            assignment = np.full(40, -1)
            for k, (_, va) in enumerate(clean.splits(groups, 5, 1401)):
                assignment[va] = k
            pd.DataFrame({'StudyInstanceUID':ids,'GroupID':groups,'fold':assignment}).to_csv(root/'predeclared_folds.csv',index=False)
            (root/'predeclared_protocol.json').write_text(json.dumps(clean.PROTOCOL))
            (root/'run_receipt.json').write_text(json.dumps({'recipe_sha256':clean.sha(root/'predeclared_protocol.json')}))
            with patch.object(diagnostics, 'compare', side_effect=fast_compare), patch.object(verifier, 'compare', side_effect=fast_compare), contextlib.redirect_stdout(io.StringIO()):
                clean.evaluate_features(feature, root/'evaluation')
                result = verifier.verify(root, refit=True)
            self.assertTrue(result['local_refit_performed'])
            self.assertLess(result['max_weight_replay_error'], 1e-12)
            self.assertFalse(result['exact_v13_improvement_verified'])
            with (root/'evaluation/reference_oof.csv').open('a') as stream:
                stream.write('corrupt\n')
            with self.assertRaisesRegex(ValueError, 'Artifact/hash mismatch'):
                verifier.verify(root)


if __name__ == '__main__':
    unittest.main()
