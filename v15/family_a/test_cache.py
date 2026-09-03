import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import cache
from contract import GROUP, IMG, N_SLOT


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_write_read_roundtrip(self):
        imgs = np.zeros((N_SLOT, GROUP, IMG, IMG), dtype=np.uint8)
        imgs[0, 0, 0, 0] = 200
        mask = np.array([True, False, True, False, True, False])
        cache.write_study(self.tmp, 'study-1', imgs, mask)
        loaded_imgs, loaded_mask = cache.load_study(self.tmp, 'study-1')
        self.assertTrue(np.array_equal(loaded_imgs, imgs))
        self.assertTrue(np.array_equal(loaded_mask.astype(bool), mask))

    def test_rejects_wrong_shape(self):
        bad = np.zeros((N_SLOT, GROUP, IMG, IMG - 1), dtype=np.uint8)
        mask = np.ones(N_SLOT, dtype=bool)
        with self.assertRaises(ValueError):
            cache.write_study(self.tmp, 'study-1', bad, mask)

    def test_rejects_all_missing_slots(self):
        imgs = np.zeros((N_SLOT, GROUP, IMG, IMG), dtype=np.uint8)
        mask = np.zeros(N_SLOT, dtype=bool)
        with self.assertRaises(ValueError):
            cache.write_study(self.tmp, 'study-1', imgs, mask)

    def test_rejects_unsafe_study_id(self):
        imgs = np.zeros((N_SLOT, GROUP, IMG, IMG), dtype=np.uint8)
        mask = np.ones(N_SLOT, dtype=bool)
        with self.assertRaises(ValueError):
            cache.write_study(self.tmp, '../escape', imgs, mask)

    def test_manifest_reports_slots_present_and_hash(self):
        ids = ['a', 'b', 'c']
        manifest = cache.synth(self.tmp, ids, seed=0)
        self.assertEqual(manifest['n_studies'], 3)
        self.assertEqual(set(manifest['studies']), set(ids))
        for uid in ids:
            self.assertGreaterEqual(manifest['studies'][uid]['n_slots_present'], 3)
            self.assertEqual(len(manifest['studies'][uid]['sha256']), 64)

    def test_build_manifest_raises_on_missing_study(self):
        cache.synth(self.tmp, ['a', 'b'], seed=0)
        with self.assertRaises(FileNotFoundError):
            cache.build_manifest(self.tmp, ['a', 'b', 'missing'])

    def test_load_batch_stacks_in_requested_order(self):
        cache.synth(self.tmp, ['a', 'b'], seed=0)
        imgs, masks = cache.load_batch(self.tmp, ['b', 'a'])
        imgs_a, _ = cache.load_study(self.tmp, 'a')
        self.assertTrue(np.array_equal(imgs[1], imgs_a))


if __name__ == '__main__':
    unittest.main()
