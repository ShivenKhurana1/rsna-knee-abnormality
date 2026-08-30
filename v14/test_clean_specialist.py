"""Synthetic software tests only; these do not measure knee MRI accuracy."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

import clean_specialist as clean
from clean_features import pixels_to_rgb


class CleanSpecialistTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(24)
        x = rng.normal(size=(40, 15))
        y = rng.integers(0, 2, size=(40, 12)).astype(float)
        groups = np.array([f'group-{i//2}' for i in range(40)])
        return x, y, groups

    def test_groups_stay_together_and_split_is_label_independent(self):
        x, y, groups = self.fixture()
        covered = []
        for tr, va in clean.splits(groups, 5, 1401):
            self.assertFalse(set(groups[tr]) & set(groups[va]))
            covered.extend(va)
        self.assertEqual(sorted(covered), list(range(40)))
        with self.assertRaises(ValueError):
            clean.splits(['x', 'x'], 2, 1)

    def test_outer_labels_and_features_do_not_change_fitting(self):
        x, y, groups = self.fixture()
        tr, va = clean.splits(groups, 5, 1401)[0]
        protocol = copy.deepcopy(clean.PROTOCOL)
        protocol['regularization_grid'] = [1.0, 0.1]
        a = clean.train_outer(x, x**2, y, groups, tr, va, 42, protocol)
        changed = y.copy()
        changed[va] = 1-changed[va]
        b = clean.train_outer(x, x**2, changed, groups, tr, va, 42, protocol)
        for name in a[0]:
            np.testing.assert_array_equal(a[0][name], b[0][name])
        altered_x = x.copy()
        altered_x[va] += 500
        c = clean.train_outer(altered_x, altered_x**2, y, groups, tr, va, 42, protocol)
        np.testing.assert_array_equal(a[1]['mean'], c[1]['mean'])
        np.testing.assert_array_equal(a[1]['weight'], c[1]['weight'])
        self.assertEqual(a[4]['blend_decisions'], c[4]['blend_decisions'])

    def test_unaffected_targets_are_exact_and_weights_reload(self):
        x, y, groups = self.fixture()
        tr, va = clean.splits(groups, 5, 1401)[0]
        result, model, *_ = clean.train_outer(x, x**2, y, groups, tr, va, 41)
        other = [j for j in range(12) if j not in clean.FOCUS]
        for name in result:
            np.testing.assert_array_equal(result[name][:, other], result['reference'][:, other])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'head.npz'
            np.savez_compressed(path, **model)
            with np.load(path, allow_pickle=False) as f:
                loaded = dict(f)
            np.testing.assert_array_equal(clean.predict_head(model, x), clean.predict_head(loaded, x))

    def test_row_space_fit_matches_full_primal(self):
        x, y, _ = self.fixture()
        x = np.c_[x, x**2, x**3]
        y = y[:, :1]
        m = clean.fit_head(x, y, .1)
        z = (x-m['mean'])/m['scale']/np.sqrt(x.shape[1])
        def obj(theta):
            logits = z@theta[:-1]+theta[-1]
            error = (expit(logits)-y[:,0])/len(y)
            return np.mean(np.logaddexp(0,logits)-y[:,0]*logits)+.05*(theta[:-1]@theta[:-1]), np.r_[z.T@error+.1*theta[:-1],error.sum()]
        r = minimize(obj, np.zeros(x.shape[1]+1), jac=True, method='L-BFGS-B', options={'gtol': 1e-8, 'ftol': 1e-12})
        self.assertTrue(r.success)
        np.testing.assert_allclose(clean.predict_head(m,x)[:,0], expit(z@r.x[:-1]+r.x[-1]), atol=1e-5)

    def test_bad_labels_features_and_overlap_fail(self):
        x, y, groups = self.fixture()
        with self.assertRaises(ValueError):
            clean.fit_head(x, y*.5, 1.)
        with self.assertRaises(ValueError):
            clean.fit_head(x*np.nan, y, 1.)
        with self.assertRaises(ValueError):
            clean.train_outer(x, x, y, groups, np.arange(30), np.arange(25,40), 1)

    def test_pixel_aspect_polarity_and_bad_images(self):
        class Slice(dict):
            pass
        s = Slice(PhotometricInterpretation='MONOCHROME2')
        s.pixel_array = np.arange(32*64).reshape(32,64)
        normal = pixels_to_rgb(s, 224)
        self.assertEqual(normal.shape, (3,224,224))
        self.assertTrue((normal[:,:56]==0).all())
        s['PhotometricInterpretation'] = 'MONOCHROME1'
        inverted = pixels_to_rgb(s, 224)
        self.assertLess(np.abs((normal+inverted)[:,60:160]-1).max(), .02)
        s.pixel_array = np.ones((32,64))
        with self.assertRaises(ValueError):
            pixels_to_rgb(s,224)


if __name__ == '__main__':
    unittest.main()
