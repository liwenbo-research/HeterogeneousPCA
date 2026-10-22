import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "distributedpca"))
from zheng_estimate import ZhengEstimate


class ZhengSpectrumTests(unittest.TestCase):
    def test_gram_spectrum_matches_full_correlation_svd_when_n_below_p(self):
        rng = np.random.default_rng(98)
        X = rng.normal(size=(9, 14))
        X[:, 0] *= 2.0
        estimator = ZhengEstimate()
        actual = estimator.compute_eigenvalues(X)

        centered = X - X.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered / len(X)
        scale = np.diag(covariance) + 1e-8
        inverse = np.diag(scale ** -0.5)
        expected = np.linalg.svd(inverse @ covariance @ inverse,
                                 compute_uv=False)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
