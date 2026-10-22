import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "distributedpca"))
from feature_extraction_path import _top_projected_basis, extract_complete_paths
from pca_utils import PCAUtils


class PathExtractionTests(unittest.TestCase):
    def test_projected_local_basis_gram_matches_covariance_eigenspace(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(8, 13))
        Q, _ = np.linalg.qr(rng.normal(size=(13, 3)))
        complement = np.eye(13) - Q @ Q.T
        actual = _top_projected_basis(X, complement, 3)
        residual = complement @ (X.T @ X / len(X)) @ complement
        values, vectors = np.linalg.eigh(residual)
        expected = vectors[:, np.argsort(values)[::-1][:3]]
        np.testing.assert_allclose(actual @ actual.T, expected @ expected.T,
                                   rtol=1e-10, atol=1e-10)

    def test_finds_all_vertex_disjoint_complete_paths(self):
        pairs = [
            np.array([[0, 0], [0, 1], [1, 1]]),
            np.array([[0, 0], [1, 1]]),
        ]
        ratios = [
            np.array([[.9, .8], [0, .85]]),
            np.array([[.95, 0], [0, .9]]),
        ]
        paths, scores = extract_complete_paths(pairs, ratios, [.7, .7])
        self.assertEqual(paths, [[0, 0, 0], [1, 1, 1]])
        self.assertEqual(len(scores), 2)

    def test_rejects_partial_paths(self):
        pairs = [np.array([[0, 0]]), np.array([[1, 0]])]
        ratios = [np.array([[.9]]), np.array([[0], [.9]])]
        paths, _ = extract_complete_paths(pairs, ratios, [.7, .7])
        self.assertEqual(paths, [])

    def test_cached_spectra_match_original_pair_statistic(self):
        rng = np.random.default_rng(9)
        X, Y = rng.normal(size=(30, 8)), rng.normal(size=(25, 8))
        pairs, ratio, inner = PCAUtils.identify_prod(X, Y, 3, 3, 2.0)
        def spectrum(Z):
            values, vectors = np.linalg.eigh(Z.T @ Z / Z.shape[0])
            order = np.argsort(values)[::-1]
            return values[order], vectors[:, order], Z.shape[0]
        cached = PCAUtils.identify_prod_from_spectra(*spectrum(X), *spectrum(Y), 3, 3, 2.0)
        self.assertTrue(np.array_equal(pairs, cached[0]))
        self.assertTrue(np.allclose(ratio, cached[1]))
        self.assertTrue(np.allclose(inner, cached[2]))


if __name__ == "__main__":
    unittest.main()
