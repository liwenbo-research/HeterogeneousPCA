import unittest

import numpy as np

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "distributedpca"))
from Generalized_C_Corr import GCC


class TestGCCGram(unittest.TestCase):
    def test_block_gram_matches_explicit_pair_matrix(self):
        rng = np.random.default_rng(321)
        factors = [rng.normal(size=(7, 2)), rng.normal(size=(7, 3)),
                   rng.normal(size=(7, 1))]
        gcc = GCC([2, 3, 1])
        phi = gcc._get_phi(factors)
        np.testing.assert_allclose(gcc._get_phi_gram(factors), phi.T @ phi,
                                   rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
