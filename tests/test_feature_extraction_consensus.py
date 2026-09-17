import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "distributedpca"))
from feature_extraction_consensus import ConsensusFE


class ConsensusFETests(unittest.TestCase):
    def test_requires_full_client_compatibility(self):
        rng = np.random.default_rng(5)
        X = [rng.normal(size=(30, 8)) for _ in range(3)]
        model = ConsensusFE().fit(X, epsilon=2.0, num_list=[2, 2, 2])
        self.assertEqual(len(model.location), model.num_cf)
        for path in model.location:
            self.assertEqual(len(path), 3)


if __name__ == "__main__":
    unittest.main()
