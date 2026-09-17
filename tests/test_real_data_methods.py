import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "distributedpca"))

from real_data_methods import fit_simulation_methods
from feature_extraction import AutoFE, FE
from feature_extraction_path import PathFE


class RealDataMethodTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12345)
        self.X_train = [rng.normal(size=(48, 12)) for _ in range(3)]
        self.X_test = [rng.normal(size=(16, 12)) for _ in range(3)]

    def test_all_simulation_methods_fit_and_evaluate(self):
        estimates = fit_simulation_methods(
            self.X_train,
            num_spike_list=[3, 3, 3],
            shared_rank=1,
            target_rank=1,
            epsilon=5.0,
        )
        self.assertEqual(tuple(estimates), ("pathfe", "consensusfe", "apm", "cp", "gcc", "tk", "kt"))
        for estimate in estimates.values():
            self.assertEqual(estimate.global_basis.shape[0], 12)
            self.assertLessEqual(estimate.num_common, 3)
            self.assertTrue(np.isfinite(estimate.reconstruction_error(self.X_test)))

    def test_centered_model_uses_training_means_on_test_data(self):
        X_train = [np.array([[3.0, 2.0], [5.0, 2.0], [7.0, 2.0]])] * 2
        X_test = [np.array([[9.0, 2.0], [11.0, 2.0]])] * 2
        estimates = fit_simulation_methods(X_train, num_spike_list=[1, 1], methods=("tk",),
                                           shared_rank=1, center=True)
        self.assertLess(estimates["tk"].reconstruction_error(X_test), 1e-20)

    def test_original_fe_remains_explicitly_available(self):
        estimates = fit_simulation_methods(self.X_train, num_spike_list=[3, 3, 3],
                                           methods=("fe",), epsilon=5.0)
        self.assertEqual(tuple(estimates), ("fe",))

    def test_automatic_ranks_do_not_fit_original_fe(self):
        with patch.object(FE, "fit", side_effect=AssertionError("FE should not run")):
            estimates = fit_simulation_methods(self.X_train, methods=("pathfe",),
                                               spike_estimator="zheng")
        self.assertEqual(tuple(estimates), ("pathfe",))

    def test_auto_fe_defaults_to_pathfe(self):
        self.assertIsInstance(AutoFE().fe, PathFE)

    def test_rank_oracle_methods_require_shared_rank(self):
        with self.assertRaises(ValueError):
            fit_simulation_methods(self.X_train, num_spike_list=[2, 2, 2], methods=("tk",))

    def test_pc_based_methods_reject_zero_selected_rank(self):
        with self.assertRaises(ValueError):
            fit_simulation_methods(self.X_train, num_spike_list=[1, 0, 1], methods=("apm",))


if __name__ == "__main__":
    unittest.main()
