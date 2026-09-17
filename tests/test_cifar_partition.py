import importlib.util
from pathlib import Path
import unittest
import numpy as np

module_path = Path(__file__).resolve().parents[1] / "real_data/Experiment-CIFAR10.py"
spec = importlib.util.spec_from_file_location("experiment_cifar10", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
partition_iid = module.partition_iid
partition_dirichlet = module.partition_dirichlet


class CifarPartitionTests(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(2000).reshape(200, 10)
        self.y = np.repeat(np.arange(10), 20)

    def test_iid_keeps_all_examples_and_classes(self):
        clients = partition_iid(self.X, self.y, n_clients=20, seed=1)
        self.assertEqual(sum(len(x) for x, _ in clients), 200)
        self.assertEqual(set(np.concatenate([labels for _, labels in clients])), set(range(10)))

    def test_dirichlet_keeps_all_examples_and_is_reproducible(self):
        a = partition_dirichlet(self.X, self.y, n_clients=20, alpha=5, seed=1)
        b = partition_dirichlet(self.X, self.y, n_clients=20, alpha=5, seed=1)
        self.assertEqual(sum(len(x) for x, _ in a), 200)
        self.assertTrue(all(np.array_equal(x1, x2) for (x1, _), (x2, _) in zip(a, b)))


if __name__ == "__main__":
    unittest.main()
