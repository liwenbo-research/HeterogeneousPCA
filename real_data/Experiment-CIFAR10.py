"""CIFAR-10 real-data experiment using the Simulation-3 methods."""

import argparse
import csv
import os
import pickle
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "distributedpca"))
from real_data_methods import fit_simulation_methods


def _load_batch(path):
    with open(path, "rb") as handle:
        data = pickle.load(handle, encoding="bytes")
    return np.asarray(data[b"data"], dtype=np.float64) / 255.0, np.asarray(data[b"labels"], dtype=np.int64)


def load_cifar10(root):
    root = Path(root)
    train_x, train_y = zip(*[_load_batch(root / f"data_batch_{i}") for i in range(1, 6)])
    test_x, test_y = _load_batch(root / "test_batch")
    return np.concatenate(train_x), np.concatenate(train_y), test_x, test_y


def partition_by_class_pair(X, y, n_clients=20, seed=12345):
    if n_clients != 20:
        raise ValueError("the Section 7.3 CIFAR-10 design uses exactly 20 clients")
    rng = np.random.default_rng(seed)
    # Each class is split into four shards and each client receives two class
    # shards. Thus every image is used exactly once and every client has two
    # classes, while classes recur across clients as in the federated setup.
    shards = {}
    for cls in range(10):
        indices = np.flatnonzero(y == cls)
        rng.shuffle(indices)
        shards[cls] = np.array_split(indices, 4)
    occurrences = [0] * 10
    clients = []
    for client in range(n_clients):
        c1, c2 = client % 10, (client + 1) % 10
        i1, i2 = occurrences[c1], occurrences[c2]
        occurrences[c1] += 1
        occurrences[c2] += 1
        indices = np.concatenate((shards[c1][i1], shards[c2][i2]))
        rng.shuffle(indices)
        clients.append((X[indices], y[indices]))
    return clients


def partition_iid(X, y, n_clients=20, seed=12345):
    """IID client partition with every client receiving all classes."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    return [(X[idx], y[idx]) for idx in np.array_split(indices, n_clients)]


def partition_dirichlet(X, y, n_clients=20, alpha=5.0, seed=12345):
    """Mildly heterogeneous partition; every client can receive all classes."""
    if alpha <= 0:
        raise ValueError("dirichlet alpha must be positive")
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(n_clients)]
    for cls in np.unique(y):
        cls_indices = np.flatnonzero(y == cls)
        rng.shuffle(cls_indices)
        proportions = rng.dirichlet(np.full(n_clients, alpha, dtype=float))
        counts = np.floor(proportions * len(cls_indices)).astype(int)
        counts[-1] += len(cls_indices) - int(counts.sum())
        start = 0
        for client, count in enumerate(counts):
            client_indices[client].extend(cls_indices[start:start + count].tolist())
            start += count
    clients = []
    for indices in client_indices:
        indices = np.asarray(indices, dtype=int)
        rng.shuffle(indices)
        clients.append((X[indices], y[indices]))
    return clients


def run(args):
    Xtr, ytr, Xte, yte = load_cifar10(args.data_root)
    partition = partition_by_class_pair
    if args.partition == "iid":
        partition = partition_iid
    elif args.partition == "dirichlet":
        partition = lambda X, y, seed: partition_dirichlet(X, y, alpha=args.dirichlet_alpha, seed=seed)
    train_clients = partition(Xtr, ytr, seed=args.seed)
    test_clients = partition(Xte, yte, seed=args.seed)
    X_train = [x for x, _ in train_clients]
    X_test = [x for x, _ in test_clients]
    if args.method == "ppca":
        sys.path.insert(0, str(Path(args.perpca_path).expanduser().resolve()))
        from algs import personalized_pca_dgd, loss
        np.random.seed(args.seed)
        ppca_args = {"ngc": args.ppca_global_rank, "nlc": args.ppca_local_rank,
                     "num_client": len(X_train), "global_epochs": args.ppca_epochs,
                     "eta": 0.1, "rho": 1.0}
        U, V, _ = personalized_pca_dgd(X_train, ppca_args)
        rows = [{"method": "ppca", "split": split,
                 "global_rank": args.ppca_global_rank, "local_rank": args.ppca_local_rank,
                 "reconstruction_error_global": float(loss(data, U[0], 0)),
                 "reconstruction_error_global_plus_local": float(loss(data, U, V))}
                for split, data in (("train", X_train), ("test", X_test))]
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
        print(f"saved {output}")
        for row in rows: print(row)
        return
    num_spike_list = None if args.num_spikes <= 0 else [args.num_spikes] * len(X_train)
    estimates = fit_simulation_methods(
        X_train,
        num_spike_list=num_spike_list,
        spike_estimator=args.spike_estimator,
        shared_rank=args.shared_rank,
        target_rank=args.target_rank,
        methods=(args.method,),
        center=args.center,
        epsilon0=args.epsilon0,
        a=args.a,
        pair_n_method=args.pair_n_method,
    )
    rows = []
    for method, estimate in estimates.items():
        for split, data in (("train", X_train), ("test", X_test)):
            rows.append({"method": method, "split": split,
                         "num_common": estimate.num_common,
                         "reconstruction_error_global": estimate.reconstruction_error(data, include_local=False),
                         "reconstruction_error_global_plus_local": estimate.reconstruction_error(data, include_local=True)})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {output}")
    for row in rows:
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(ROOT / "local" / "data" / "cifar10"),
                        help="directory containing data_batch_1..5 and test_batch")
    parser.add_argument("--output", default="results/real_data/cifar10/simulation-methods.csv")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--partition", choices=("class-pair", "iid", "dirichlet"), default="dirichlet")
    parser.add_argument("--dirichlet-alpha", type=float, default=5.0)
    parser.add_argument("--num-spikes", type=int, default=55,
                        help="fixed FE rank budget; PPCA reference uses ngc=50 and nlc=5")
    parser.add_argument("--spike-estimator", choices=("zheng", "mbe", "yao", "liu", "ke"),
                        default="zheng", help="automatic spike estimator when --num-spikes <= 0")
    parser.add_argument("--shared-rank", type=int, default=2)
    parser.add_argument("--method", choices=("fe", "pathfe", "consensusfe", "apm", "cp", "gcc", "tk", "kt", "ppca"), default="pathfe")
    parser.add_argument("--perpca-path", default=os.environ.get("PERPCA_PATH", str(ROOT / "local" / "perpca")))
    parser.add_argument("--ppca-global-rank", type=int, default=50)
    parser.add_argument("--ppca-local-rank", type=int, default=5)
    parser.add_argument("--ppca-epochs", type=int, default=50)
    parser.add_argument("--target-rank", type=int, default=None,
                        help="total target rank for KT; local rank is target minus shared")
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--epsilon0", type=float, default=0.1,
                        help="FE threshold epsilon_0")
    parser.add_argument("--a", type=float, default=10.0,
                        help="FE threshold exponent parameter a")
    parser.add_argument("--pair-n-method", choices=("min", "harmonic"), default="min",
                        help="effective pair sample size used by FE")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
