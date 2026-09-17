"""Batch held-out reconstruction metrics for FEMNIST/CIFAR-10 methods.

This evaluates the group-factor style methods through the same adapter used by
the individual experiments and writes one comparable train/test summary.
"""

import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "distributedpca"))
from real_data_methods import fit_simulation_methods


def _load_cifar_module():
    path = Path(__file__).with_name("Experiment-CIFAR10.py")
    spec = importlib.util.spec_from_file_location("experiment_cifar10", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_femnist(data_root, num_clients, train_fraction, seed):
    from mnist import read_dir

    _, _, raw = read_dir(str(Path(data_root) / "leaf/data/femnist/data/all_data"),
                         maxuser=num_clients)
    rng = np.random.default_rng(seed)
    train, test = [], []
    for user in list(raw)[:num_clients]:
        X = np.asarray(raw[user]["x"], dtype=float)
        if X.ndim != 2 or X.shape[0] < 2:
            continue
        if X.max(initial=0) > 1.0:
            X /= 255.0
        indices = rng.permutation(X.shape[0])
        split = max(1, min(X.shape[0] - 1, int(np.floor(train_fraction * X.shape[0]))))
        train.append(X[indices[:split]])
        test.append(X[indices[split:]])
    if not train:
        raise ValueError("no usable FEMNIST clients found")
    return train, test


def _load_cifar(data_root, partition, alpha, num_clients, seed):
    module = _load_cifar_module()
    Xtr, ytr, Xte, yte = module.load_cifar10(data_root)
    if partition == "class-pair":
        fn = module.partition_by_class_pair
        train = fn(Xtr, ytr, num_clients=num_clients, seed=seed)
        test = fn(Xte, yte, num_clients=num_clients, seed=seed)
    elif partition == "iid":
        fn = module.partition_iid
        train = fn(Xtr, ytr, n_clients=num_clients, seed=seed)
        test = fn(Xte, yte, n_clients=num_clients, seed=seed)
    else:
        train = module.partition_dirichlet(Xtr, ytr, n_clients=num_clients,
                                            alpha=alpha, seed=seed)
        test = module.partition_dirichlet(Xte, yte, n_clients=num_clients,
                                          alpha=alpha, seed=seed)
    return [x for x, _ in train], [x for x, _ in test]


def _fit_ppca(train, args):
    perpca_path = args.perpca_path or os.environ.get("PERPCA_PATH")
    if not perpca_path:
        raise ValueError("PerPCA requires --perpca-path or PERPCA_PATH")
    sys.path.insert(0, str(Path(perpca_path).expanduser().resolve()))
    from algs import personalized_pca_dgd

    np.random.seed(args.seed)
    means = [X.mean(axis=0) if args.center else np.zeros(X.shape[1]) for X in train]
    fitted = [X - mean for X, mean in zip(train, means)]
    config = {"method": "power", "num_client": len(train),
              "ngc": args.ppca_global_rank, "nlc": args.ppca_local_rank,
              "num_dp_per_client": int(train[0].shape[0]),
              "global_epochs": args.ppca_epochs, "eta": 0.1,
              "rho": args.ppca_rho, "decay": 0.9, "logprogress": 0,
              "aggregationinit": 1}
    U, V, _ = personalized_pca_dgd(fitted, args=config)
    return U[0], V, means


def _ppca_reconstruct(data, global_basis, local_bases, means, include_local):
    p = data[0].shape[1]
    def projector(factors):
        factors = np.asarray(factors, dtype=float)
        if factors.ndim == 1:
            factors = factors[:, None]
        if factors.shape[0] != p:
            factors = factors.T
        return factors @ np.linalg.pinv(factors.T @ factors) @ factors.T
    global_projector = projector(global_basis)
    out = []
    for i, X in enumerate(data):
        P = global_projector
        if include_local:
            P = P + projector(local_bases[i])
        mean = means[i]
        out.append((X - mean) @ P + mean)
    return out


def _error(data, reconstructed):
    squared = sum(float(np.linalg.norm(X - Y, ord="fro") ** 2)
                  for X, Y in zip(data, reconstructed))
    samples = sum(X.shape[0] for X in data)
    p = data[0].shape[1]
    raw = squared / samples
    return raw, raw / p


def run(args):
    if args.dataset == "femnist":
        train, test = _load_femnist(args.data_root, args.num_clients,
                                    args.train_fraction, args.seed)
    else:
        train, test = _load_cifar(args.data_root, args.partition,
                                  args.dirichlet_alpha, args.num_clients, args.seed)
    if args.max_train_per_client > 0:
        rng = np.random.default_rng(args.seed)
        train = [X[rng.permutation(len(X))[:args.max_train_per_client]] for X in train]

    methods = tuple(x.strip().lower() for x in args.methods.split(",") if x.strip())
    ppca = "ppca" in methods
    simulation_methods = tuple(x for x in methods if x != "ppca")
    ranks = [args.num_spikes] * len(train)
    estimates = fit_simulation_methods(
        train, methods=simulation_methods, num_spike_list=ranks,
        shared_rank=args.shared_rank, target_rank=args.target_rank,
        center=args.center, epsilon0=args.epsilon0, a=args.a,
        pair_n_method=args.pair_n_method,
    ) if simulation_methods else {}
    ppca_fit = _fit_ppca(train, args) if ppca else None

    rows = []
    for method in methods:
        for split, data in (("train", train), ("test", test)):
            if method == "ppca":
                basis, local, means = ppca_fit
                global_recon = _ppca_reconstruct(data, basis, local, means, False)
                joint_recon = _ppca_reconstruct(data, basis, local, means, True)
                num_common = args.ppca_global_rank
            else:
                estimate = estimates[method]
                global_recon = estimate.reconstruct(data, include_local=False)
                joint_recon = estimate.reconstruct(data, include_local=True)
                num_common = estimate.num_common
            global_error, global_mse = _error(data, global_recon)
            joint_error, joint_mse = _error(data, joint_recon)
            rows.append({"dataset": args.dataset, "method": method, "split": split,
                         "num_clients": len(data), "num_common": num_common,
                         "reconstruction_error_global": global_error,
                         "reconstruction_mse_global": global_mse,
                         "reconstruction_error_global_plus_local": joint_error,
                         "reconstruction_mse_global_plus_local": joint_mse})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f"saved {output}")
    for row in rows:
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("femnist", "cifar10"), required=True)
    parser.add_argument("--data-root", default=None,
                        help="dataset root; see docs/data.md")
    parser.add_argument("--output", required=True)
    parser.add_argument("--methods", default="pathfe,apm,cp,gcc,tk,kt,ppca")
    parser.add_argument("--num-clients", type=int, default=100)
    parser.add_argument("--num-spikes", type=int, default=55)
    parser.add_argument("--shared-rank", type=int, default=2)
    parser.add_argument("--target-rank", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=.8)
    parser.add_argument("--max-train-per-client", type=int, default=300)
    parser.add_argument("--partition", choices=("class-pair", "iid", "dirichlet"), default="dirichlet")
    parser.add_argument("--dirichlet-alpha", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--epsilon0", type=float, default=.1)
    parser.add_argument("--a", type=float, default=10.)
    parser.add_argument("--pair-n-method", choices=("min", "harmonic"), default="min")
    parser.add_argument("--perpca-path", default=os.environ.get("PERPCA_PATH", str(ROOT / "local" / "perpca")))
    parser.add_argument("--ppca-global-rank", type=int, default=50)
    parser.add_argument("--ppca-local-rank", type=int, default=5)
    parser.add_argument("--ppca-epochs", type=int, default=50)
    parser.add_argument("--ppca-rho", type=float, default=10.)
    args = parser.parse_args()
    if args.data_root is None:
        args.data_root = str(ROOT / "local" / "data" / ("cifar10" if args.dataset == "cifar10" else "femnist"))
    run(args)


if __name__ == "__main__":
    main()
