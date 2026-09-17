"""FEMNIST real-data experiment for the Simulation-3 methods."""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "distributedpca"))
sys.path.insert(0, str(ROOT / "local" / "perpca"))
from real_data_methods import fit_simulation_methods


def run(args):
    try:
        from mnist import read_dir
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "FEMNIST requires the LEAF mnist.py loader; see docs/data.md."
        ) from exc
    all_data_dir = Path(args.data_root) / "leaf/data/femnist/data/all_data"
    _, _, raw = read_dir(str(all_data_dir), maxuser=args.num_clients)
    rng = np.random.default_rng(args.seed)
    X_train, X_test = [], []
    for user in list(raw)[:args.num_clients]:
        X = np.asarray(raw[user]["x"], dtype=float)
        if X.ndim != 2 or X.shape[0] < 2:
            continue
        indices = rng.permutation(X.shape[0])
        split = max(1, min(X.shape[0] - 1, int(np.floor(args.train_fraction * X.shape[0]))))
        X_train.append(X[indices[:split]])
        X_test.append(X[indices[split:]])
    if not X_train:
        raise ValueError("no usable FEMNIST clients found")
    if args.method == "ppca":
        from algs import personalized_pca_dgd, loss
        np.random.seed(args.seed)
        ppca_args = {
            "method": "power", "num_client": len(X_train),
            "ngc": args.ppca_global_rank, "nlc": args.ppca_local_rank,
            "num_dp_per_client": int(X_train[0].shape[0]),
            "global_epochs": args.ppca_epochs, "eta": 0.1, "rho": 10,
            "decay": 0.9, "logprogress": 0, "aggregationinit": 1,
        }
        U, V, _ = personalized_pca_dgd(X_train, args=ppca_args)
        rows = []
        for split, data in (("train", X_train), ("test", X_test)):
            rows.append({
                "method": "ppca", "split": split, "num_clients": len(data),
                "global_rank": args.ppca_global_rank, "local_rank": args.ppca_local_rank,
                "reconstruction_error_global": float(loss(data, U[0], 0)),
                "reconstruction_error_global_plus_local": float(loss(data, U, V)),
            })
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
        print(f"saved {output}")
        for row in rows: print(row)
        return
    requested_num_spikes = None if args.num_spikes <= 0 else [args.num_spikes] * len(X_train)
    estimates = fit_simulation_methods(
        X_train,
        methods=(args.method,),
        num_spike_list=requested_num_spikes,
        spike_estimator=args.spike_estimator,
        epsilon=args.epsilon,
        epsilon0=args.epsilon0,
        a=args.a,
        pair_n_method=args.pair_n_method,
        center=args.center,
        shared_rank=args.shared_rank,
        target_rank=args.target_rank,
    )
    estimate = estimates[args.method]
    rows = []
    for split, data in (("train", X_train), ("test", X_test)):
        rows.append({
            "method": args.method, "split": split, "num_clients": len(data),
            "num_spikes": str(estimate.num_spike_list), "spike_estimator": args.spike_estimator,
            "num_common": estimate.num_common,
            "a": args.a, "epsilon0": args.epsilon0,
            "reconstruction_error_global": estimate.reconstruction_error(data, include_local=False),
            "reconstruction_error_global_plus_local": estimate.reconstruction_error(data, include_local=True),
        })
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
    parser.add_argument("--data-root", default=str(ROOT / "local" / "data" / "femnist"),
                        help="LEAF data root containing leaf/data/femnist")
    parser.add_argument("--output", default="results/real_data/femnist/pathfe-only.csv")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num-clients", type=int, default=100)
    parser.add_argument("--method", choices=("fe", "pathfe", "consensusfe", "apm", "cp", "gcc", "tk", "kt", "ppca"), default="pathfe")
    parser.add_argument("--num-spikes", type=int, default=0,
                        help="fixed total rank; 0 estimates one rank per client with FE")
    parser.add_argument("--spike-estimator", choices=("zheng", "yao", "combine"), default="zheng")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--epsilon0", type=float, default=0.1)
    parser.add_argument("--a", type=float, default=10.0)
    parser.add_argument("--pair-n-method", choices=("min", "harmonic"), default="min")
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--shared-rank", type=int, default=None)
    parser.add_argument("--target-rank", type=int, default=None,
                        help="total target rank for KT; local rank is target minus shared")
    parser.add_argument("--ppca-global-rank", type=int, default=50)
    parser.add_argument("--ppca-local-rank", type=int, default=5)
    parser.add_argument("--ppca-epochs", type=int, default=50)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
