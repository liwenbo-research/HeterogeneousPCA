"""Export held-out FEMNIST/CIFAR-10 reconstructions for visual comparison.

Methods are fitted on client training matrices only.  Test images are then
projected with the fitted global and (when available) local subspaces.
"""

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "distributedpca"))
from real_data_methods import fit_simulation_methods


def _cifar_batch(path):
    with open(path, "rb") as handle:
        obj = pickle.load(handle, encoding="bytes")
    return np.asarray(obj[b"data"], dtype=float) / 255.0, np.asarray(obj[b"labels"])


def load_cifar(root):
    train = [_cifar_batch(Path(root) / f"data_batch_{i}") for i in range(1, 6)]
    test_x, test_y = _cifar_batch(Path(root) / "test_batch")
    return np.concatenate([x for x, _ in train]), np.concatenate([y for _, y in train]), test_x, test_y


def partition_by_class_pair(X, y, n_clients, seed):
    rng = np.random.default_rng(seed)
    shards = {}
    for cls in range(10):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        shards[cls] = np.array_split(idx, max(1, n_clients // 5))
    occurrences = [0] * 10
    clients = []
    for i in range(n_clients):
        c1, c2 = i % 10, (i + 1) % 10
        s1, s2 = occurrences[c1], occurrences[c2]
        occurrences[c1] += 1; occurrences[c2] += 1
        parts = [shards[c1][s1], shards[c2][s2]]
        idx = np.concatenate(parts)
        rng.shuffle(idx)
        clients.append(X[idx])
    return clients


def _save_image(arr, path, dataset):
    if dataset == "femnist":
        plt.imsave(path, np.clip(arr.reshape(28, 28), 0, 1), cmap="gray", vmin=0, vmax=1)
    else:
        # CIFAR stores channel-major vectors.
        image = np.clip(arr.reshape(3, 32, 32).transpose(1, 2, 0), 0, 1)
        plt.imsave(path, image)


def _image_array(arr, dataset):
    if dataset == "femnist":
        return np.clip(arr.reshape(28, 28), 0, 1)
    return np.clip(arr.reshape(3, 32, 32).transpose(1, 2, 0), 0, 1)


def _projector(factors, p):
    factors = np.asarray(factors, dtype=float)
    if factors.size == 0:
        return np.zeros((p, p))
    factors = np.atleast_2d(factors)
    if factors.shape[0] != p:
        factors = factors.T
    return factors @ np.linalg.pinv(factors.T @ factors) @ factors.T


def _fit_perpca(train, args, global_rank=None, local_ranks=None):
    """Fit the external PerPCA implementation and return global/local bases."""
    perpca_path = args.perpca_path or os.environ.get("PERPCA_PATH")
    if not perpca_path:
        raise ValueError("PerPCA requires --perpca-path or PERPCA_PATH")
    sys.path.insert(0, str(Path(perpca_path).expanduser().resolve()))
    from algs import personalized_pca_dgd

    np.random.seed(args.seed)
    ppca_args = {
        "method": "power", "num_client": len(train),
        "ngc": args.ppca_global_rank if global_rank is None else int(global_rank),
        "nlc": args.ppca_local_rank if local_ranks is None else [int(x) for x in local_ranks],
        "num_dp_per_client": int(train[0].shape[0]),
        "global_epochs": args.ppca_epochs, "eta": 0.1, "rho": args.ppca_rho,
        "decay": 0.9, "logprogress": 0, "aggregationinit": 1,
    }
    U, V, _ = personalized_pca_dgd(train, args=ppca_args)
    return np.asarray(U[0], dtype=float), [np.asarray(item, dtype=float) for item in V]


def _project_reconstruction(data, global_basis, local_basis=None, means=None):
    """Project n x p matrices using PerPCA bases, restoring optional means."""
    p = data[0].shape[1]
    global_projector = _projector(global_basis, p)
    outputs = []
    for i, X in enumerate(data):
        mean = np.zeros(p) if means is None else means[i]
        centered = X - mean
        projector = global_projector
        if local_basis is not None:
            projector = projector + _projector(local_basis[i], p)
        outputs.append(centered @ projector + mean)
    return outputs


def run(args):
    rng = np.random.default_rng(args.seed)
    if args.dataset == "femnist":
        sys.path.insert(0, str(Path(args.perpca_path).expanduser().resolve()))
        from mnist import read_dir
        _, _, raw = read_dir(str(Path(args.data_root) / "leaf/data/femnist/data/all_data"), maxuser=args.num_clients)
        train, test = [], []
        for user in list(raw)[:args.num_clients]:
            x = np.asarray(raw[user]["x"], dtype=float)
            if x.max(initial=0) > 1.0:
                x /= 255.0
            idx = rng.permutation(len(x))
            split = max(1, min(len(x) - 1, int(args.train_fraction * len(x))))
            train.append(x[idx[:split]])
            test.append(x[idx[split:]])
    else:
        _, ytr, xte, yte = load_cifar(args.data_root)
        # Load train once more only for partitioning; this keeps the test set held out.
        xtr = np.concatenate([_cifar_batch(Path(args.data_root) / f"data_batch_{i}")[0] for i in range(1, 6)])
        train = partition_by_class_pair(xtr, ytr, args.num_clients, args.seed)
        test = partition_by_class_pair(xte, yte, args.num_clients, args.seed)
        if args.max_train_per_client > 0:
            train = [x[rng.permutation(len(x))[:args.max_train_per_client]] for x in train]

    methods = tuple(item.strip().lower() for item in args.methods.split(",") if item.strip())
    ppca_requested = "ppca" in methods
    simulation_methods = tuple(method for method in methods if method != "ppca")
    ranks = [args.num_spikes] * len(train)
    transfer_common = None
    transfer_private = None
    if args.transfer_pathfe_ranks:
        pathfe = fit_simulation_methods(
            train, methods=("pathfe",), num_spike_list=ranks,
            center=args.center, epsilon0=args.epsilon0, a=args.a,
        )["pathfe"]
        ranks = pathfe.num_spike_list
        transfer_common = int(pathfe.num_common)
        transfer_private = [int(basis.shape[1]) for basis in pathfe.local_bases]
        transfer_ranks = [transfer_common + rank for rank in transfer_private]
        base_methods = tuple(method for method in simulation_methods
                             if method not in ("pathfe", "kt"))
        estimates = fit_simulation_methods(
            train, methods=base_methods, num_spike_list=ranks, shared_rank=args.shared_rank,
            target_rank=args.target_rank, center=args.center, epsilon0=args.epsilon0, a=args.a,
        ) if base_methods else {}
        if "pathfe" in simulation_methods:
            estimates["pathfe"] = pathfe
        if "kt" in simulation_methods:
            estimates["kt"] = fit_simulation_methods(
                train, methods=("kt",), num_spike_list=transfer_ranks,
                shared_rank=transfer_common, target_rank=transfer_ranks[0],
                center=args.center, epsilon0=args.epsilon0, a=args.a,
            )["kt"]
    else:
        estimates = fit_simulation_methods(
            train, methods=simulation_methods, num_spike_list=ranks, shared_rank=args.shared_rank,
            target_rank=args.target_rank, center=args.center, epsilon0=args.epsilon0, a=args.a,
        ) if simulation_methods else {}
    if ppca_requested:
        means = [x.mean(axis=0) if args.center else np.zeros(x.shape[1]) for x in train]
        fitted_train = [x - mean for x, mean in zip(train, means)]
        global_basis, local_bases = _fit_perpca(
            fitted_train, args,
            global_rank=transfer_common,
            local_ranks=transfer_private,
        )
        estimates["ppca"] = (
            global_basis, local_bases,
            args.ppca_global_rank if transfer_common is None else transfer_common,
        )
    out = Path(args.output) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    selected = args.images_per_client
    reconstructed = {}
    for method, estimate in estimates.items():
        if method == "ppca":
            global_basis, local_bases, num_common = estimate
            means = [x.mean(axis=0) if args.center else np.zeros(x.shape[1]) for x in train]
            global_all = _project_reconstruction(test, global_basis, means=means)
            joint_all = _project_reconstruction(test, global_basis, local_bases, means=means)
            reconstructed[method] = (global_all, joint_all)
        else:
            num_common = estimate.num_common
            reconstructed[method] = (estimate.reconstruct(test, include_local=False),
                                      estimate.reconstruct(test, include_local=True))
    for client, data in enumerate(test):
        chosen = np.arange(min(selected, len(data)))
        for local_index in chosen:
            original = data[local_index]
            original_path = out / f"original-c{client:03d}-i{local_index:03d}.png"
            _save_image(original, original_path, args.dataset)
            rows.append({"dataset": args.dataset, "method": "original", "reconstruction_type": "original",
                         "client": client, "test_index": int(local_index), "mse": 0.0,
                         "psnr": float("inf"), "num_common": "", "private_ranks": "",
                         "rank_source": "", "output": str(original_path)})
        for method, (global_all, joint_all) in reconstructed.items():
            recon_global = global_all[client]
            recon_joint = joint_all[client]
            for local_index in chosen:
                original = data[local_index]
                for kind, matrix in (("global", recon_global), ("global_plus_local", recon_joint)):
                    image = matrix[local_index]
                    mse = float(np.mean((image - original) ** 2))
                    psnr = float(10 * np.log10(1.0 / max(mse, 1e-12)))
                    path = out / f"{method}-{kind}-c{client:03d}-i{local_index:03d}.png"
                    _save_image(image, path, args.dataset)
                    rows.append({"dataset": args.dataset, "method": method, "reconstruction_type": kind,
                                 "client": client, "test_index": int(local_index), "num_common": num_common,
                                 "private_ranks": "" if transfer_private is None else str(transfer_private),
                                 "rank_source": "PathFE" if args.transfer_pathfe_ranks else "configured",
                                 "mse": mse,
                                 "psnr": psnr, "output": str(path)})
        for local_index in chosen:
            panels = [_image_array(data[local_index], args.dataset)]
            labels = ["Original"]
            for method, (global_all, joint_all) in reconstructed.items():
                panels.extend([_image_array(global_all[client][local_index], args.dataset),
                               _image_array(joint_all[client][local_index], args.dataset)])
                labels.extend([f"{method} global", f"{method} + local"])
            fig, axes = plt.subplots(1, len(panels), figsize=(2.2 * len(panels), 2.7))
            for ax, panel, label in zip(np.atleast_1d(axes), panels, labels):
                ax.imshow(panel, cmap="gray" if args.dataset == "femnist" else None, vmin=0, vmax=1)
                ax.set_title(label, fontsize=8); ax.axis("off")
            fig.tight_layout(pad=.4)
            fig.savefig(out / f"comparison-c{client:03d}-i{local_index:03d}.png", dpi=150)
            plt.close(fig)
    with (out / "detail.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f"saved {out} ({len(rows)} reconstructions)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("femnist", "cifar10"), required=True)
    p.add_argument("--data-root", default=None,
                   help="dataset root; see docs/data.md")
    p.add_argument("--output", default="results/real_data/reconstruction")
    p.add_argument("--methods", default="pathfe,apm,cp,gcc,tk,kt")
    p.add_argument("--num-clients", type=int, default=100)
    p.add_argument("--num-spikes", type=int, default=55)
    p.add_argument("--shared-rank", type=int, default=2)
    p.add_argument("--target-rank", type=int, default=None)
    p.add_argument("--images-per-client", type=int, default=2)
    p.add_argument("--max-train-per-client", type=int, default=300)
    p.add_argument("--train-fraction", type=float, default=.8)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--center", action="store_true")
    p.add_argument("--epsilon0", type=float, default=.1)
    p.add_argument("--a", type=float, default=10.)
    p.add_argument("--perpca-path", default=os.environ.get("PERPCA_PATH", str(ROOT / "local" / "perpca")))
    p.add_argument("--ppca-global-rank", type=int, default=50)
    p.add_argument("--ppca-local-rank", type=int, default=5)
    p.add_argument("--ppca-epochs", type=int, default=50)
    p.add_argument("--ppca-rho", type=float, default=10.0)
    p.add_argument("--transfer-pathfe-ranks", action="store_true",
                   help="use PathFE's common and client-private ranks for KT and PerPCA")
    args = p.parse_args()
    if args.data_root is None:
        args.data_root = str(ROOT / "local" / "data" / ("cifar10" if args.dataset == "cifar10" else "femnist"))
    run(args)


if __name__ == "__main__":
    main()
