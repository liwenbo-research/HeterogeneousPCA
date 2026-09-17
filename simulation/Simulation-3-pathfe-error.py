"""Focused PathFE error rerun for Simulation-3 threshold tuning."""
import argparse, os, random, sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src", "distributedpca"))
from feature_extraction_path import PathFE
from pca_utils import PCAUtils


def proj(f, p):
    f = np.asarray(f, dtype=float)
    if f.size == 0:
        return np.zeros((p, p))
    f = np.atleast_2d(f)
    if f.shape[0] != p:
        f = f.T
    return f @ np.linalg.pinv(f.T @ f) @ f.T


def one(seed, p, args):
    n = int(round(p / args.ratio))
    np.random.seed(seed)
    random.seed(seed)
    covs = PCAUtils.generate_cov_matrix(
        args.n_clients, args.orthonormal, [8, 8, args.n_clients + 1],
        args.lam, p, 2, private_rho=args.private_rho,
    )
    X = [np.random.multivariate_normal(np.zeros(p), cov, size=n) for cov in covs]
    ranks = [3] * args.n_clients
    epsilon = n ** (0.5 - 2 / args.a - args.epsilon0)
    path = PathFE().fit(
        X, epsilon, num_list=ranks, epsilon0=args.epsilon0,
        a=args.a, pair_n_method=args.pair_n_method,
    )

    true_global = np.sqrt(p) * np.linalg.svd(covs[0])[0][:, :2]
    pt = proj(true_global, p)
    comp = np.eye(p) - pt
    true_local = []
    for cov in covs:
        vals, vecs = np.linalg.eigh(comp @ cov @ comp)
        true_local.append(vecs[:, np.argsort(vals)[::-1][:1]] * np.sqrt(p))

    pg = proj(path.global_factors, p)
    tr = float(np.trace(pg @ pt) / 2)
    common_norm = sum(np.linalg.norm(x @ pt, "fro") ** 2 for x in X)
    diff = sum(np.linalg.norm(x @ pg - x @ pt, "fro") ** 2 for x in X)
    local_errors, recon = [], []
    for x, tl, el in zip(X, true_local, path.est_pf):
        pl, ptl = proj(el, p), proj(tl, p)
        local_errors.append(np.linalg.norm(pl - ptl, "fro") ** 2)
        recon.append(np.linalg.norm(x - x @ (pg + pl), "fro") ** 2 / x.shape[0])
    global_error = float(np.linalg.norm(pg - pt, "fro") ** 2)
    local_error = float(np.mean(local_errors))
    return {
        "p": p, "method": "pathfe", "seed": seed,
        "rho": args.private_rho, "orthonormal": args.orthonormal,
        "num_common": int(path.num_cf), "tr": tr,
        "scaled_projection_distance": np.sqrt(max(0.0, 1.0 - tr)),
        "global_subspace_error": global_error,
        "common_component_rrmse": np.sqrt(diff / max(common_norm, 1e-12)),
        "local_subspace_error": local_error,
        "subspace_error": global_error + local_error,
        "reconstruction_error": float(np.mean(recon)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--n-runs", type=int, default=50)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--p-list", type=int, nargs="+", default=[200, 500, 800])
    ap.add_argument("--n-clients", type=int, default=10)
    ap.add_argument("--ratio", type=float, default=1.0)
    ap.add_argument("--lam", type=float, nargs=3, default=[15, 5, 2.5])
    ap.add_argument("--epsilon0", type=float, default=0.10)
    ap.add_argument("--a", type=float, default=6.0)
    ap.add_argument("--pair-n-method", choices=("min", "harmonic"), default="min")
    ap.add_argument("--private-rho", type=float, default=0.2)
    ap.add_argument("--orthonormal", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rows = []
    for p in args.p_list:
        vals = Parallel(n_jobs=args.n_jobs)(delayed(one)(args.seed + i, p, args) for i in range(args.n_runs))
        frame = pd.DataFrame(vals).groupby(["p", "method"], as_index=False).mean(numeric_only=True)
        frame["n_runs"] = args.n_runs
        frame["private_rho"] = args.private_rho
        frame["epsilon0"] = args.epsilon0
        frame["a"] = args.a
        frame["pair_n_method"] = args.pair_n_method
        frame["lam"] = str(list(args.lam))
        rows.append(frame)
    out = pd.concat(rows, ignore_index=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
