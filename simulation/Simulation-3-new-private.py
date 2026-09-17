"""Simulation-3 runs for the K+1 private-block construction.

Modes:
  common: estimated common rank for FE, PathFE, APM, GCC, and CP.
  error: global/local/reconstruction errors for FE, PathFE, APM, GCC, CP,
         KnowledgeTransfer, and PerPCA.
"""
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
ALG = os.path.join(os.path.dirname(HERE), "src", "distributedpca")
sys.path.insert(0, ALG)

from apm import APM
from Circular_Projection import CP
from Generalized_C_Corr import GCC
from feature_extraction import FE
from feature_extraction_path import PathFE
from knowledge_transfer import KnowledgeTransfer
from pca_utils import PCAUtils


def projector(factors, p):
    factors = np.asarray(factors, dtype=float)
    if factors.size == 0:
        return np.zeros((p, p))
    factors = np.atleast_2d(factors)
    if factors.shape[0] != p:
        factors = factors.T
    return factors @ np.linalg.pinv(factors.T @ factors) @ factors.T


def generate(seed, p, args):
    n = int(round(p / args.ratio))
    np.random.seed(seed)
    random.seed(seed)
    covs = PCAUtils.generate_cov_matrix(
        args.n_clients,
        args.orthonormal,
        [8, 8, args.n_clients + 1],
        args.lam,
        p,
        2,
        private_rho=args.private_rho,
    )
    X = [np.random.multivariate_normal(np.zeros(p), cov, size=n) for cov in covs]
    return X, covs, n


def fit_rank_methods(X, args, n):
    ranks = [3] * args.n_clients
    epsilon = n ** (0.5 - 2 / args.a - args.epsilon0)
    out = {}
    rank_methods = [("pathfe", PathFE)]
    if args.include_fe:
        rank_methods.append(("fe", FE))
    for name, cls in rank_methods:
        method = cls()
        method.fit(
            X,
            epsilon,
            num_list=ranks,
            epsilon0=args.epsilon0,
            a=args.a,
            pair_n_method=args.pair_n_method,
        )
        out[name] = method
    for name, cls in (("apm", APM), ("gcc", GCC), ("cp", CP)):
        method = cls(ranks)
        method.fit([z.copy() for z in X])
        out[name] = method
    return out


def common_one(seed, p, args):
    X, _, n = generate(seed, p, args)
    methods = fit_rank_methods(X, args, n)
    return [{
        "p": p,
        "seed": seed,
        "rho": args.private_rho,
        "orthonormal": args.orthonormal,
        "lam": str(list(args.lam)),
        "method": name,
        "num_common": int(method.num_cf if name in ("fe", "pathfe") else method.num_common),
    } for name, method in methods.items()]


def true_locals(covs, true_global, p):
    pt = projector(true_global, p)
    comp = np.eye(p) - pt
    locals_ = []
    for cov in covs:
        vals, vecs = np.linalg.eigh(comp @ cov @ comp)
        locals_.append(vecs[:, np.argsort(vals)[::-1][:1]] * np.sqrt(p))
    return locals_


def error_one(seed, p, args):
    X, covs, n = generate(seed, p, args)
    methods = fit_rank_methods(X, args, n)
    ranks = [3] * args.n_clients
    true_global = np.sqrt(p) * np.linalg.svd(covs[0])[0][:, :2]
    true_global_projector = projector(true_global, p)
    true_local = true_locals(covs, true_global, p)

    if not args.perpca_path:
        raise RuntimeError("--perpca-path or PERPCA_PATH is required for error mode")
    if args.perpca_path not in sys.path:
        sys.path.insert(0, args.perpca_path)
    from algs import personalized_pca_dgd

    kt = KnowledgeTransfer(ranks, target_index=0, shared_rank=2, target_rank=2)
    kt.fit([z.copy() for z in X])
    ppca_u, ppca_v, _ = personalized_pca_dgd(
        X,
        {"ngc": 2, "nlc": 1, "num_client": args.n_clients, "global_epochs": args.ppca_epochs, "eta": 0.1, "rho": 1.0},
    )
    methods.update(
        kt=kt,
        ppca=(ppca_u[0] * np.sqrt(p), ppca_v, 2),
    )

    rows = []
    for name in (*(("fe",) if args.include_fe else ()), "pathfe", "apm", "gcc", "cp", "kt", "ppca"):
        method = methods[name]
        if name == "ppca":
            gf, local_factors, num_common = method
        else:
            gf = method.global_factors
            local_factors = getattr(method, "est_pf", None) if name in ("fe", "pathfe") else getattr(method, "local_factor_list", None)
            num_common = int(method.num_cf if name in ("fe", "pathfe") else method.num_common)
        pg = projector(gf, p)
        tr = float(np.trace(pg @ true_global_projector) / 2)
        common_norm = sum(np.linalg.norm(x @ true_global_projector, "fro") ** 2 for x in X)
        diff = sum(np.linalg.norm(x @ pg - x @ true_global_projector, "fro") ** 2 for x in X)
        row = {
            "p": p, "seed": seed, "rho": args.private_rho, "orthonormal": args.orthonormal,
            "lam": str(list(args.lam)), "method": name, "num_common": num_common,
            "tr": tr, "scaled_projection_distance": np.sqrt(max(0.0, 1.0 - tr)),
            "global_subspace_error": np.linalg.norm(pg - true_global_projector, "fro") ** 2,
            "common_component_rrmse": np.sqrt(diff / max(common_norm, 1e-12)),
        }
        if local_factors is None:
            row.update(local_subspace_error=np.nan, subspace_error=np.nan, reconstruction_error=np.nan)
        else:
            local_errors, reconstruction = [], []
            for x, tl, el in zip(X, true_local, local_factors):
                pl = projector(el, p)
                ptl = projector(tl, p)
                local_errors.append(np.linalg.norm(pl - ptl, "fro") ** 2)
                reconstruction.append(np.linalg.norm(x - x @ (pg + pl), "fro") ** 2 / x.shape[0])
            row["local_subspace_error"] = float(np.mean(local_errors))
            row["subspace_error"] = row["global_subspace_error"] + row["local_subspace_error"]
            row["reconstruction_error"] = float(np.mean(reconstruction))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("common", "error"), required=True)
    ap.add_argument("--include-fe", action="store_true", help="also run the original FE comparator")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--n-runs", type=int, default=50)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--p-list", type=int, nargs="+", default=[100, 300, 500])
    ap.add_argument("--n-clients", type=int, default=10)
    ap.add_argument("--ratio", type=float, default=1.0)
    ap.add_argument("--lam", type=float, nargs=3, required=True)
    ap.add_argument("--epsilon0", type=float, default=0.1)
    ap.add_argument("--a", type=float, default=10.0)
    ap.add_argument("--pair-n-method", choices=("min", "harmonic"), default="min")
    ap.add_argument("--private-rho", type=float, default=0.2)
    ap.add_argument("--ppca-epochs", type=int, default=50)
    ap.add_argument("--perpca-path", default=os.environ.get(
        "PERPCA_PATH", os.path.join(os.path.dirname(HERE), "local", "perpca")))
    ap.add_argument("--orthonormal", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = []
    for p in args.p_list:
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(common_one if args.mode == "common" else error_one)(args.seed + i, p, args)
            for i in range(args.n_runs)
        )
        rows.extend(x for result in results for x in result)
    raw = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if args.mode == "common":
        grouped = raw.groupby(["p", "method"], as_index=False).agg(
            estimated_common_mean=("num_common", "mean"),
            estimated_common_std=("num_common", "std"),
            under_rate=("num_common", lambda x: np.mean(x < 2)),
            over_rate=("num_common", lambda x: np.mean(x > 2)),
        )
        grouped["true_common_rank"] = 2
        grouped["n_runs"] = args.n_runs
        grouped["seed"] = args.seed
        grouped["orthonormal"] = args.orthonormal
        grouped["private_rho"] = args.private_rho
        grouped["lam"] = str(list(args.lam))
        grouped.to_csv(args.output, index=False)
        print(grouped.to_string(index=False))
    else:
        grouped = raw.groupby(["p", "method"], as_index=False).mean(numeric_only=True)
        grouped["n_runs"] = args.n_runs
        grouped["seed"] = args.seed
        grouped["orthonormal"] = args.orthonormal
        grouped["private_rho"] = args.private_rho
        grouped["lam"] = str(list(args.lam))
        grouped.to_csv(args.output, index=False)
        print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()
