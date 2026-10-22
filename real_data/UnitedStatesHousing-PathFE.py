"""Reproduce housing ranks, method metrics, and residual correlations together."""

import csv
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "archive" / "paper-feature-extraction" / "Plots" / "pca_real_data" / "states_csv"
OUTPUT_DIR = ROOT / "results" / "real_data" / "housing"
sys.path.insert(0, str(ROOT / "src" / "distributedpca"))

from zheng_estimate import ZhengEstimate
from apm import APM
from Circular_Projection import CP
from feature_extraction_path import PathFE
from Generalized_C_Corr import GCC
from knowledge_transfer import KnowledgeTransfer


def load_state_returns():
    states = []
    matrices = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        prices = pd.read_csv(path, index_col=0).to_numpy(dtype=float)
        returns = (np.log(prices[1:] / prices[:-1]).T * 100.0)
        if not np.isfinite(returns).all():
            raise ValueError(f"non-finite housing returns in {path.name}")
        states.append(path.stem)
        matrices.append(returns)
    if len(states) != 16:
        raise ValueError(f"expected 16 states, found {len(states)}")
    if len({x.shape[1] for x in matrices}) != 1:
        raise ValueError("all state matrices must have a common time dimension")
    return states, matrices


def average_absolute_correlation(x, y):
    x = x[np.isfinite(x).all(axis=1)]
    y = y[np.isfinite(y).all(axis=1)]
    x_std = x.std(axis=1, ddof=1)
    y_std = y.std(axis=1, ddof=1)
    x = x[x_std > 0]
    y = y[y_std > 0]
    if not len(x) or not len(y):
        return np.nan
    x = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, ddof=1, keepdims=True)
    y = (y - y.mean(axis=1, keepdims=True)) / y.std(axis=1, ddof=1, keepdims=True)
    return float(np.abs(x @ y.T / (x.shape[1] - 1)).mean())


def correlation_matrix(residuals):
    count = len(residuals)
    result = np.empty((count, count), dtype=float)
    for i in range(count):
        for j in range(i, count):
            result[i, j] = result[j, i] = average_absolute_correlation(residuals[i], residuals[j])
    return result


def relative_importance(data, global_factors, local_factors):
    """Compute the paper's county-averaged RIG, RIF, and residual RIE."""
    p = data[0].shape[1]
    rig, rif, rie = [], [], []
    for matrix, local in zip(data, local_factors):
        common_loading = matrix @ global_factors / p
        local_loading = matrix @ local / p
        total_energy = np.linalg.norm(matrix, axis=1) ** 2
        common_share = np.linalg.norm(common_loading, axis=1) ** 2 / total_energy
        local_share = np.linalg.norm(local_loading, axis=1) ** 2 / total_energy
        rig.append(float(np.mean(common_share) * p))
        rif.append(float(np.mean(local_share) * p))
        rie.append(float(1.0 - rig[-1] - rif[-1]))

    weights = np.asarray([matrix.shape[0] for matrix in data], dtype=float)
    weights /= weights.sum()
    result = []
    for name, values in (("RIG", rig), ("RIF", rif), ("RIE", rie)):
        result.append({
            "mean": float(np.mean(values)),
            "w_mean": float(np.dot(values, weights)),
            "by_state": values,
        })
    return result


def add_metric_rows(rows, states, rank_source, method, data, global_factors, local_factors):
    metrics = relative_importance(data, global_factors, local_factors)
    for i, state in enumerate(states):
        rows.append({"rank_source": rank_source, "method": method, "state": state,
                     "RIG": metrics[0]["by_state"][i],
                     "RIF": metrics[1]["by_state"][i],
                     "RIE": metrics[2]["by_state"][i]})
    rows.append({"rank_source": rank_source, "method": method, "state": "Mean",
                 "RIG": metrics[0]["mean"], "RIF": metrics[1]["mean"],
                 "RIE": metrics[2]["mean"]})
    rows.append({"rank_source": rank_source, "method": method, "state": "W-Mean",
                 "RIG": metrics[0]["w_mean"], "RIF": metrics[1]["w_mean"],
                 "RIE": metrics[2]["w_mean"]})


def fit_baseline(estimator, data, **kwargs):
    estimator.fit(data, **kwargs)
    return estimator


def main():
    states, data = load_state_returns()
    np.random.seed(int(os.environ.get("HOUSING_SEED", "12345")))
    spike_estimator = ZhengEstimate()
    ranks = [int(spike_estimator.estimate(matrix)) for matrix in data]

    a = float(os.environ.get("HOUSING_A", "10"))
    epsilon0 = float(os.environ.get("HOUSING_EPSILON0", "0.1"))
    pair_epsilons = []
    for left, right in zip(data[:-1], data[1:]):
        effective_n = min(left.shape[0], right.shape[0])
        pair_epsilons.append(float(effective_n ** (1.0 / 2.0 - 2.0 / a - epsilon0)))

    pathfe_estimate = PathFE().fit(data, epsilon=pair_epsilons, num_list=ranks)
    private_ranks = [int(basis.shape[1]) for basis in pathfe_estimate.est_pf]
    expected_private_ranks = [max(rank - pathfe_estimate.num_cf, 0) for rank in ranks]
    if private_ranks != expected_private_ranks:
        raise RuntimeError("FE private basis dimensions disagree with estimated ranks")
    p = data[0].shape[1]
    fe_rows = []
    add_metric_rows(
        fe_rows, states, "pathfe-estimated", "fe", data,
        pathfe_estimate.global_factors,
        [basis * np.sqrt(p) for basis in pathfe_estimate.est_pf],
    )
    all_metric_rows = list(fe_rows)
    for name, estimator in (
        ("apm", fit_baseline(APM(ranks), data, method="GAP")),
        ("cp", fit_baseline(CP(ranks), data)),
        ("gcc", fit_baseline(GCC(ranks), data)),
    ):
        if estimator.num_common > min(ranks):
            raise RuntimeError(f"{name}: common rank exceeds a state total rank")
        add_metric_rows(
            all_metric_rows, states, "pathfe-estimated", name, data,
            estimator.global_factors, estimator.local_factor_list,
        )
    global_projector = pathfe_estimate.global_factors @ pathfe_estimate.global_factors.T / p
    after_global = [x - x @ global_projector for x in data]
    after_local = []
    for x, local_basis in zip(after_global, pathfe_estimate.est_pf):
        local_projector = local_basis @ local_basis.T
        after_local.append(x - x @ local_projector)

    perpca_path = Path(os.environ.get("PERPCA_PATH", str(ROOT / "local" / "perpca"))).expanduser().resolve()
    if not (perpca_path / "algs.py").is_file():
        raise FileNotFoundError(f"PerPCA implementation not found: {perpca_path / 'algs.py'}")
    sys.path.insert(0, str(perpca_path))
    from algs import personalized_pca_dgd

    rank_scenarios = (
        ("fixed-rank3-common2", 2, [3] * len(data), [1] * len(data)),
        ("pathfe-estimated", int(pathfe_estimate.num_cf),
         [int(pathfe_estimate.num_cf + basis.shape[1]) for basis in pathfe_estimate.est_pf],
         [int(basis.shape[1]) for basis in pathfe_estimate.est_pf]),
    )
    output_rows = {}
    estimated_rank_rows = []
    for rank_source, common_rank, estimated_ranks, private_ranks in rank_scenarios:
        rows = []
        estimated_rank_rows.extend(
            {"rank_source": rank_source, "state": state, "total_rank": total_rank,
             "common_rank": common_rank, "private_rank": private_rank}
            for state, total_rank, private_rank in zip(states, estimated_ranks, private_ranks)
        )
        kt = KnowledgeTransfer(
            num_spike_list=estimated_ranks,
            target_index=0,
            shared_rank=common_rank,
            target_rank=estimated_ranks[0],
        ).fit(data)
        add_metric_rows(
            rows, states, rank_source, "kt-est", data,
            kt.global_factors,
            [factors for factors in kt.local_factor_list],
        )

        np.random.seed(int(os.environ.get("PERPCA_SEED", "12345")))
        ppca_args = {
            "ngc": common_rank,
            "nlc": private_ranks,
            "num_client": len(data),
            "num_dp_per_client": int(data[0].shape[0]),
            "global_epochs": int(os.environ.get("PERPCA_EPOCHS", "50")),
            "eta": 0.1,
            "rho": float(os.environ.get("PERPCA_RHO", "1.0")),
            "decay": 0.9,
            "aggregationinit": 1,
        }
        ppca_global, ppca_local, _ = personalized_pca_dgd(data, ppca_args)
        ppca_global_factors = ppca_global[0] * np.sqrt(p)
        ppca_local_factors = [factors * np.sqrt(p) for factors in ppca_local]
        add_metric_rows(
            rows, states, rank_source, "ppca-est", data,
            ppca_global_factors, ppca_local_factors,
        )
        output_rows[rank_source] = rows
        all_metric_rows.extend(rows)

    for row in all_metric_rows:
        if row["rank_source"] != "pathfe-estimated" or row["state"] not in states:
            continue
        private_rank = private_ranks[states.index(row["state"])]
        if private_rank == 0 and abs(row["RIF"]) > 1e-12:
            raise RuntimeError(
                f"{row['method']} in {row['state']}: zero private rank has nonzero RIF"
            )

    matrices = [correlation_matrix(data), correlation_matrix(after_global),
                correlation_matrix(after_local)]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.8), constrained_layout=True)
    image = None
    for axis, matrix, title in zip(
        axes,
        matrices,
        ("Original", "After removing common factors",
         "After removing common and state-specific factors"),
    ):
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        axis.set_xticks(range(len(states)), labels=states, rotation=90, fontsize=9)
        axis.set_yticks(range(len(states)), labels=states, fontsize=9)
        axis.set_xticks(np.arange(-0.5, len(states), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(states), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.5)
        axis.tick_params(which="minor", bottom=False, left=False)
        axis.set_title(title, fontsize=12)
    figure.colorbar(image, ax=axes, shrink=0.82, label="Average absolute correlation")
    figure.savefig(OUTPUT_DIR / "pathfe-residual-correlations.png", dpi=240)
    plt.close(figure)

    with (OUTPUT_DIR / "pathfe-ranks.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("state", "total_rank", "common_rank", "private_rank"))
        writer.writerows((state, rank, pathfe_estimate.num_cf, private_rank)
                         for state, rank, private_rank in zip(states, ranks, private_ranks))
    with (OUTPUT_DIR / "pathfe-metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("rank_source", "method", "state", "RIG", "RIF", "RIE"))
        writer.writeheader()
        writer.writerows(fe_rows)
    with (OUTPUT_DIR / "housing-method-metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("rank_source", "method", "state", "RIG", "RIF", "RIE"))
        writer.writeheader()
        writer.writerows(all_metric_rows)
    with (OUTPUT_DIR / "estimated-ranks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("rank_source", "state", "total_rank", "common_rank", "private_rank"))
        writer.writeheader()
        writer.writerows(estimated_rank_rows)
    result_files = {
        "fixed-rank3-common2": OUTPUT_DIR / "kt-ppca-fixed-rank3-common2.csv",
        "pathfe-estimated": OUTPUT_DIR / "kt-ppca-pathfe-estimated-ranks.csv",
    }
    for rank_source, output_path in result_files.items():
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("rank_source", "method", "state", "RIG", "RIF", "RIE"))
            writer.writeheader()
            writer.writerows(output_rows[rank_source])
        print(f"saved {output_path}")
    print(f"PathFE estimated common rank: {pathfe_estimate.num_cf}")
    print(f"saved {OUTPUT_DIR / 'estimated-ranks.csv'}")
    print(f"saved {OUTPUT_DIR / 'pathfe-metrics.csv'}")
    print(f"saved {OUTPUT_DIR / 'housing-method-metrics.csv'}")
    print(f"saved {OUTPUT_DIR / 'pathfe-residual-correlations.png'}")


if __name__ == "__main__":
    main()
