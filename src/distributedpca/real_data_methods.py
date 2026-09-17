"""Adapters for applying the Simulation-3 common-subspace methods to real data.

All fitting occurs on the supplied training matrices.  The returned estimates
retain preprocessing means and can therefore reconstruct held-out matrices
without fitting on them.  PerPCA is deliberately not included: it remains an
external dependency rather than an implementation maintained in this project.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from Circular_Projection import CP
from Generalized_C_Corr import GCC
from apm import APM
from feature_extraction import AutoFE, FE
from feature_extraction_path import PathFE
from feature_extraction_consensus import ConsensusFE
from knowledge_transfer import KnowledgeTransfer
from pca_utils import PCAUtils


METHOD_IDS = ("fe", "pathfe", "consensusfe", "apm", "cp", "gcc", "tk", "kt")


def _validate_clients(X_list: Sequence[np.ndarray]) -> list[np.ndarray]:
    if not X_list:
        raise ValueError("X_list must contain at least one client")
    matrices = [np.asarray(X, dtype=float) for X in X_list]
    if matrices[0].ndim != 2 or matrices[0].shape[0] == 0:
        raise ValueError("each client matrix must be nonempty and two-dimensional")
    p = matrices[0].shape[1]
    for X in matrices:
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != p:
            raise ValueError("clients must be nonempty matrices with a common feature dimension")
    return matrices


def _basis(factors: np.ndarray, p: int) -> np.ndarray:
    factors = np.asarray(factors, dtype=float)
    if factors.size == 0:
        return np.zeros((p, 0))
    factors = np.atleast_2d(factors)
    if factors.shape[0] != p:
        factors = factors.T
    return np.linalg.svd(factors, full_matrices=False)[0]


def _projector(basis: np.ndarray, p: int) -> np.ndarray:
    basis = _basis(basis, p)
    return basis @ basis.T


@dataclass
class MethodEstimate:
    """Fitted global/local subspaces for one of the Simulation-3 methods."""

    method_id: str
    global_basis: np.ndarray
    local_bases: list[np.ndarray] | None
    num_common: int
    means: list[np.ndarray]
    centered: bool
    num_spike_list: list[int]

    def reconstruct(self, X_list: Sequence[np.ndarray], *, include_local: bool = True) -> list[np.ndarray]:
        matrices = _validate_clients(X_list)
        if len(matrices) != len(self.means):
            raise ValueError("evaluation data must contain the fitted clients in the same order")
        p = self.global_basis.shape[0]
        global_projector = _projector(self.global_basis, p)
        reconstructions = []
        for i, (X, mean) in enumerate(zip(matrices, self.means)):
            centered_X = X - mean if self.centered else X
            projector = global_projector.copy()
            if include_local and self.local_bases is not None:
                projector += _projector(self.local_bases[i], p)
            reconstructions.append(centered_X @ projector + (mean if self.centered else 0.0))
        return reconstructions

    def reconstruction_error(self, X_list: Sequence[np.ndarray], *, include_local: bool = True) -> float:
        matrices = _validate_clients(X_list)
        reconstructions = self.reconstruct(matrices, include_local=include_local)
        squared_error = sum(float(np.linalg.norm(X - X_hat, ord="fro") ** 2)
                            for X, X_hat in zip(matrices, reconstructions))
        return squared_error / sum(X.shape[0] for X in matrices)


def _estimate_spike_counts(X_list: list[np.ndarray], spike_estimator: str, epsilon0: float, a: float,
                           pair_n_method: str) -> list[int]:
    auto = AutoFE(
        spike_estimator=spike_estimator,
        epsilon0=epsilon0,
        a=a,
        pair_n_method=pair_n_method,
    )
    auto.fit(X_list)
    return auto.num_list


def fit_simulation_methods(
    X_train: Sequence[np.ndarray],
    *,
    num_spike_list: Sequence[int] | None = None,
    shared_rank: int | None = None,
    target_index: int = 0,
    target_rank: int | None = None,
    methods: Sequence[str] = METHOD_IDS,
    center: bool = False,
    spike_estimator: str = "zheng",
    epsilon: float | None = None,
    epsilon0: float = 0.1,
    a: float = 10.0,
    pair_n_method: str = "min",
) -> dict[str, MethodEstimate]:
    """Fit the non-PerPCA methods used by Simulation-3 on client training data.

    ``APM``, ``CP``, and ``GCC`` infer their common rank internally.  ``Tk``
    and ``KnowledgeTransfer`` need ``shared_rank`` because their Simulation-3
    versions are rank-oracle methods.  When ``num_spike_list`` is omitted, all
    methods share the selected automatic spike estimator rather than using an
    oracle rank.
    """
    selected_methods = tuple(methods)
    unknown = set(selected_methods) - set(METHOD_IDS)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    matrices = _validate_clients(X_train)
    p = matrices[0].shape[1]
    if a <= 0:
        raise ValueError("a must be positive")
    if pair_n_method not in ("min", "harmonic"):
        raise ValueError("pair_n_method must be 'min' or 'harmonic'")

    means = [X.mean(axis=0) if center else np.zeros(p) for X in matrices]
    fitted_data = [X - mean for X, mean in zip(matrices, means)]
    if num_spike_list is None:
        ranks = _estimate_spike_counts(fitted_data, spike_estimator, epsilon0, a, pair_n_method)
    else:
        if len(num_spike_list) != len(fitted_data):
            raise ValueError("num_spike_list must have one rank per client")
        ranks = [int(rank) for rank in num_spike_list]
    ranks = [max(0, min(rank, X.shape[0], p)) for rank, X in zip(ranks, fitted_data)]

    pc_based_methods = {"apm", "cp", "gcc", "tk", "kt"} & set(selected_methods)
    if pc_based_methods and any(rank == 0 for rank in ranks):
        raise ValueError(
            "APM, CP, GCC, Tk, and KnowledgeTransfer require at least one "
            "selected component for every client; supply num_spike_list or "
            "choose a spike estimator that does so"
        )

    epsilon_supplied = epsilon is not None
    if not epsilon_supplied:
        n_min = min(X.shape[0] for X in fitted_data)
        epsilon = float(n_min ** (1 / 2 - 2 / a - epsilon0))

    estimates: dict[str, MethodEstimate] = {}
    if "fe" in selected_methods:
        fe_kwargs = {"epsilon": epsilon if epsilon_supplied else 0.0, "num_list": ranks}
        # An explicitly supplied epsilon is a direct override. Otherwise use
        # the epsilon0/a pairwise rule (with unequal-client sample sizes).
        if not epsilon_supplied:
            fe_kwargs.update(epsilon0=epsilon0, a=a, pair_n_method=pair_n_method)
        fe = FE().fit(fitted_data, **fe_kwargs)
        estimates["fe"] = MethodEstimate("fe", _basis(fe.global_factors, p),
                                          [_basis(item, p) for item in fe.est_pf], fe.num_cf,
                                          means, center, ranks)
    if "pathfe" in selected_methods:
        pathfe_kwargs = {"epsilon": epsilon if epsilon_supplied else 0.0, "num_list": ranks}
        if not epsilon_supplied:
            pathfe_kwargs.update(epsilon0=epsilon0, a=a, pair_n_method=pair_n_method)
        pathfe = PathFE().fit(fitted_data, **pathfe_kwargs)
        estimates["pathfe"] = MethodEstimate("pathfe", _basis(pathfe.global_factors, p),
                                              [_basis(item, p) for item in pathfe.est_pf], pathfe.num_cf,
                                              means, center, ranks)
    if "consensusfe" in selected_methods:
        consensus_kwargs = {"epsilon": epsilon if epsilon_supplied else 0.0, "num_list": ranks}
        if not epsilon_supplied:
            consensus_kwargs.update(epsilon0=epsilon0, a=a, pair_n_method=pair_n_method)
        consensus = ConsensusFE().fit(fitted_data, **consensus_kwargs)
        estimates["consensusfe"] = MethodEstimate("consensusfe", _basis(consensus.global_factors, p),
                                                   [_basis(item, p) for item in consensus.est_pf], consensus.num_cf,
                                                   means, center, ranks)
    if "apm" in selected_methods:
        apm = APM(ranks)
        apm.fit([X.copy() for X in fitted_data])
        estimates["apm"] = MethodEstimate("apm", _basis(apm.global_factors, p),
                                           [_basis(item, p) for item in apm.local_factor_list], apm.num_common,
                                           means, center, ranks)
    if "cp" in selected_methods:
        cp = CP(ranks)
        cp.fit([X.copy() for X in fitted_data])
        estimates["cp"] = MethodEstimate("cp", _basis(cp.global_factors, p),
                                          [_basis(item, p) for item in cp.local_factor_list], cp.num_common,
                                          means, center, ranks)
    if "gcc" in selected_methods:
        gcc = GCC(ranks)
        gcc.fit([X.copy() for X in fitted_data])
        estimates["gcc"] = MethodEstimate("gcc", _basis(gcc.global_factors, p),
                                           [_basis(item, p) for item in gcc.local_factor_list], gcc.num_common,
                                           means, center, ranks)

    requires_shared_rank = {"tk", "kt"} & set(selected_methods)
    if requires_shared_rank and shared_rank is None:
        raise ValueError("shared_rank is required for Tk and KnowledgeTransfer")
    if requires_shared_rank and shared_rank is not None and not 0 <= int(shared_rank) <= min(ranks):
        raise ValueError(f"shared_rank must be between 0 and min(num_spike_list)={min(ranks)}")
    if "tk" in selected_methods:
        tk = PCAUtils.tk(fitted_data, int(shared_rank), min(ranks))
        estimates["tk"] = MethodEstimate("tk", _basis(tk, p), None, int(shared_rank), means, center, ranks)
    if "kt" in selected_methods:
        if not 0 <= target_index < len(fitted_data):
            raise ValueError("target_index is out of range")
        if target_rank is None:
            target_rank = int(shared_rank)
        kt = KnowledgeTransfer(ranks, target_index=target_index, shared_rank=int(shared_rank),
                               target_rank=int(target_rank))
        kt.fit([X.copy() for X in fitted_data])
        estimates["kt"] = MethodEstimate(
            "kt", _basis(kt.global_factors, p),
            [_basis(item, p) for item in kt.local_factor_list],
            kt.num_common, means, center, ranks,
        )
    return estimates
