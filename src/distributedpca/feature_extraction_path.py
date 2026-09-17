"""Graph-based path extraction variant of Feature Extraction.

``PathFE`` keeps FE's pairwise statistic and threshold unchanged.  It differs
only in how accepted pairwise matches are assembled: a layered graph is used
to find high-scoring paths that visit every client, rather than relying on the
row order of a padded match matrix.
"""

import numpy as np

from feature_extraction import FE
from pca_utils import PCAUtils


def extract_complete_paths(pairs_list, ratio_list, thresholds):
    """Return vertex-disjoint, full-client paths ranked by total edge ratio.

    All supplied edges have already passed FE's original pairwise threshold.
    A returned path therefore has exactly one accepted edge between every two
    consecutive clients.  Repeatedly taking the best complete path while
    removing its client-layer vertices prevents the same client PC from being
    reported as multiple common directions.
    """
    if not pairs_list:
        return [], []
    n_layers = len(pairs_list) + 1
    edges_by_layer = []
    for pairs, ratios, threshold in zip(pairs_list, ratio_list, thresholds):
        edges = []
        for source, target in np.asarray(pairs, dtype=int):
            score = float(ratios[source, target] - threshold)
            edges.append((int(source), int(target), score))
        edges_by_layer.append(edges)

    blocked = [set() for _ in range(n_layers)]
    paths, scores = [], []
    while True:
        states = {}
        for source, _, _ in edges_by_layer[0]:
            if source not in blocked[0]:
                states.setdefault(source, (0.0, [source]))
        for layer, edges in enumerate(edges_by_layer):
            next_states = {}
            for source, target, score in edges:
                if source in blocked[layer] or target in blocked[layer + 1] or source not in states:
                    continue
                total, path = states[source]
                candidate = (total + score, path + [target])
                previous = next_states.get(target)
                if previous is None or candidate[0] > previous[0]:
                    next_states[target] = candidate
            states = next_states
            if not states:
                break
        if not states:
            break
        score, path = max(states.values(), key=lambda item: item[0])
        if len(path) != n_layers:
            break
        paths.append(path)
        scores.append(score)
        for layer, node in enumerate(path):
            blocked[layer].add(node)
    return paths, scores


class PathFE(FE):
    """FE with graph-based complete-path selection; pairwise FE is unchanged."""

    def fit(self, X_list, epsilon, num_list, correctCommonFeature=None, epsilon0=None, a=None,
            pair_n_method="min"):
        if not X_list:
            return super().fit([], epsilon=epsilon, num_list=num_list)
        p = int(X_list[0].shape[1])
        if any(int(X.shape[1]) != p for X in X_list):
            raise ValueError("X_list must have a common feature dimension")
        if (epsilon0 is None) != (a is None):
            raise ValueError("epsilon0 and a must be provided together")
        if a is not None and a <= 0:
            raise ValueError("a must be positive")

        if epsilon0 is not None:
            pair_epsilons = [
                PCAUtils.pair_effective_n(X_list[i].shape[0], X_list[i + 1].shape[0], pair_n_method)
                ** (1 / 2 - 2 / a - epsilon0)
                for i in range(len(X_list) - 1)
            ]
        elif np.ndim(epsilon) == 1 and not np.isscalar(epsilon):
            pair_epsilons = list(epsilon)
        else:
            pair_epsilons = [epsilon] * (len(X_list) - 1)
        if len(pair_epsilons) != len(X_list) - 1:
            raise ValueError("epsilon sequence must have one value per adjacent client pair")

        pairs_list, ratio_list, thresholds = [], [], []
        spectra = []
        for X in X_list:
            covariance = X.T @ X / max(X.shape[0], 1)
            values, vectors = np.linalg.eigh(covariance)
            order = np.argsort(values)[::-1]
            spectra.append((values[order], vectors[:, order], X.shape[0]))
        for i, pair_epsilon in enumerate(pair_epsilons):
            values_X, vectors_X, n_X = spectra[i]
            values_Y, vectors_Y, n_Y = spectra[i + 1]
            pairs, ratio, _ = PCAUtils.identify_prod_from_spectra(
                values_X, vectors_X, n_X, values_Y, vectors_Y, n_Y,
                num_list[i], num_list[i + 1], pair_epsilon
            )
            pairs_list.append(pairs)
            ratio_list.append(ratio)
            thresholds.append(1 - 1 / pair_epsilon)
        location, self.path_scores = extract_complete_paths(pairs_list, ratio_list, thresholds)

        self.location = location
        self.num_cf = len(location)
        self.pairs_correct_ratio = (
            0.0 if correctCommonFeature is None else PCAUtils.column_match_ratio(location, correctCommonFeature)
        )
        # Re-estimate the shared space from a sample-size-weighted projector
        # barycenter. This reduces the influence of noisy small clients while
        # preserving PathFE's path selection itself.
        self.est_cf, self.global_factors, self.est_cf_Proj = self._weighted_common_feature(X_list, location)
        complement = np.eye(p) - (self.est_cf_Proj if self.est_cf_Proj.size else np.zeros((p, p)))
        pf_list, pf_proj_list = [], []
        for X, rank in zip(X_list, num_list):
            residual = complement @ (X.T @ X / max(X.shape[0], 1)) @ complement
            local_rank = max(int(rank) - self.num_cf, 0)
            if local_rank:
                vals, vecs = np.linalg.eigh(residual)
                pf = vecs[:, np.argsort(vals)[::-1][:local_rank]]
                pf_proj = pf @ pf.T
            else:
                pf = np.zeros((p, 0))
                pf_proj = np.zeros((p, p))
            pf_list.append(pf)
            pf_proj_list.append(pf_proj)
        self.est_pf = pf_list
        self.est_pf_proj = np.stack(pf_proj_list, axis=0)
        return self

    def _weighted_common_feature(self, data_list, location):
        num_cf = len(location)
        p = int(data_list[0].shape[1])
        if num_cf == 0:
            return np.zeros((p, 0)), np.zeros((p, 0)), np.zeros((p, p))
        weights = np.asarray([x.shape[0] for x in data_list], dtype=float)
        avg = np.zeros((p, p), dtype=float)
        for i, X in enumerate(data_list):
            eigvecs = PCAUtils.compute_eigenvectors(X)
            cf = eigvecs[:, np.asarray(location)[:, i]]
            avg += weights[i] * (cf @ cf.T)
        avg /= max(weights.sum(), 1.0)
        vals, vecs = np.linalg.eigh(avg)
        order = np.argsort(vals)[::-1][:num_cf]
        est_cf = vecs[:, order]
        return est_cf, np.sqrt(p) * est_cf, est_cf @ est_cf.T
