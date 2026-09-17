"""All-client consensus variant of FE.

Pairwise FE statistics and thresholds are unchanged.  ConsensusFE replaces the
ordered adjacent chain with an all-client compatibility graph and only accepts
directions having one mutually compatible eigenvector in every client.
"""

import numpy as np

from feature_extraction import FE
from pca_utils import PCAUtils


class ConsensusFE(FE):
    def fit(self, X_list, epsilon, num_list, correctCommonFeature=None, epsilon0=None, a=None,
            pair_n_method="min"):
        if not X_list:
            return super().fit([], epsilon=epsilon, num_list=num_list)
        p = int(X_list[0].shape[1])
        if (epsilon0 is None) != (a is None):
            raise ValueError("epsilon0 and a must be provided together")
        if epsilon0 is not None:
            eps = [PCAUtils.pair_effective_n(X_list[i].shape[0], X_list[j].shape[0], pair_n_method)
                   ** (1 / 2 - 2 / a - epsilon0)
                   for i in range(len(X_list)) for j in range(i + 1, len(X_list))]
        elif np.ndim(epsilon) == 1 and not np.isscalar(epsilon):
            raise ValueError("ConsensusFE requires a scalar epsilon when epsilon0/a are omitted")
        else:
            eps = [epsilon] * (len(X_list) * (len(X_list) - 1) // 2)

        spectra = []
        for X in X_list:
            values, vectors = np.linalg.eigh(X.T @ X / max(X.shape[0], 1))
            order = np.argsort(values)[::-1]
            spectra.append((values[order], vectors[:, order], X.shape[0]))
        n_clients = len(X_list)
        edges = {}
        eidx = 0
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                vi, ui, ni = spectra[i]
                vj, uj, nj = spectra[j]
                pairs, ratio, _ = PCAUtils.identify_prod_from_spectra(
                    vi, ui, ni, vj, uj, nj, num_list[i], num_list[j], eps[eidx]
                )
                threshold = 1 - 1 / eps[eidx]
                edges[i, j] = {(int(x), int(y)): float(ratio[x, y] - threshold) for x, y in pairs}
                edges[j, i] = {(int(y), int(x)): float(ratio[x, y] - threshold) for x, y in pairs}
                eidx += 1

        blocked = [set() for _ in range(n_clients)]
        locations, scores = [], []
        # Anchor candidates are tried in descending average compatibility.
        anchors = range(min(int(num_list[0]), spectra[0][1].shape[1]))
        while True:
            best = None
            for anchor in anchors:
                if anchor in blocked[0]:
                    continue
                assignment = [anchor]
                total = 0.0
                valid = True
                for client in range(1, n_clients):
                    candidates = []
                    for node in range(min(int(num_list[client]), spectra[client][1].shape[1])):
                        if node in blocked[client]:
                            continue
                        vals = [edges[min(prev, client), max(prev, client)].get(
                            (assignment[prev], node) if prev < client else (node, assignment[prev]), -np.inf)
                                for prev in range(client)]
                        if all(np.isfinite(vals)):
                            candidates.append((sum(vals), node))
                    if not candidates:
                        valid = False
                        break
                    score, node = max(candidates)
                    total += score
                    assignment.append(node)
                if valid and (best is None or total > best[0]):
                    best = (total, assignment)
            if best is None:
                break
            total, assignment = best
            locations.append(assignment)
            scores.append(total)
            for client, node in enumerate(assignment):
                blocked[client].add(node)

        self.location = locations
        self.path_scores = scores
        self.num_cf = len(locations)
        self.pairs_correct_ratio = 0.0 if correctCommonFeature is None else PCAUtils.column_match_ratio(locations, correctCommonFeature)
        self.est_cf, self.global_factors, self.est_cf_Proj = self.estCommonFeature(X_list, locations)
        complement = np.eye(p) - (self.est_cf_Proj if self.est_cf_Proj.size else np.zeros((p, p)))
        self.est_pf, pf_proj = [], []
        for X, rank in zip(X_list, num_list):
            vals, vecs = np.linalg.eigh(complement @ (X.T @ X / max(X.shape[0], 1)) @ complement)
            local_rank = max(int(rank) - self.num_cf, 0)
            basis = vecs[:, np.argsort(vals)[::-1][:local_rank]] if local_rank else np.zeros((p, 0))
            self.est_pf.append(basis)
            pf_proj.append(basis @ basis.T)
        self.est_pf_proj = np.stack(pf_proj, axis=0)
        return self
