"""Oracle Grassmannian-barycenter knowledge transfer for PCA."""

import numpy as np


class KnowledgeTransfer:
    """Algorithm 1 of Zhou et al. (2024), with target-only fine tuning."""

    def __init__(self, num_spike_list, target_index=0, shared_rank=2, target_rank=2):
        self.num_spike_list = [int(r) for r in num_spike_list]
        self.target_index = int(target_index)
        self.shared_rank = int(shared_rank)
        self.target_rank = int(target_rank)

    def fit(self, data_list, source_indices=None):
        if not data_list or len(data_list) != len(self.num_spike_list):
            raise ValueError("data_list and num_spike_list must have the same nonzero length")
        if not 0 <= self.target_index < len(data_list):
            raise ValueError("target_index is out of range")
        if source_indices is None:
            source_indices = [i for i in range(len(data_list)) if i != self.target_index]
        participants = [self.target_index] + [int(i) for i in source_indices]
        if len(set(participants)) != len(participants):
            raise ValueError("source_indices contains duplicates or the target")

        p = int(data_list[0].shape[1])
        projectors, weights = [], []
        for i in participants:
            X = np.asarray(data_list[i], dtype=float)
            if X.ndim != 2 or X.shape[1] != p:
                raise ValueError("all datasets must have the same feature dimension")
            rank = min(self.num_spike_list[i], p, X.shape[0])
            _, _, vt = np.linalg.svd(X, full_matrices=False)
            U = vt[:rank].T
            projectors.append(U @ U.T)
            weights.append(X.shape[0])

        barycenter = sum(w * P for w, P in zip(weights, projectors)) / sum(weights)
        vals, vecs = np.linalg.eigh(barycenter)
        shared_rank = min(self.shared_rank, p, len(vals))
        shared = vecs[:, np.argsort(vals)[::-1][:shared_rank]]

        private_rank = max(self.target_rank - shared_rank, 0)
        if private_rank:
            X = np.asarray(data_list[self.target_index], dtype=float)
            complement = np.eye(p) - shared @ shared.T
            cov = complement @ (X.T @ X / max(X.shape[0], 1)) @ complement
            vals, vecs = np.linalg.eigh(cov)
            private = vecs[:, np.argsort(vals)[::-1][:private_rank]]
        else:
            private = np.zeros((p, 0))

        self.num_common = shared_rank
        self.global_factors = np.sqrt(p) * shared
        self.private_factors = np.sqrt(p) * private
        self.target_projector = shared @ shared.T
        self.private_projector = private @ private.T
        # Complete personalized output: estimate each client's local PCs in
        # the orthogonal complement of the transferred shared subspace.
        self.local_factor_list = []
        self.local_projector_list = []
        complement = np.eye(p) - shared @ shared.T
        for i, data in enumerate(data_list):
            X = np.asarray(data, dtype=float)
            local_rank = max(min(self.num_spike_list[i], p) - shared_rank, 0)
            if local_rank == 0:
                local = np.zeros((p, 0))
            else:
                cov = complement @ (X.T @ X / max(X.shape[0], 1)) @ complement
                vals, vecs = np.linalg.eigh(cov)
                local = vecs[:, np.argsort(vals)[::-1][:local_rank]]
            self.local_factor_list.append(np.sqrt(p) * local)
            self.local_projector_list.append(local @ local.T)
        return self
