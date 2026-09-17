import numpy as np

from spike_num import SpikeNum


class LiuEstimate(SpikeNum):
    """EDA/EDB spike-count criteria from Liu et al. (2023)."""

    def __init__(self):
        super().__init__()

    def estimate(self, X, criterion="edb", center=True):
        """Estimate the number of spikes with either the EDA or EDB criterion.

        ``criterion`` is ``"edb"`` by default because EDB is the primary
        reported criterion in the paper. Set it to ``"eda"`` to use EDA.
        """
        X = np.asarray(X, dtype=float)
        n, p = X.shape
        if n < 3 or p < 3:
            raise ValueError("X must have at least three rows and columns")
        if criterion not in {"eda", "edb"}:
            raise ValueError("criterion must be 'eda' or 'edb'")

        if center:
            X = X - X.mean(axis=0, keepdims=True)

        # The paper works with the nonzero sample eigenvalues. This also
        # implements its pseudo-EDA/pseudo-EDB branch when p > n.
        eigvals = np.linalg.svd(X, compute_uv=False) ** 2 / n
        spectrum_size = len(eigvals)
        if spectrum_size < 3:
            return 0

        w = min(int(np.floor(6 * n**0.1)), spectrum_size - 2)
        if w < 1:
            return 0

        gaps = eigvals[:-1] - eigvals[1:]
        theta = np.exp(np.clip(gaps, -700, 700))
        tail_sum = np.cumsum(theta[::-1])[::-1]
        ks = np.arange(1, w + 1)
        tail_counts = spectrum_size - ks - 1
        tilde_theta = tail_sum[ks] / tail_counts

        leading_gaps = eigvals[0] - eigvals[ks]
        bulk_size = p if p <= n else n
        eda = -n * leading_gaps + n * (bulk_size - ks - 1) * np.log(tilde_theta) + 2 * p * ks
        edb = (
            -n * np.log(p) * leading_gaps
            + n * (bulk_size - ks - 1) * np.log(tilde_theta)
            + p * ks * np.log(n)
        )

        scores = eda if criterion == "eda" else edb
        return int(np.argmin(scores)) + 1
