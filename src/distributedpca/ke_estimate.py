import math
from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from TracyWidom import TracyWidom

from spike_num import SpikeNum


class KeEstimate(SpikeNum):
    """BEMA0 from Ke, Ma, and Lin (2021).

    This implements Algorithm 1 from the paper for the standard spiked
    covariance model with scalar residual variance.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def mpDensity(ndf, pdim, var=1.0):
        gamma = pdim / ndf
        root_gamma = math.sqrt(gamma)
        return var * (1 - root_gamma) ** 2, var * (1 + root_gamma) ** 2

    @staticmethod
    def dmp(x, ndf, pdim, var=1.0, log=False):
        gamma = pdim / ndf
        lower, upper = KeEstimate.mpDensity(ndf, pdim, var)
        if x <= lower or x >= upper:
            return -np.inf if log else 0.0

        density = math.sqrt((x - lower) * (upper - x)) / (2 * math.pi * min(gamma, 1.0) * var * x)
        return math.log(density) if log else density

    @staticmethod
    def pmp(q, ndf, pdim, var=1.0, lower_tail=True, log_p=False):
        lower, upper = KeEstimate.mpDensity(ndf, pdim, var)
        if q <= lower:
            probability = 0.0
        elif q >= upper:
            probability = 1.0
        else:
            probability = quad(
                lambda x: KeEstimate.dmp(x, ndf, pdim, var),
                lower,
                q,
                epsabs=1e-10,
                epsrel=1e-10,
            )[0]
        probability = min(max(probability, 0.0), 1.0)
        if not lower_tail:
            probability = 1.0 - probability
        return math.log(probability) if log_p else probability

    @staticmethod
    def qmp(probability, ndf, pdim, var=1.0, lower_tail=True, log_p=False):
        if log_p:
            probability = math.exp(probability)
        if not lower_tail:
            probability = 1.0 - probability

        lower, upper = KeEstimate.mpDensity(ndf, pdim, var)
        if probability <= 0.0:
            return lower
        if probability >= 1.0:
            return upper
        return brentq(
            lambda x: KeEstimate.pmp(x, ndf, pdim, var) - probability,
            lower,
            upper,
            xtol=1e-12,
            rtol=1e-12,
        )

    @staticmethod
    @lru_cache(maxsize=64)
    def _bulk_quantiles(ndf, pdim, alpha):
        p_tilde = min(ndf, pdim)
        start = max(1, math.ceil(alpha * p_tilde))
        end = min(p_tilde, math.floor((1 - alpha) * p_tilde))
        ranks = np.arange(start, end + 1)
        if len(ranks) == 0:
            raise ValueError("alpha leaves no bulk eigenvalues for BEMA0")

        # q_k is the (k / p_tilde)-upper MP quantile in Algorithm 1.
        quantiles = np.array(
            [KeEstimate.qmp(1.0 - rank / p_tilde, ndf, pdim) for rank in ranks]
        )
        return ranks, quantiles

    def estimate_standard(self, X, alpha=0.2, beta=0.1, center=True):
        X = np.asarray(X, dtype=float)
        ndf, pdim = X.shape
        if ndf < 2 or pdim < 2:
            raise ValueError("X must have at least two rows and columns")
        if not 0.0 < alpha < 0.5:
            raise ValueError("alpha must be in (0, 0.5)")
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must be in (0, 1)")

        if center:
            X = X - X.mean(axis=0, keepdims=True)
        eigvals = np.linalg.svd(X, compute_uv=False) ** 2 / ndf

        ranks, quantiles = self._bulk_quantiles(ndf, pdim, float(alpha))
        sigma_sq = float(quantiles @ eigvals[ranks - 1]) / float(quantiles @ quantiles)

        gamma = pdim / ndf
        tw_quantile = TracyWidom(beta=1).cdfinv(1.0 - beta)
        threshold = sigma_sq * (
            (1.0 + math.sqrt(gamma)) ** 2
            + tw_quantile
            * ndf ** (-2.0 / 3.0)
            * gamma ** (-1.0 / 6.0)
            * (1.0 + math.sqrt(gamma)) ** (4.0 / 3.0)
        )
        return int(np.sum(eigvals > threshold))

    @staticmethod
    @lru_cache(maxsize=128)
    def _general_bulk_quantiles(ndf, pdim, theta, n_samples, seed):
        """Monte Carlo GetQT1 for the Gamma residual model in Algorithm 2."""
        rng = np.random.default_rng(seed)
        p_tilde = min(ndf, pdim)
        quantiles = np.zeros(p_tilde)
        for _ in range(n_samples):
            residual_scale = rng.gamma(shape=theta, scale=1.0 / theta, size=pdim)
            null_data = rng.standard_normal((ndf, pdim)) * np.sqrt(residual_scale)
            null_data -= null_data.mean(axis=0, keepdims=True)
            quantiles += np.linalg.svd(null_data, compute_uv=False) ** 2 / ndf
        return tuple(quantiles / n_samples)

    @staticmethod
    def _general_threshold(ndf, pdim, sigma_sq, theta, beta, n_samples, seed):
        rng = np.random.default_rng(seed)
        largest_eigenvalues = np.empty(n_samples)
        for sample_idx in range(n_samples):
            residual_scale = rng.gamma(shape=theta, scale=sigma_sq / theta, size=pdim)
            null_data = rng.standard_normal((ndf, pdim)) * np.sqrt(residual_scale)
            null_data -= null_data.mean(axis=0, keepdims=True)
            singular_values = np.linalg.svd(null_data, compute_uv=False)
            largest_eigenvalues[sample_idx] = singular_values[0] ** 2 / ndf
        return float(np.quantile(largest_eigenvalues, 1.0 - beta))

    def estimate_general(
        self,
        X,
        alpha=0.2,
        beta=0.1,
        theta_grid=None,
        n_quantile_samples=10,
        n_threshold_samples=100,
        random_state=None,
        center=True,
    ):
        """Implement Algorithm 2 (BEMA) for Gamma-heteroscedastic residuals.

        The paper uses Monte Carlo GetQT1 and 500 samples for the final
        threshold. Smaller counts are configurable for simulation studies.
        """
        X = np.asarray(X, dtype=float)
        ndf, pdim = X.shape
        if ndf < 2 or pdim < 2:
            raise ValueError("X must have at least two rows and columns")
        if not 0.0 < alpha < 0.5:
            raise ValueError("alpha must be in (0, 0.5)")
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must be in (0, 1)")
        if n_quantile_samples < 1 or n_threshold_samples < 1:
            raise ValueError("Monte Carlo sample counts must be positive")

        if center:
            X = X - X.mean(axis=0, keepdims=True)
        eigvals = np.linalg.svd(X, compute_uv=False) ** 2 / ndf
        p_tilde = len(eigvals)
        start = max(1, math.ceil(alpha * p_tilde))
        end = min(p_tilde, math.floor((1 - alpha) * p_tilde))
        ranks = np.arange(start, end + 1)
        if len(ranks) == 0:
            raise ValueError("alpha leaves no bulk eigenvalues for BEMA")

        if theta_grid is None:
            theta_grid = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
        theta_grid = np.asarray(theta_grid, dtype=float)
        if theta_grid.ndim != 1 or len(theta_grid) == 0 or np.any(theta_grid <= 0):
            raise ValueError("theta_grid must be a nonempty vector of positive values")

        # GetQT1 depends only on (n, p, theta), not on the observed sample.
        # A fixed stream makes its cached approximation reproducible.
        quantile_seed = 20260827
        base_seed = 20260827 if random_state is None else int(random_state)
        bulk_eigvals = eigvals[ranks - 1]
        best_loss = np.inf
        best_theta = None
        best_sigma_sq = None
        for theta_idx, theta in enumerate(theta_grid):
            quantiles = np.asarray(
                self._general_bulk_quantiles(
                    ndf,
                    pdim,
                    float(theta),
                    int(n_quantile_samples),
                    quantile_seed + theta_idx,
                )
            )[ranks - 1]
            sigma_sq = float(quantiles @ bulk_eigvals) / float(quantiles @ quantiles)
            loss = float(np.sum((bulk_eigvals - sigma_sq * quantiles) ** 2))
            if loss < best_loss:
                best_loss = loss
                best_theta = float(theta)
                best_sigma_sq = sigma_sq

        threshold = self._general_threshold(
            ndf,
            pdim,
            best_sigma_sq,
            best_theta,
            beta,
            int(n_threshold_samples),
            base_seed + 100003,
        )
        return int(np.sum(eigvals > threshold))

    def estimate(self, X, alpha=0.2, beta=0.1, model="general", **kwargs):
        """Estimate spikes with BEMA (general) or BEMA0 (standard)."""
        if model == "general":
            return self.estimate_general(X, alpha=alpha, beta=beta, **kwargs)
        if model == "standard":
            return self.estimate_standard(X, alpha=alpha, beta=beta, **kwargs)
        raise ValueError("model must be 'general' or 'standard'")
