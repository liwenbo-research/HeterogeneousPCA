import numpy as np
from spike_num import SpikeNum

class ZhengEstimate(SpikeNum):

    def __init__(self):
        super().__init__()

    def compute_eigenvalues(self, X):
        X = np.asarray(X, dtype=float)
        X = X - X.mean(axis=0, keepdims=True)
        n_k = X.shape[0]
        cov = X.T @ X / n_k
        eps = 1e-8
        D_inv_sqrt = np.diag((np.diag(cov) + eps) ** (-0.5))

        # 构造相关矩阵
        standardized = X @ D_inv_sqrt
        if n_k < X.shape[1]:
            gram = (standardized @ standardized.T) / n_k
            values = np.linalg.eigvalsh((gram + gram.T) * 0.5)[::-1]
            return np.pad(np.maximum(values, 0.0), (0, X.shape[1] - n_k))
        cor = (standardized.T @ standardized) / n_k
        return np.linalg.svd(cor, compute_uv=False)
    

    def correct_eigenvalues(self,S,rmax,p,n):
        rmax = min(rmax,p-1)
        S = np.asarray(S, dtype=float)
        S_c = np.zeros(rmax, dtype=float)
        eps = 1e-12
        for i in range(rmax):
            if n <= 1 or not np.isfinite(S[i]) or abs(S[i]) < eps:
                S_c[i] = S[i]
                continue
            rho = (p-(i+1))/(n-1)
            denom = (S[i+1:] - S[i])
            denom0 = (S[i+1] - S[i])
            if not np.isfinite(denom0) or abs(denom0) < eps or np.any(np.abs(denom) < eps):
                S_c[i] = S[i]
                continue
            m = (p - (i + 1)) ** (-1) * (np.sum(1.0 / denom) + 4.0 / denom0)
            m_bar = -(1 - rho) / S[i] + rho * m
            if not np.isfinite(m_bar) or abs(m_bar) < eps:
                S_c[i] = S[i]
                continue
            S_c[i] = -1.0 / m_bar
        return S_c

    
    def estimate(self, X):
        n, p = X.shape
        if n <= 1:
            return 0
        evs = self.compute_eigenvalues(X)
        evs_c = self.correct_eigenvalues(evs,p-1,p,n)
        mask = evs_c > 1 + (p / (n - 1)) ** 0.5
        indices = np.nonzero(mask)[0]
        if indices.size == 0:
            return 0  # 没有 spike
        else:
            return int(indices.max()) + 1 # 返回 spike 数量
