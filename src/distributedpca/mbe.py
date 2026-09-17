import numpy as np
from spike_num import SpikeNum
from pca_utils import PCAUtils

class MBE(SpikeNum):

    def __init__(self):
        super().__init__()  

    # def argmin_within_threshold(self, x: np.ndarray, threshold: float):
    #     """返回 abs(x) <= threshold 的第一个元素索引"""
    #     indices = np.where(np.abs(x) <= threshold)[0]
    #     if len(indices) == 0:
    #         return 1
    #     return indices[0]

    def estimate(self, X: np.ndarray, threshold: float, n_repeat: int = 50, ip: bool = True, avg: bool = True, s: int = 20):
        """
        估计 spike 数量的完整流程：
        """
        n, p = X.shape

        if avg:
            if ip:
                diag_accum = np.zeros(p)
                for _ in range(n_repeat):
                    X1, X2 = PCAUtils.sample_and_remainder(X, n // 2)

                    eigvecs1 = PCAUtils.compute_eigenvectors(X1)
                    eigvecs2 = PCAUtils.compute_eigenvectors(X2)

                    inner_prod = PCAUtils.all_column_inner_products(eigvecs1, eigvecs2)
                    diag_accum += np.abs(np.diag(inner_prod))

                diag_mean = diag_accum / n_repeat

            else:
                eigvecs1_accum = np.zeros((p, p))
                eigvecs2_accum = np.zeros((p, p))

                for _ in range(n_repeat):
                    X1, X2 = PCAUtils.sample_and_remainder(X, n // 2)

                    eigvecs1 = PCAUtils.compute_eigenvectors(X1)
                    eigvecs2 = PCAUtils.compute_eigenvectors(X2)

                    eigvecs1_accum += eigvecs1
                    eigvecs2_accum += eigvecs2

                eigvecs1_mean = eigvecs1_accum / n_repeat
                eigvecs2_mean = eigvecs2_accum / n_repeat

                inner_prod = PCAUtils.all_column_inner_products(eigvecs1_mean, eigvecs2_mean)
                diag_mean = np.diag(inner_prod)
        else:
            X1, X2 = PCAUtils.sample_and_remainder(X, n // 2)

            eigvecs1 = PCAUtils.compute_eigenvectors(X1)
            eigvecs2 = PCAUtils.compute_eigenvectors(X2)

            inner_prod = PCAUtils.all_column_inner_products(eigvecs1, eigvecs2)
            diag_mean = np.abs(np.diag(inner_prod))

        inn = np.asarray(diag_mean, dtype=float).ravel()
        s = int(s)

        max_s = len(inn) - 2
        s_eff = min(s, max_s)
        if s_eff <= 0:
            return s,diag_mean

        for j in range(1, s_eff + 1):
            cond = (inn[j] < threshold) and (inn[j + 1] < threshold)
            if cond:
                return j,diag_mean
        return s,diag_mean
       
