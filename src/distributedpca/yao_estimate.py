import numpy as np
from spike_num import SpikeNum
import math

class YaoEstimate(SpikeNum):

    def __init__(self):
        super().__init__()  # 保留接口，但不使用 device

    def computeC(self, p: int, n: int, sigma: float, n_repeat: int = 500):
        """
        计算 Yao C 统计量
        """
        deltas = []

        for _ in range(n_repeat):
            X = np.random.randn(p, n) * sigma
            W = (X @ X.T) / n
            eigvals = np.linalg.eigvalsh(W)
            delta = eigvals[-1] - eigvals[-2]
            deltas.append(delta)

        deltas = np.array(deltas)
        deltas_sorted = np.sort(deltas)[::-1]  # 降序

        if len(deltas_sorted) >= 11:
            mean_delta = (deltas_sorted[9] + deltas_sorted[10]) / 2
        else:
            raise ValueError("样本数太小，无法取第10和第11大值")

        C = mean_delta * (n ** (2/3)) / math.sqrt(2 * math.log(math.log(n)))

        return C

    def estimate(self, X, sigma: float = 1.0, center: bool = False):
        """
        基于 Yao 方法估计 spike 个数
        """
        n, p = X.shape

        # Step1: 中心化
        if center:
            X_centered = X - X.mean(axis=0, keepdims=True)
        else:
            X_centered = X

        # Step2: 计算 C
        C = self.computeC(p=p, n=n, sigma=sigma)

        # Step3: 协方差特征值
        cov = (X_centered.T @ X_centered) / n
        eigvals = np.linalg.eigvalsh(cov)[::-1]  # 降序

        # Step4: 阈值
        dn = C * (n ** (-2/3)) * math.sqrt(2 * math.log(math.log(n)))

        # Step5: 相邻差
        delta_eig = eigvals[:-1] - eigvals[1:]

        # Step6: 找 j
        j_found = None
        for j in range(len(delta_eig) - 2):
            if delta_eig[j+1] < dn and delta_eig[j+2] < dn:
                j_found = j
                break

        # Step7
        if j_found is not None:
            q = j_found + 1
        else:
            q = len(delta_eig)

        return q, dn, delta_eig