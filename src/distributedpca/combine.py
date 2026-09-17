import numpy as np
from mbe import MBE
from yao_estimate import YaoEstimate

class Combine(MBE, YaoEstimate):
    def __init__(self):
        super().__init__()  # 去掉 device

    # def estimate_v1(self, X, threshold, n_repeat=50, ip=True, avg=True):
    #     res_mbe, _ = MBE.estimate(self, X, threshold, n_repeat, ip=ip, avg=avg)
    #     res_yao, _, _ = YaoEstimate.estimate(self, X)
    #     res = min(res_mbe, res_yao)
    #     return res

    # def estimate_v2(self, X, threshold, n_repeat=50, ip=True, avg=True):
    #     res_mbe, _ = MBE.estimate(self, X, threshold, n_repeat, ip=ip, avg=avg)
    #     res_yao, _, _ = YaoEstimate.estimate(self, X)
    #     res = max(res_mbe, res_yao)
    #     return res

    def estimate(
        self,
        X,
        threshold,
        yao_threshold=None,
        n_repeat=50,
        ip=True,
        avg=True,
        s=20,
        decision_rule="or",
    ):
        if decision_rule not in {"or", "and"}:
            raise ValueError("decision_rule must be 'or' or 'and'")
        if yao_threshold is None:
            _, yao_threshold, delta_eig = YaoEstimate.estimate(self, X)
        else:
            n = X.shape[0]
            cov = (X.T @ X) / n
            eigvals = np.linalg.eigvalsh(cov)[::-1]
            delta_eig = eigvals[:-1] - eigvals[1:]
        _, diag_mean = MBE.estimate(self, X, threshold, n_repeat, ip=ip, avg=avg)

        delta = np.asarray(delta_eig, dtype=float).ravel()
        inn = np.asarray(diag_mean, dtype=float).ravel()
        s = int(s)

        max_s = min(len(delta), len(inn)) - 2
        s_eff = min(s, max_s)
        if s_eff <= 0:
            return s

        for j in range(1, s_eff + 1):
            cond1 = (delta[j] < yao_threshold) and (delta[j + 1] < yao_threshold)
            cond2 = (inn[j] < threshold) and (inn[j + 1] < threshold)
            if (cond1 and cond2) if decision_rule == "and" else (cond1 or cond2):
                return j
        return s       

        # # mask1: delta_eig 阈值判断
        # mask1 = (delta_eig[:-1] < dn) & (delta_eig[1:] < dn)

        # # mask2: diag_mean 阈值判断，注意长度对齐
        # mask2 = np.abs(diag_mean[:-2]) <= threshold

        # # 合并 mask
        # mask = mask1 | mask2
        # indices = np.where(mask)[0]

        # if len(indices) == 0:
        #     j_found = None
        # else:
        #     j_found = indices[0]

        # return j_found
    # def estimate_q_or(delta, sigma, eps, s):
    #     """
    #     delta[k] = δ_{k+1}, sigma[k] = σ_{k+1}  (k从0开始)
    #     返回 q = min{j=1..s: (δ_{j+1}<eps and δ_{j+2}<eps) or (σ_{j+1}<eps and σ_{j+2}<eps)}
    #     若找不到返回 s
    #     """
    #     delta = np.asarray(delta, dtype=float).ravel()
    #     sigma = np.asarray(sigma, dtype=float).ravel()
    #     s = int(s)

    #     max_s = min(len(delta), len(sigma)) - 2
    #     s_eff = min(s, max_s)
    #     if s_eff <= 0:
    #         return s

    #     for j in range(1, s_eff + 1):
    #         cond1 = (delta[j] < eps) and (delta[j + 1] < eps)
    #         cond2 = (sigma[j] < eps) and (sigma[j + 1] < eps)
    #         if cond1 or cond2:
    #             return j

    #     return s
