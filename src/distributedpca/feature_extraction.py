import numpy as np
from pca_utils import PCAUtils
from combine import Combine
from zheng_estimate import ZhengEstimate
from yao_estimate import YaoEstimate


class FE:
    def __init__(self):
        self.location = []
        self.num_cf = 0
        self.pairs_correct_ratio = 0.0
        self.est_cf = np.array([])
        self.global_factors = np.array([])
        self.est_cf_Proj = np.array([])
        self.est_pf = []
        self.est_pf_proj = np.array([]) 

    def estCommonFeature(self, Data_list, location: list):
        num_cf = len(location)

        if len(location) == 0:
            return np.array([]), np.array([]), np.array([])

        location_arr = np.array(location)
        avgProj = 0.0
        for i, X in enumerate(Data_list):
            eigvecs = PCAUtils.compute_eigenvectors(X)
            cf = eigvecs[:, location_arr[:, i]]
            avgProj += cf @ cf.T
        avgProj = avgProj / len(Data_list)
        U, _, _ = np.linalg.svd(avgProj)
        est_cf = U[:, :num_cf]
        est_global_factors = np.sqrt(est_cf.shape[0]) * est_cf
        est_cf_Proj = est_cf @ est_cf.T
        return est_cf, est_global_factors, est_cf_Proj

    def fit(
        self,
        X_list,
        epsilon,
        num_list,
        correctCommonFeature=None,
        epsilon0=None,
        a=None,
        pair_n_method="min",
    ):
        if len(X_list) == 0:
            self.location = []
            self.num_cf = 0
            self.pairs_correct_ratio = 0.0
            self.est_cf = np.array([])
            self.global_factors = np.array([])
            self.est_cf_Proj = np.array([])
            self.est_pf = []
            self.est_pf_proj = np.array([])
            return self

        p = int(X_list[0].shape[1])
        for X in X_list:
            if int(X.shape[1]) != p:
                raise ValueError("X_list 中所有数据集的特征维度 p 必须一致")

        if (epsilon0 is None) != (a is None):
            raise ValueError("epsilon0 and a must be provided together")
        if a is not None and a <= 0:
            raise ValueError("a must be positive")

        pair_epsilons = None
        if epsilon0 is not None:
            pair_epsilons = [
                PCAUtils.pair_effective_n(X_list[i].shape[0], X_list[i + 1].shape[0], pair_n_method)
                ** (1 / 2 - 2 / a - epsilon0)
                for i in range(len(X_list) - 1)
            ]
        elif np.ndim(epsilon) == 1 and not np.isscalar(epsilon):
            pair_epsilons = list(epsilon)
            if len(pair_epsilons) != len(X_list) - 1:
                raise ValueError("epsilon sequence must have one value per adjacent client pair")

        pairs_list = []
        for i in range(len(X_list) - 1):
            pair_epsilon = epsilon if pair_epsilons is None else pair_epsilons[i]
            pairs, _, _ = PCAUtils.identify_prod(
                X_list[i], X_list[i + 1], num_list[i], num_list[i + 1], pair_epsilon
            )
            pairs_list.append(pairs)

        pairs_matrix = PCAUtils.concat_pad(pairs_list)
        location = PCAUtils.extract_paths(pairs_matrix)

        self.location = location
        self.num_cf = len(location)
        if correctCommonFeature is None:
            self.pairs_correct_ratio = 0.0
        else:
            self.pairs_correct_ratio = PCAUtils.column_match_ratio(location, correctCommonFeature)

        self.est_cf, self.global_factors, self.est_cf_Proj = self.estCommonFeature(X_list, location)
        I = np.eye(p)
        P = self.est_cf_Proj if self.est_cf_Proj.size else np.zeros((p, p))
        P_perp = I - P

        pf_list = []
        pf_proj_list = []

        for i, X in enumerate(X_list):
            n = int(X.shape[0])
            cov = (X.T @ X) / max(n, 1)
            resid = P_perp @ cov @ P_perp

            r = int(num_list[i]) - int(self.num_cf)
            r = max(r, 0)

            if r == 0:
                pf = np.zeros((p, 0))
                pf_proj = np.zeros((p, p))
            else:
                vals, vecs = np.linalg.eigh(resid)
                idx = np.argsort(vals)[::-1]
                vecs = vecs[:, idx]
                pf = vecs[:, :r]
                pf_proj = pf @ pf.T

            pf_list.append(pf)
            pf_proj_list.append(pf_proj)

        self.est_pf = pf_list
        self.est_pf_proj = np.stack(pf_proj_list, axis=0)
        return self


class AutoFE:
    def __init__(
        self,
        epsilon=None,
        threshold=None,
        n_repeat=50,
        ip=True,
        avg=True,
        s=20,
        spike_estimator="combine",
        combine_decision_rule="or",
        epsilon0=None,
        a=None,
        pair_n_method="min",
        feature_model=None,
    ):
        if feature_model is None:
            # Import lazily because PathFE subclasses FE from this module.
            from feature_extraction_path import PathFE
            feature_model = PathFE
        self.epsilon = epsilon
        self.threshold = threshold
        self.n_repeat = n_repeat
        self.ip = ip
        self.avg = avg
        self.s = s
        self.spike_estimator = spike_estimator
        self.combine_decision_rule = combine_decision_rule
        self.epsilon0 = epsilon0
        self.a = a
        self.pair_n_method = pair_n_method
        self.feature_model = feature_model

        self.num_list = []
        self.fe = feature_model()

    def fit(self, Data_list):
        if len(Data_list) == 0:
            self.num_list = []
            self.fe.fit([], epsilon=0.0, num_list=[])
            return self

        comb = Combine()
        zheng = ZhengEstimate()
        yao = YaoEstimate()
        num_list = []
        for X in Data_list:
            n = int(X.shape[0])
            p = int(X.shape[1])

            epsilon = self.epsilon
            if epsilon is None:
                epsilon = float(np.sqrt(n) / np.log(max(n, 3)))

            threshold = self.threshold
            if threshold is None:
                threshold = float(1 / np.power(max(n, 1), 1 / 3) / (p / max(n, 1)))

            if self.spike_estimator == "combine":
                q = comb.estimate(
                    X, threshold=threshold, n_repeat=self.n_repeat, ip=self.ip,
                    avg=self.avg, s=self.s, decision_rule=self.combine_decision_rule,
                )
            elif self.spike_estimator == "zheng":
                q = zheng.estimate(X)
            elif self.spike_estimator == "yao":
                q, _, _ = yao.estimate(X)
            else:
                raise ValueError(f"unknown spike_estimator: {self.spike_estimator}")
            num_list.append(int(q))

        self.num_list = num_list

        n_min = min(int(X.shape[0]) for X in Data_list)
        epsilon_fit = self.epsilon
        if epsilon_fit is None:
            epsilon_fit = float(np.sqrt(n_min) / np.log(max(n_min, 3)))

        self.fe = self.feature_model().fit(
            Data_list,
            epsilon=epsilon_fit,
            num_list=num_list,
            epsilon0=self.epsilon0,
            a=self.a,
            pair_n_method=self.pair_n_method,
        )
        return self

    @property
    def global_factors(self):
        return self.fe.global_factors

    @property
    def location(self):
        return self.fe.location

    @property
    def num_cf(self):
        return self.fe.num_cf
