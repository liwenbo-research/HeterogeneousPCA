import numpy as np
import functools


class CP():
    def __init__(self, num_spike_list, device=None):
        self.num_spike_list = num_spike_list
        self.device = device


    def fit(self, Data_list):
        num_client = len(Data_list)
        n_samples = [Data_list[s].shape[0] for s in range(num_client)]
        p = Data_list[0].shape[1]
        N_star = min(np.min(n_samples),p)

        eigenspace = [self._compute_eigen_vectors(Data_list[k], self.num_spike_list[k]) for k in range(num_client)]
        projections = [eigenspace[k]@eigenspace[k].T for k in range(num_client)]
        projection_prod= functools.reduce(lambda a, b: a @ b, projections)
        projection_average=  np.stack(projections).mean(0)

        circular_proj = projection_prod.T @ projection_prod
        circular_vec = self._compute_eigen_vectors(circular_proj)
        ARSSrs =  1- np.sum(circular_vec * (projection_average @ circular_vec), axis=0)
        projectionRatio = self._logistic_fun(1, 0.001, 14, np.log(np.log(N_star)) * ARSSrs[1:])-self._logistic_fun(1, 0.001, 14, np.log(np.log(N_star)) * ARSSrs[:-1])

        self.num_common = np.argmax(projectionRatio) + 1
        # A shared subspace cannot be larger than any participating client PC space.
        self.num_common = min(int(self.num_common), min(self.num_spike_list))
        U_common_est = circular_vec[:,:self.num_common]
        self.positions = [np.argmax(np.abs(U_common_est.T @ eigenspace[k])+1,axis =1) for k in range(num_client)]
        self.global_factors = np.sqrt(p) * U_common_est
        self.global_loadings = [1 / p * Data_list[k] @ self.global_factors for k in range(num_client)]
        common_projection_matrix = U_common_est @ U_common_est.T
        Data_list_hat = [Data_list[k] -  Data_list[k]@common_projection_matrix for k in range(num_client)]
        self.local_factor_list =  [ np.sqrt(p) * self._compute_eigen_vectors(Data_list_hat[k], self.num_spike_list[k]-self.num_common) for k in range(num_client)]
        self.local_loading_list =  [1 / p * Data_list_hat[k] @ self.local_factor_list[k] for k in range(num_client)]

    def _compute_projection_space(self, Data, number_spike):
        U, L, Vt = np.linalg.svd(Data)
        Vs = Vt.T[:, :number_spike]
        Projection = Vs @ Vs.T
        return Projection
    def _compute_eigen_vectors(self, Data, number_spike = 'None'):
        n, p = Data.shape
        if number_spike == 'None':
            number_spike = p
        U, L, Vt = np.linalg.svd(Data)
        Vs = Vt.T[:, :number_spike]
        return Vs


    def _logistic_fun(self, max_value,initial,r,t):
        a=(max_value-initial)/initial;
        return max_value/(1+a*np.exp(-r*t))

    def common_factors(self):
        return self.global_factors

    def common_locations(self):
        return self.positions

    def number_of_global_factors(self):
        return self.num_common
