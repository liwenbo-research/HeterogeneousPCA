import numpy as np



class APM():
    def __init__(self, num_spike_list, device=None):
        self.num_spike_list = num_spike_list
        self.device = device


    def fit(self, Data_list, method = "IC"):
        num_client = len(Data_list)
        n_list =   [Data_list[k].shape[0] for k in range(num_client)]
        p =  Data_list[0].shape[1]
        eigen_space = [self._compute_eigen_vectors(Data_list[k], self.num_spike_list[k]) for k in range(num_client)]
        projections = [eigen_space[k]@eigen_space[k].T for k in range(num_client)]

        weight_project = np.stack(projections).mean(0)

        w, v = np.linalg.eigh(weight_project)

        idx = np.argsort(-w)
        w = w[idx]
        v = v[:, idx]
        if method== "IC":
            error_list = [Data_list[k] - Data_list[k] @ projections[k] for k in range(num_client)]

            sigma_total = np.sum(np.concatenate(Data_list) ** 2) / (np.sum(n_list)) / p
            sigma_error = np.sum(np.concatenate(error_list) ** 2) / (np.sum(n_list)) / p
            level = (1 - 1 / num_client) * np.exp(sigma_error / sigma_total)
            n_min = min(n_list)
            delta = np.sqrt((min(n_min, p)))
            gnt = level * (np.sqrt(n_min) + p) / np.sqrt(n_min) / p * np.log(np.log(delta))

            self.num_common = np.sum(w > 1 - gnt).item()
        if method == "GAP":
            self.num_common = np.argmax(w[:-1]-w[1:]).item()+1

        # A shared subspace cannot be larger than any participating client PC space.
        self.num_common = min(int(self.num_common), min(self.num_spike_list))

        U_common_est = v[:,:self.num_common]
        self.positions = [np.argmax(np.abs(U_common_est.T @ eigen_space[k])+1,axis =1) for k in range(num_client)]

        self.global_factors = np.sqrt(p) * U_common_est
        self.global_loadings = [1 / p * Data_list[k] @ self.global_factors for k in range(num_client)]
        common_projection_matrix = U_common_est @ U_common_est.T
        Data_list_hat = [Data_list[k] -  Data_list[k]@common_projection_matrix for k in range(num_client)]
        self.local_factor_list =  [ np.sqrt(p) * self._compute_eigen_vectors(Data_list_hat[k], self.num_spike_list[k]-self.num_common) for k in range(num_client)]
        self.local_loading_list =  [1 / p * Data_list_hat[k] @ self.local_factor_list[k] for k in range(num_client)]

    def _compute_eigen_vectors(self, Data, number_spike):
        U, L, Vt = np.linalg.svd(Data)
        Vs = Vt.T[:, :number_spike]
        return Vs
    def common_factors(self):
        return self.global_factors

    def common_locations(self):
        return self.positions
    def number_of_global_factors(self):
        return self.num_common
