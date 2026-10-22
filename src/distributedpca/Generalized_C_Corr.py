import numpy as np


class GCC():
    def __init__(self, num_spike_list, device=None):
        self.num_spike_list = num_spike_list
        self.device = device


    def fit(self, Data_list):
        num_client = len(Data_list)
        n_samples = [Data_list[s].shape[0] for s in range(num_client)]
        p = Data_list[0].shape[1]
        all_factors = [np.sqrt(p) * self._compute_sig_values_vectors(Data_list[k], self.num_spike_list[k])[1] for k in range(num_client)]
        gram = self._get_phi_gram(all_factors)

        eigen_values, eigen_vectores = np.linalg.eigh(gram)
        idx = np.argsort(eigen_values)
        eigen_values = eigen_values[idx]
        eigen_vectores = eigen_vectores[:, idx]

        self.num_common = np.argmax(eigen_values[1:]/eigen_values[:-1]) + 1
        # A shared subspace cannot be larger than any participating client PC space.
        self.num_common = min(int(self.num_common), min(self.num_spike_list))
        Qr = eigen_vectores[:,:self.num_common]

        currentindex = 0
        for k in range(num_client):
            Qrk = Qr[currentindex:(currentindex+self.num_spike_list[k]),:]
            currentindex += self.num_spike_list[k]
            PhiQrk = all_factors[k]@Qrk
            if k == 0:
                PhiQr = PhiQrk
            else:
                PhiQr = np.concatenate((PhiQr, PhiQrk), axis=1)
        U_common_est = self._compute_sig_values_vectors(PhiQr.T)[1][:,:self.num_common]
        self.positions = [np.argmax(np.abs(U_common_est.T @ all_factors[k])+1,axis =1) for k in range(num_client)]
        self.global_factors = np.sqrt(p) * U_common_est
        self.global_loadings = [1 / p * Data_list[k] @ self.global_factors for k in range(num_client)]
        common_projection_matrix = U_common_est @ U_common_est.T
        Data_list_hat = [Data_list[k] -  Data_list[k]@common_projection_matrix for k in range(num_client)]
        self.local_factor_list =  [ np.sqrt(p) * self._compute_sig_values_vectors(Data_list_hat[k], self.num_spike_list[k]-self.num_common)[1] for k in range(num_client)]
        self.local_loading_list =  [1 / p * Data_list_hat[k] @ self.local_factor_list[k] for k in range(num_client)]

    def _compute_projection_space(self, Data, number_spike):
        U, L, Vt = np.linalg.svd(Data)
        Vs = Vt.T[:, :number_spike]
        Projection = Vs @ Vs.T
        return Projection
    def _compute_sig_values_vectors(self, Data, number_spike = 'None'):
        n, p = Data.shape
        if number_spike == 'None':
            number_spike = p
        U, L, Vt = np.linalg.svd(Data)
        Vs = Vt.T[:, :number_spike]
        return L, Vs



    def _get_phi(self,K_list):
        R = len(K_list)
        T = K_list[0].shape[0]
        d = [K.shape[1] for K in K_list]
        csumd = np.cumsum(d)

        Phi = np.zeros((T * R * (R - 1) // 2, sum(d)))
        row = 0

        for m in range(R - 1):
            for h in range(m + 1, R):
                mstart = 0 if m == 0 else csumd[m - 1]
                mend = csumd[m]
                hstart = csumd[h - 1]
                hend = csumd[h]

                Phi[row:row + T, mstart:mend] = K_list[m]
                Phi[row:row + T, hstart:hend] = -K_list[h]
                row += T

        return Phi

    def _get_phi_gram(self, K_list):
        """Compute ``Phi.T @ Phi`` without materializing the pairwise Phi."""
        num_client = len(K_list)
        dimensions = [K.shape[1] for K in K_list]
        offsets = np.cumsum([0] + dimensions)
        gram = np.zeros((offsets[-1], offsets[-1]), dtype=float)
        for i, Ki in enumerate(K_list):
            i0, i1 = offsets[i:i + 2]
            gram[i0:i1, i0:i1] = (num_client - 1) * (Ki.T @ Ki)
            for j in range(i + 1, num_client):
                j0, j1 = offsets[j:j + 2]
                cross = -(Ki.T @ K_list[j])
                gram[i0:i1, j0:j1] = cross
                gram[j0:j1, i0:i1] = cross.T
        return gram

    def common_factors(self):
        return self.global_factors

    def common_locations(self):
        return self.positions

    def number_of_global_factors(self):
        return self.num_common
