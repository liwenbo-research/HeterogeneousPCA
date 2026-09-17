import math
import numpy as np
from scipy.linalg import block_diag


class PCAUtils:
    @staticmethod
    def create_sigma_matrix_custom_bulk(dim, lam, p, bulk_values, nuisance_blocks):
        """Keep the existing block spikes and append diagonal/nuisance bulk blocks."""
        scale = [(lam[i] - 0.25) / dim[i] for i in range(len(lam))]
        spike_blocks = [
            value * (np.ones((size, 1)) @ np.ones((size, 1)).T) + 0.25 * np.eye(size)
            for value, size in zip(scale, dim)
        ]
        remaining = p - sum(dim) - sum(block.shape[0] for block in nuisance_blocks)
        if remaining < len(bulk_values):
            raise ValueError("p is too small for the requested nuisance bulk blocks")
        base_size, remainder = divmod(remaining, len(bulk_values))
        diagonal_blocks = [
            value * np.eye(base_size + (index < remainder))
            for index, value in enumerate(bulk_values)
        ]
        sigma = block_diag(*spike_blocks, *diagonal_blocks, *nuisance_blocks)
        if sigma.shape != (p, p):
            raise RuntimeError(f"Sigma shape mismatch: {sigma.shape}")
        return sigma

    @staticmethod
    def create_sigma_matrix_gamma_bulk(dim, lam, p, theta, sigma_sq=1.0, random_state=None):
        """Generate BEMA's general spiked covariance model.

        The residual variances are iid Gamma(theta, theta / sigma_sq), and
        the signal directions are uniformly sampled from the Stiefel manifold.
        """
        del dim  # The BEMA model is defined by rank and spike strengths only.
        lam = np.asarray(lam, dtype=float)
        if lam.ndim != 1 or len(lam) == 0 or np.any(lam <= 0):
            raise ValueError("lam must be a nonempty vector of positive spike strengths")
        if theta <= 0 or sigma_sq <= 0:
            raise ValueError("theta and sigma_sq must be positive")
        if p <= len(lam):
            raise ValueError("p must exceed the number of spikes")

        rng = np.random.default_rng(random_state)
        residual_variances = rng.gamma(shape=theta, scale=sigma_sq / theta, size=p)
        gaussian_matrix = rng.standard_normal((p, len(lam)))
        eigenvectors, _ = np.linalg.qr(gaussian_matrix, mode="reduced")
        return np.diag(residual_variances) + (eigenvectors * lam) @ eigenvectors.T

    @staticmethod
    def create_sigma_matrix_gamma_correlated_bulk(
        dim, lam, p, theta, sigma_sq=1.0, correlated_size=40,
        correlated_variance=0.15, correlated_rho=0.03, random_state=None,
    ):
        """Fixed block spikes with Gamma residuals plus a weak correlated block.

        The Gamma diagonal component follows BEMA's bulk generation. The
        correlated block is a controlled misspecification that preserves
        covariance-scale spikes while testing methods against residual
        dependence.
        """
        rng = np.random.default_rng(random_state)
        dim = [int(value) for value in dim]
        lam = np.asarray(lam, dtype=float)
        if len(dim) != len(lam) or np.any(lam <= 0):
            raise ValueError("dim and lam must have matching positive entries")
        correlated_size = int(correlated_size)
        remaining = p - sum(dim)
        if correlated_size < 2 or correlated_size >= remaining:
            raise ValueError("correlated_size must fit inside the residual dimensions")
        if theta <= 0 or sigma_sq <= 0 or correlated_variance <= 0:
            raise ValueError("theta, sigma_sq, and correlated_variance must be positive")
        if not (-1 / (correlated_size - 1) < correlated_rho < 1):
            raise ValueError("correlated_rho does not yield a positive definite block")

        spike_blocks = []
        for size, strength in zip(dim, lam):
            scale = (strength - 0.25) / size
            spike_blocks.append(scale * np.ones((size, size)) + 0.25 * np.eye(size))
        diagonal_size = remaining - correlated_size
        residual_variances = rng.gamma(shape=theta, scale=sigma_sq / theta, size=diagonal_size)
        diagonal_block = np.diag(residual_variances)
        correlation = (1 - correlated_rho) * np.eye(correlated_size) + correlated_rho * np.ones((correlated_size, correlated_size))
        correlated_block = correlated_variance * correlation
        return block_diag(*spike_blocks, diagonal_block, correlated_block)

    @staticmethod
    def create_sigma_matrix(
        dim,
        lam,
        p,
        num_bulk=2,
        bulk_values=None,
        correlated_bulk_size=0,
        correlated_bulk_variance=0.15,
        correlated_bulk_rho=0.0,
    ):
        scale = [(lam[i] - 0.25) / dim[i] for i in range(len(lam))]

        block_spike_list = []
        for i in range(len(lam)):
            ones = np.ones((dim[i], 1))
            I = np.eye(dim[i])
            block = scale[i] * (ones @ ones.T) + 0.25 * I
            block_spike_list.append(block)

        block_spike = block_diag(*block_spike_list)
        dim_spike_total = sum(dim)

        dim_remaining = p - dim_spike_total
        if dim_remaining <= 0:
            raise ValueError(f"p={p} 不足以容纳 spike 维度 {dim_spike_total}")

        correlated_bulk_size = int(correlated_bulk_size)
        if not 0 <= correlated_bulk_size < dim_remaining:
            raise ValueError("correlated_bulk_size 必须在 [0, 剩余维度) 内")
        if correlated_bulk_variance <= 0:
            raise ValueError("correlated_bulk_variance 必须为正")
        if correlated_bulk_size > 1 and not (
            -1 / (correlated_bulk_size - 1) < correlated_bulk_rho < 1
        ):
            raise ValueError("correlated_bulk_rho 不满足等相关矩阵的正定条件")

        if bulk_values is not None:
            bulk_values = [float(value) for value in bulk_values]
            if not bulk_values:
                raise ValueError("bulk_values 不能为空")
            num_bulk = len(bulk_values)
        num_bulk = min(int(num_bulk), 2)
        if bulk_values is None:
            bulk_values = [1.0 - 0.5 * i for i in range(num_bulk)]
        elif len(bulk_values) < num_bulk:
            raise ValueError("bulk_values 长度不足")

        diagonal_bulk_dim = dim_remaining - correlated_bulk_size
        bulk_dim = diagonal_bulk_dim // num_bulk
        remainder = diagonal_bulk_dim % num_bulk

        block_bulk_list = []
        for i in range(num_bulk):
            weight = bulk_values[i]
            size = bulk_dim + (1 if i < remainder else 0)
            block = weight * np.eye(size)
            block_bulk_list.append(block)

        if correlated_bulk_size:
            correlation = (
                (1 - correlated_bulk_rho) * np.eye(correlated_bulk_size)
                + correlated_bulk_rho * np.ones((correlated_bulk_size, correlated_bulk_size))
            )
            block_bulk_list.append(correlated_bulk_variance * correlation)

        Sigma = block_diag(block_spike, *block_bulk_list)

        if Sigma.shape != (p, p):
            raise RuntimeError(f"Sigma 维度错误: {Sigma.shape}")

        return Sigma

    @staticmethod
    def create_sigma_matrix_one_bulk(dim, lam, p):
        cons = [(lam[i] - 0.25) / dim[i] for i in range(len(lam))]

        block_spike_list = []
        for i in range(len(lam)):
            ones = np.ones((dim[i], 1))
            I = np.eye(dim[i])
            block = cons[i] * (ones @ ones.T) + 0.25 * I
            block_spike_list.append(block)

        if len(block_spike_list) > 1:
            block_spike = block_diag(*block_spike_list)
        else:
            block_spike = block_spike_list[0]

        dim_sum = sum(dim)
        block_non1 = np.eye(p - dim_sum)

        Sigma = block_diag(block_spike, block_non1)
        return Sigma

    @staticmethod
    def generate_data_from_sigma(Sigma, n):
        p = Sigma.shape[0]
        mean = np.zeros(p)
        return np.random.multivariate_normal(mean, Sigma, size=n)

    @staticmethod
    def generate_data_1(Sigma, n, df=50, normal=True, distribution=None):
        if distribution is None:
            distribution = "normal" if normal else "t"
        if distribution == "exp":
            z = np.random.exponential(scale=1.0, size=(n, Sigma.shape[0])) - 1.0
            return z @ np.linalg.cholesky(Sigma).T
        if distribution == "normal":
            return PCAUtils.generate_data_from_sigma(Sigma, n)
        scale = (df - 2) / df
        Z = PCAUtils.generate_data_from_sigma(scale * Sigma, n)
        W = np.random.chisquare(df, size=n)
        return Z / np.sqrt(W / df)[:, None]

    @staticmethod
    def compute_eigenvectors(X, center=False):
        X = np.asarray(X, dtype=float)
        if center:
            X = X - X.mean(axis=0, keepdims=True)

        n = X.shape[0]
        cov = (X.T @ X) / n
        cov = (cov + cov.T) * 0.5
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        return eigenvectors[:, np.argsort(eigenvalues)[::-1]]

    @staticmethod
    def compute_eigenvalues(X, center=False):
        X = np.asarray(X, dtype=float)
        if center:
            X = X - X.mean(axis=0, keepdims=True)

        n = X.shape[0]
        cov = (X.T @ X) / n
        cov = (cov + cov.T) * 0.5
        return np.linalg.eigvalsh(cov)[::-1]

    @staticmethod
    def compute_eigenvectors_from_cov(S):
        vals, vecs = np.linalg.eigh(S)
        idx = vals.argsort()[::-1]
        vecs = vecs[:, idx]
        return vecs

    @staticmethod
    def sample_and_remainder(X, n_samples):
        n = X.shape[0]
        if n_samples > n:
            raise ValueError("n_samples 超过样本数")

        idx = np.random.permutation(n)
        return X[idx[:n_samples]], X[idx[n_samples:]]

    @staticmethod
    def all_column_inner_products(X, Y):
        if X.shape[0] != Y.shape[0]:
            raise ValueError("行数必须一致")
        return X.T @ Y

    @staticmethod
    def compute_phi1(X, sample_eigenvalues, i, num_spike):
        n, p = X.shape
        lam_i = sample_eigenvalues[i]
        if lam_i < 1e-12:
            return 0.0
        tail = sample_eigenvalues[num_spike:]
        
        # 避免除零：只对大于 lam_i 的 tail 元素求倒数，或者添加 epsilon
        # 实际上 lam_i 应该大于 tail 中的所有元素
        diff = tail - lam_i
        mask = np.abs(diff) > 1e-12
        if not np.any(mask):
            return 0.0
            
        term2 = (1 / n) * np.sum(1 / diff[mask])
        phi1 = -(1 - (p - num_spike) / n) / lam_i + term2
        return phi1

    @staticmethod
    def compute_phi2(X, sample_eigenvalues, i, num_spike):
        n, p = X.shape
        lam_i = sample_eigenvalues[i]
        if lam_i < 1e-12:
            return 1.0 # 避免除零，返回一个正数
        tail = sample_eigenvalues[num_spike:]
        
        diff = tail - lam_i
        mask = np.abs(diff) > 1e-12
        if not np.any(mask):
            return 1.0
            
        term2 = (1 / n) * np.sum(1 / (diff[mask] ** 2))
        phi2 = (1 - (p - num_spike) / n) / (lam_i ** 2) + term2
        return phi2

    @staticmethod
    def compute_theta(X, i, num_spike):
        sample_eigenvalues = PCAUtils.compute_eigenvalues(X)
        if i >= len(sample_eigenvalues):
            return 0.0
            
        phi1 = PCAUtils.compute_phi1(X, sample_eigenvalues, i, num_spike=num_spike)
        phi2 = PCAUtils.compute_phi2(X, sample_eigenvalues, i, num_spike=num_spike)
        lam_i = sample_eigenvalues[i]
        
        if lam_i * phi2 == 0 or (-phi1 / (lam_i * phi2)) < 0:
            return 0.0
            
        theta = math.sqrt(-phi1 / (lam_i * phi2))
        return theta

    @staticmethod
    def pair_effective_n(n1, n2, method="min"):
        """Aggregate two clients' sample sizes for an FE pairwise threshold."""
        n1 = int(n1)
        n2 = int(n2)
        if n1 <= 0 or n2 <= 0:
            raise ValueError("pairwise sample sizes must be positive")
        if method == "min":
            return float(min(n1, n2))
        if method == "harmonic":
            return float(2 * n1 * n2 / (n1 + n2))
        raise ValueError("pair_n_method must be 'min' or 'harmonic'")

    @staticmethod
    def identify_prod(X, Y, num1, num2, alpha):
        # Compute each client's spectrum once.  The old implementation called
        # compute_eigenvalues once per candidate spike inside compute_theta,
        # turning one pair comparison into O(num_spike) repeated SVDs.
        sample_eigenvalues_X = PCAUtils.compute_eigenvalues(X)
        sample_eigenvalues_Y = PCAUtils.compute_eigenvalues(Y)
        eigvecs_X = PCAUtils.compute_eigenvectors_from_cov((X.T @ X) / max(X.shape[0], 1))[:, :num1]
        eigvecs_Y = PCAUtils.compute_eigenvectors_from_cov((Y.T @ Y) / max(Y.shape[0], 1))[:, :num2]

        inner_X_Y = PCAUtils.all_column_inner_products(eigvecs_X, eigvecs_Y)
        inner_X_Y_abs = np.abs(inner_X_Y)

        def theta_from_values(values, i, num_spike, n_obs):
            lam_i = values[i]
            if lam_i < 1e-12:
                return 0.0
            tail = values[num_spike:]
            diff = tail - lam_i
            mask = np.abs(diff) > 1e-12
            if not np.any(mask):
                return 0.0
            phi1 = -(1 - (len(values) - num_spike) / n_obs) / lam_i
            phi1 += (1 / n_obs) * np.sum(1 / diff[mask])
            phi2 = (1 - (len(values) - num_spike) / n_obs) / (lam_i ** 2)
            phi2 += (1 / n_obs) * np.sum(1 / (diff[mask] ** 2))
            ratio = -phi1 / (lam_i * phi2) if lam_i * phi2 else -1
            return math.sqrt(ratio) if ratio >= 0 else 0.0

        thetaX = np.array([theta_from_values(sample_eigenvalues_X, i, num1, X.shape[0]) for i in range(num1)])
        thetaY = np.array([theta_from_values(sample_eigenvalues_Y, j, num2, Y.shape[0]) for j in range(num2)])

        denom = thetaX[:, None] * thetaY[None, :]
        # 避免除以零，对于 denom 为 0 的位置，ratio 设为 0
        ratio = np.zeros_like(inner_X_Y_abs)
        nonzero_mask = denom > 1e-12
        ratio[nonzero_mask] = inner_X_Y_abs[nonzero_mask] / denom[nonzero_mask]

        mask = ratio > (1 - 1 / alpha)
        indices = np.argwhere(mask)
        return indices, ratio, inner_X_Y_abs

    @staticmethod
    def identify_prod_from_spectra(values_X, vectors_X, n_X, values_Y, vectors_Y, n_Y, num1, num2, alpha):
        """Same FE pair statistic as ``identify_prod``, using cached spectra."""
        eigvecs_X = np.asarray(vectors_X)[:, :num1]
        eigvecs_Y = np.asarray(vectors_Y)[:, :num2]
        inner_X_Y_abs = np.abs(eigvecs_X.T @ eigvecs_Y)

        def theta(values, i, rank, n_obs):
            lam = values[i]
            if lam < 1e-12:
                return 0.0
            diff = values[rank:] - lam
            mask = np.abs(diff) > 1e-12
            if not np.any(mask):
                return 0.0
            phi1 = -(1 - (len(values) - rank) / n_obs) / lam + np.sum(1 / diff[mask]) / n_obs
            phi2 = (1 - (len(values) - rank) / n_obs) / lam**2 + np.sum(1 / diff[mask]**2) / n_obs
            value = -phi1 / (lam * phi2) if lam * phi2 else -1.0
            return math.sqrt(value) if value >= 0 else 0.0

        theta_X = np.array([theta(values_X, i, num1, n_X) for i in range(num1)])
        theta_Y = np.array([theta(values_Y, j, num2, n_Y) for j in range(num2)])
        denominator = theta_X[:, None] * theta_Y[None, :]
        ratio = np.zeros_like(inner_X_Y_abs)
        valid = denominator > 1e-12
        ratio[valid] = inner_X_Y_abs[valid] / denominator[valid]
        return np.argwhere(ratio > 1 - 1 / alpha), ratio, inner_X_Y_abs

    @staticmethod
    def concat_pad(arr_list, pad_value=-1):
        max_rows = max(a.shape[0] for a in arr_list)

        padded_list = []
        for a in arr_list:
            rows, cols = a.shape
            if rows < max_rows:
                pad = np.full((max_rows - rows, cols), pad_value)
                a = np.vstack((a, pad))
            padded_list.append(a)

        return np.hstack(padded_list)

    @staticmethod
    def extract_paths(result):
        rows, cols = result.shape
        path_list = []

        for start_row in range(rows):
            c = 1
            cur_row = start_row
            path = [result[cur_row, 0], result[cur_row, 1]]

            while c < cols - 1:
                target = result[cur_row, c]
                next_col = c + 1
                found = False

                for r in range(rows):
                    if result[r, next_col] == target:
                        cur_row = r
                        c = next_col + 1
                        if c < cols:
                            path.append(result[cur_row, c])
                        found = True
                        break

                if not found:
                    break
            if c == cols - 1:
                path_list.append(path)

        return path_list

    @staticmethod
    def column_match_ratio(A, B):
        A = np.array(A)
        B = np.array(B)

        if A.size == 0:
            return 0.0

        cols_a = A.shape[1]
        cols_b = B.shape[1]
        cols = min(cols_a, cols_b)

        total_matches = 0
        for c in range(cols):
            set_a = set(A[:, c])
            set_b = set(B[:, c])
            matches = len(set_a.intersection(set_b))
            total_matches += matches

        total_b = B.size
        return total_matches / total_b

    @staticmethod
    def build_correct_common_feature(num_datasets):
        pattern = [(0, 1), (0, 2), (1, 2)]
        row0 = []
        row1 = []
        for i in range(num_datasets):
            a, b = pattern[i % len(pattern)]
            row0.append(a)
            row1.append(b)
        return [row0, row1]

    @staticmethod
    def TR(A, B):
        if B.size == 0:
            return 0.0

        M = A.T @ B @ np.linalg.pinv(B.T @ B) @ B.T @ A
        tr = np.trace(M) / np.trace(A.T @ A)
        return float(tr)

    @staticmethod
    def tk(Data_list, num_cf, num_spike):
        p = Data_list[0].shape[1]
        k = num_spike
        m = len(Data_list)

        hat_Sigma = np.zeros((p, p))
        for X in Data_list:
            U = PCAUtils.compute_eigenvectors(X)[:, :k]
            hat_Sigma += U @ U.T

        hat_Sigma /= m
        V, _, _ = np.linalg.svd(hat_Sigma)
        cf = V[:, :num_cf] * np.sqrt(p)
        return cf

    @staticmethod
    def create_sigma_matrix_private(dim, lam, p, private_vec=None, num_bulk=1):
        block_spike_list = []
        for i in range(len(lam)):
            if i == len(lam) - 1 and private_vec is not None:
                v = np.asarray(private_vec, dtype=float).reshape(-1, 1)
                block = (lam[i] - 0.25) * (v @ v.T) + 0.25 * np.eye(v.shape[0])
            else:
                ones = np.ones((dim[i], 1))
                I = np.eye(dim[i])
                scale = (lam[i] - 0.25) / dim[i]
                block = scale * (ones @ ones.T) + 0.25 * I
            block_spike_list.append(block)

        block_spike = PCAUtils._block_diag(block_spike_list)
        dim_spike_total = block_spike.shape[0]

        dim_remaining = p - dim_spike_total
        if dim_remaining <= 0:
            raise ValueError(f"总维度 p={p} 太小，不足以容纳 spike 维度 {dim_spike_total}。")

        bulk_dim = dim_remaining // num_bulk
        remainder = dim_remaining % num_bulk

        block_bulk_list = []
        num_bulk = min(num_bulk, 3)
        for i in range(num_bulk):
            weight = 1.0 - 0.25 * i
            size = bulk_dim + (1 if i < remainder else 0)
            block = weight * np.eye(size)
            block_bulk_list.append(block)

        Sigma = PCAUtils._block_diag([block_spike] + block_bulk_list)

        if Sigma.shape != (p, p):
            raise RuntimeError(f"Sigma 维度错误: {Sigma.shape}, 期望 ({p}, {p})")

        return Sigma

    @staticmethod
    def _block_diag(mats):
        sizes = [m.shape[0] for m in mats]
        total = sum(sizes)
        out = np.zeros((total, total))
        offset = 0
        for m in mats:
            n = m.shape[0]
            out[offset:offset + n, offset:offset + n] = m
            offset += n
        return out

    @staticmethod
    def generate_private_vec(num_clients, orthonormal=True, rho=0.2):
        """Generate one unit private direction per client.

        The private block has ``num_clients + 1`` coordinates.  The extra
        coordinate is the common anchor used by the controlled nonorthogonal
        construction; it remains unused in the orthogonal construction.
        """
        if num_clients <= 0:
            raise ValueError("num_clients must be positive")
        if not 0 <= rho < 1:
            raise ValueError("rho must satisfy 0 <= rho < 1")

        private_dim = num_clients + 1
        if orthonormal:
            # v_k = e_k, k=1,...,K; e_0 is reserved as the nonorthogonal anchor.
            return np.eye(private_dim, dtype=float)[:, 1:]

        directions = np.zeros((private_dim, num_clients), dtype=float)
        directions[0, :] = math.sqrt(rho)
        directions[1:, :] = math.sqrt(1 - rho) * np.eye(num_clients)
        return directions

    @staticmethod
    def generate_cov_matrix(num, orthonormal, dim, lam, p, num_bulk=2, private_rho=0.2):
        Sigma_list = []
        Q = PCAUtils.generate_private_vec(num, orthonormal, rho=private_rho)

        lam_orders = [
            [lam[0], lam[1], lam[2]],
            [lam[0], lam[2], lam[1]],
            [lam[1], lam[2], lam[0]],
        ]

        for i in range(num):
            lam_i = lam_orders[i % len(lam_orders)]
            Sigma = PCAUtils.create_sigma_matrix_private(
                dim,
                lam_i,
                p,
                private_vec=Q[:, i],
                num_bulk=num_bulk,
            )
            Sigma_list.append(Sigma)

        return Sigma_list

    @staticmethod
    def generate_data_multi(
        num, orthonormal, dim, lam, p, n, num_bulk=2, normal=True, private_rho=0.2
    ):
        Data_list = []
        Cov_list = PCAUtils.generate_cov_matrix(
            num, orthonormal, dim, lam, p, num_bulk, private_rho=private_rho
        )

        for i in range(num):
            Sigma = Cov_list[i]
            if normal:
                mean = np.zeros(p)
                X = np.random.multivariate_normal(mean, Sigma, size=n)
            else:
                L = np.linalg.cholesky(Sigma)
                Z = np.random.exponential(scale=1.0, size=(p, n)) - 1.0
                X = (L @ Z).T
            Data_list.append(X)

        return Data_list, Cov_list
