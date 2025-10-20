import numpy as np
import pandas as pd
import scipy
from numpy.linalg import inv as inv
from scipy.linalg import khatri_rao as kr_prod
from scipy.linalg import cholesky as cholesky_lower
from scipy.linalg import solve_triangular as solve_ut
from scipy.stats import wishart, invwishart
import matplotlib.pyplot as plt


class BTMF:
    def __init__(self, dense_tensor, sparse_tensor, time_lags, rank, burn_iter, gibbs_iter):
        # Y ≈ WᵀX + ε
        # xₜ = A₁xₜ₋₁ + A₂xₜ₋₂ + ... + Aₚxₜ₋ₚ + uₜ
        # 参数模型初始化
        self.dense_tensor = dense_tensor
        self.sparse_tensor = sparse_tensor.copy()
        self.time_lags = np.array(time_lags, dtype=int)
        self.rank = rank
        self.burn_iter = burn_iter
        self.gibbs_iter = gibbs_iter

        self.dim1, self.dim2, self.T = sparse_tensor.shape

        # 将三维张量重塑为二维矩阵 (dim1*dim2, T)
        self.sparse_mat = self.sparse_tensor.reshape(self.dim1 * self.dim2, self.T)
        self.dense_mat = self.dense_tensor.reshape(self.dim1 * self.dim2, self.T)

        # <<< 新增：数据归一化 >>>
        # 1. 计算缩放因子（用真实数据的最大值）
        self.scaler = np.max(self.dense_tensor)
        if self.scaler == 0:
            self.scaler = 1  # 防止除以0

        # 2. 对数据进行缩放
        self.sparse_mat = self.sparse_mat / self.scaler
        self.dense_mat = self.dense_mat / self.scaler

        self.init_parameters()

    def init_parameters(self):
        """初始化模型参数"""
        # 空间因子矩阵 W (dim1*dim2, rank)
        self.W = 0.1 * np.random.randn(self.dim1 * self.dim2, self.rank)

        # 时间因子矩阵 X (T, rank)
        self.X = 0.1 * np.random.randn(self.T, self.rank)

        # VAR系数矩阵 A (rank*d, rank)
        d = len(self.time_lags)
        self.A = 0.1 * np.random.randn(self.rank * d, self.rank)

        # 精度参数 (标量)
        self.tau = 1.0

        # 缺失位置掩码
        self.ind = ~np.isnan(self.sparse_mat)
        self.sparse_mat[np.isnan(self.sparse_mat)] = 0

    def get_reg_strength(self, r):
        """根据rank自适应选择正则化强度"""
        if r <= 20:
            return 1e-3  # 适用于 rank <= 20
        elif r <= 40:
            return 1e-2  # 适用于 20 < rank <= 40
        else:
            return 5e-2  # 适用于 rank > 40 (比如 Seattle)

    def _sample_factor_w(self):
        """采样空间因子W (使用固定自适应正则化)"""
        dim, rank = self.W.shape

        # <<< 使用自适应正则化 >>>
        lambda_w = self.get_reg_strength(self.rank)
        Lambda_w = lambda_w * np.eye(rank)

        tau_ind = self.tau * self.ind
        tau_sparse_mat = self.tau * self.sparse_mat

        var1 = self.X.T  # (rank, T)
        var2 = kr_prod(var1, var1)  # (rank*rank, T*T)
        var3 = (var2 @ tau_ind.T).reshape([rank, rank, dim]) + Lambda_w[..., np.newaxis]
        var4 = var1 @ tau_sparse_mat.T

        for i in range(dim):
            self.W[i, :] = mvnrnd_pre(np.linalg.solve(var3[:, :, i], var4[:, i]), var3[:, :, i])

    def _sample_factor_x(self):
        """采样时间因子X (使用固定自适应正则化)"""
        T, rank = self.X.shape
        d = len(self.time_lags)
        self.max_lag = np.max(self.time_lags)

        if T <= self.max_lag:
            return  # 序列太短，无法采样

        # <<< 使用自适应正则化 >>>
        lambda_x = self.get_reg_strength(self.rank)
        Lambda_x = lambda_x * np.eye(rank)

        # 计算时间自回归部分的贡献
        X_hat_AR = np.zeros((T, rank))
        for t in range(self.max_lag, T):
            X_hat_AR[t, :] = self.A.T @ self.X[t - self.time_lags, :].ravel()

        # 构造用于采样 X_t 的线性模型
        for t in range(self.max_lag, T):
            Lambda_post = self.W.T @ self.W + Lambda_x
            y_obs = self.sparse_mat[:, t] - self.W @ X_hat_AR[t, :].T
            mu_AR = X_hat_AR[t, :]

            mu_post = np.linalg.solve(Lambda_post,
                                      self.W.T @ y_obs + Lambda_x @ mu_AR.T)

            try:
                L = np.linalg.cholesky(Lambda_post)
                X_new = mu_post + np.random.randn(rank) @ L.T
            except np.linalg.LinAlgError:
                print("警告: Cholesky分解失败，使用求逆。")
                Lambda_post_inv = np.linalg.inv(Lambda_post)
                X_new = np.random.multivariate_normal(mu_post, Lambda_post_inv)

            self.X[t, :] = X_new

    def _sample_var_coefficient(self):
        """采样VAR系数A (为高rank增强数值稳定性)"""
        T, rank = self.X.shape
        d = len(self.time_lags)
        tmax = np.max(self.time_lags)

        if T <= tmax:
            return np.eye(rank)  # 返回一个单位矩阵作为默认值

        Z_mat = self.X[tmax:, :]
        Q_mat = np.zeros((T - tmax, rank * d))
        for k in range(d):
            Q_mat[:, k * rank: (k + 1) * rank] = self.X[tmax - self.time_lags[k]: T - self.time_lags[k], :]

        var_Psi0 = Q_mat.T @ Q_mat + 1e-3 * np.eye(rank * d)
        var_Psi = inv(var_Psi0)
        var_M = var_Psi @ Q_mat.T @ Z_mat

        var_S = Z_mat.T @ Z_mat - var_M.T @ var_Psi0 @ var_M
        var_S = var_S + 1e-3 * np.eye(rank)
        Sigma = invwishart.rvs(df=rank + T - tmax, scale=var_S)

        cov_A = np.kron(Sigma, var_Psi)
        cov_A = cov_A + 1e-3 * np.eye(cov_A.shape[0])

        try:
            self.A = np.random.multivariate_normal(var_M.ravel(), cov_A).reshape(rank * d, rank)
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"警告: multivariate_normal采样失败 ({e})，使用Cholesky分解备选方案。")
            try:
                L = np.linalg.cholesky(cov_A)
                self.A = var_M.ravel() + np.random.randn(var_M.size) @ L.T
                self.A = self.A.reshape(rank * d, rank)
            except np.linalg.LinAlgError:
                print("警告: Cholesky分解也失败，使用均值作为采样结果。")
                self.A = var_M.reshape(rank * d, rank)

        return Sigma

    def _sample_precision_tau(self):
        """采样精度参数tau（使用温和的先验）"""
        mat_hat = self.W @ self.X.T
        # 使用温和的先验，因为数据已经归一化
        var_alpha = 1e-3 + 0.5 * np.sum(self.ind)
        var_beta = 1e-3 + 0.5 * np.sum(((self.sparse_mat - mat_hat) ** 2) * self.ind)
        self.tau = np.random.gamma(var_alpha, 1 / var_beta)

    def gibbs_sampling(self):
        W_plus = np.zeros_like(self.W)
        X_plus = np.zeros_like(self.X)
        A_plus = np.zeros_like(self.A)
        mat_hat_plus = np.zeros_like(self.sparse_mat)

        for it in range(self.burn_iter + self.gibbs_iter):
            is_verbose_iter = (it + 1) % 50 == 0 or (it + 1) <= 10
            if is_verbose_iter:
                print(f"  Gibbs Iteration: {it + 1} / {self.burn_iter + self.gibbs_iter}")
                lambda_w = self.get_reg_strength(self.rank)
                lambda_x = self.get_reg_strength(self.rank)
                print(f"    > Sampling W (reg={lambda_w:.0e})...")
            self._sample_factor_w()

            if is_verbose_iter:
                print("    > Sampling Var Coefficient...")
            Sigma = self._sample_var_coefficient()

            if is_verbose_iter:
                print(f"    > Sampling X (reg={lambda_x:.0e})...")
            self._sample_factor_x()

            if it < 20:
                self.tau = 1.0
                if is_verbose_iter:
                    print(f"    > Sampling Tau (tau={self.tau:.2f}) [FORCED]...")
            else:
                if is_verbose_iter:
                    print(f"    > Sampling Tau (tau={self.tau:.2f})...")
                self._sample_precision_tau()

            mat_hat = self.W @ self.X.T

            if it + 1 > self.burn_iter:
                W_plus += self.W
                X_plus += self.X
                A_plus += self.A
                mat_hat_plus += mat_hat

        # 计算后验均值
        self.W = W_plus / self.gibbs_iter
        self.X = X_plus / self.gibbs_iter
        self.A = A_plus / self.gibbs_iter
        mat_hat = mat_hat_plus / self.gibbs_iter

        # 将结果重塑回张量形式
        tensor_hat = mat_hat.reshape(self.dim1, self.dim2, self.T)
        tensor_hat = tensor_hat * self.scaler
        tensor_hat[tensor_hat < 0] = 0

        return tensor_hat

    def compute_mape(self, var, var_hat):
        return np.sum(np.abs(var - var_hat) / (var + 1e-6)) / var.shape[0]

    def compute_rmse(self, var, var_hat):
        return np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])


def mvnrnd_pre(mu, Lambda):
    L = cholesky_lower(Lambda)
    return np.random.multivariate_normal(mu, inv(Lambda))
