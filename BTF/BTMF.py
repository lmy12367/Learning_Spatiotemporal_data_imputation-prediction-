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
        #参数模型初始化
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

        # 因子W的先验超参数
        self.Lambda_w = np.eye(self.rank)
        self.mu_w_hyper = np.zeros(self.rank)

        # 因子X的先验超参数
        self.Lambda_x = np.eye(self.rank)
        self.mu_x_hyper = np.zeros(self.rank)

        # 缺失位置掩码
        self.ind = ~np.isnan(self.sparse_mat)
        self.sparse_mat[np.isnan(self.sparse_mat)] = 0

    def _sample_factor_w(self):
        """采样空间因子W"""
        dim, rank = self.W.shape

        # 更新Lambda_w的超参数
        W_bar = np.mean(self.W, axis=0)
        temp = dim / (dim + 1)
        var_W_hyper = inv(np.eye(rank) + (self.W - W_bar).T @ (self.W - W_bar) + temp * np.outer(W_bar, W_bar))
        self.Lambda_w = wishart.rvs(df=dim + rank, scale=var_W_hyper)
        self.mu_w_hyper = mvnrnd_pre(temp * W_bar, (dim + 1) * self.Lambda_w)

        # 更新W
        tau_ind = self.tau * self.ind
        tau_sparse_mat = self.tau * self.sparse_mat

        var1 = self.X.T  # (rank, T)
        var2 = kr_prod(var1, var1)  # (rank*rank, T*T)
        var3 = (var2 @ tau_ind.T).reshape([rank, rank, dim]) + self.Lambda_w[:, :, None]
        var4 = var1 @ tau_sparse_mat.T + (self.Lambda_w @ self.mu_w_hyper)[:, None]

        for i in range(dim):
            self.W[i, :] = mvnrnd_pre(np.linalg.solve(var3[:, :, i], var4[:, i]), var3[:, :, i])

    def _sample_factor_x(self):
        """采样时间因子X (基于 Y=W@X.T 的清晰实现)"""
        T, rank = self.X.shape
        d = len(self.time_lags)
        self.max_lag = np.max(self.time_lags)

        if T <= self.max_lag:
            return  # 序列太短，无法采样

        #计算时间自回归部分的贡献
        X_hat_AR = np.zeros((T, rank))
        for t in range(self.max_lag, T):
            X_hat_AR[t, :] = self.A.T @ self.X[t - self.time_lags, :].ravel()

        #构造用于采样 X_t 的线性模型
        #我们逐个时间点 t (从 max_lag 到 T-1) 进行采样
        for t in range(self.max_lag, T):
            # 观测值: sparse_mat[:, t] (dim1*dim2, 1)
            # 设计矩阵: W (dim1*dim2, rank)
            # 已知部分: W @ X_hat_AR[t, :].T
            # 后验精度 Lambda_post = W.T @ W + Lambda_x
            Lambda_post = self.W.T @ self.W + self.Lambda_x

            # 后验均值 mu_post = Lambda_post^{-1} @ (W.T @ y_obs + Lambda_x @ mu_AR)
            # y_obs 是观测值减去其他已知项
            y_obs = self.sparse_mat[:, t] - self.W @ X_hat_AR[t, :].T
            # VAR模型的预测值作为先验均值
            mu_AR = X_hat_AR[t, :]

            mu_post = np.linalg.solve(Lambda_post,
                                      self.W.T @ y_obs + self.Lambda_x @ mu_AR.T)

            #采样新的 X_t
            try:
                L = np.linalg.cholesky(Lambda_post)
                X_new = mu_post + np.random.randn(rank) @ L.T
            except np.linalg.LinAlgError:
                print("警告: Cholesky分解失败，使用求逆。")
                Lambda_post_inv = np.linalg.inv(Lambda_post)
                X_new = np.random.multivariate_normal(mu_post, Lambda_post_inv)
            #更新self.X
            self.X[t, :] = X_new

        #更新超参数
        self.mu_x_hyper = np.mean(self.X, axis=0)
        self.Lambda_x = inv(np.cov(self.X.T) + 0.001 * np.eye(rank))

    def _sample_var_coefficient(self):
        """采样VAR系数A"""
        T, rank = self.X.shape
        d = len(self.time_lags)
        tmax = np.max(self.time_lags)

        Z_mat = self.X[tmax:, :]
        Q_mat = np.zeros((T - tmax, rank * d))
        for k in range(d):
            Q_mat[:, k * rank: (k + 1) * rank] = self.X[tmax - self.time_lags[k]: T - self.time_lags[k], :]

        var_Psi0 = np.eye(rank * d) + Q_mat.T @ Q_mat
        var_Psi = inv(var_Psi0)
        var_M = var_Psi @ Q_mat.T @ Z_mat

        var_S = np.eye(rank) + Z_mat.T @ Z_mat - var_M.T @ var_Psi0 @ var_M
        Sigma = invwishart.rvs(df=rank + T - tmax, scale=var_S)

        self.A = np.random.multivariate_normal(var_M.ravel(), np.kron(Sigma, var_Psi)).reshape(rank * d, rank)
        return Sigma

    def _sample_precision_tau(self):
        """采样精度参数tau"""
        mat_hat = self.W @ self.X.T
        var_alpha = 1e-6 + 0.5 * np.sum(self.ind)
        var_beta = 1e-6 + 0.5 * np.sum(((self.sparse_mat - mat_hat) ** 2) * self.ind)
        self.tau = np.random.gamma(var_alpha, 1 / var_beta)

    def gibbs_sampling(self):
        W_plus = np.zeros_like(self.W)
        X_plus = np.zeros_like(self.X)
        A_plus = np.zeros_like(self.A)
        mat_hat_plus = np.zeros_like(self.sparse_mat)

        pos_test = np.where((self.dense_mat != 0) & (self.ind == 0))
        dense_test = self.dense_mat[pos_test]

        for it in range(self.burn_iter + self.gibbs_iter):
            self._sample_factor_w()
            Sigma = self._sample_var_coefficient()
            self._sample_factor_x()
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
        tensor_hat[tensor_hat < 0] = 0

        return tensor_hat

    def compute_mape(self, var, var_hat):
        return np.sum(np.abs(var - var_hat) / (var + 1e-6)) / var.shape[0]

    def compute_rmse(self, var, var_hat):
        return np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])



def mvnrnd_pre(mu, Lambda):
    L = cholesky_lower(Lambda)
    return np.random.multivariate_normal(mu, inv(Lambda))

