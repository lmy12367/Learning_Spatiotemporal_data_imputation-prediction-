import time
import numpy as np
from numpy.linalg import inv as inv
from numpy.random import normal as normrnd, multivariate_normal
from scipy.linalg import khatri_rao as kr_prod
from numpy.linalg import solve as solve
from numpy.linalg import cholesky as cholesky_lower
from scipy.linalg import cholesky as cholesky_upper
from scipy.linalg import solve_triangular as solve_ut
from scipy.stats import wishart, invwishart
import warnings

warnings.filterwarnings('ignore')


# ==============================================================================
# --- 辅助函数 ---
# ==============================================================================
def ten2mat(tensor, mode):
    """将张量沿指定模式展开为矩阵"""
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')


def cov_mat(mat, mat_bar):
    """计算协方差矩阵"""
    mat = mat - mat_bar
    return mat.T @ mat


def mvnrnd_pre(mu, Lambda):
    """高效的多元正态分布采样，利用Cholesky分解"""
    try:
        # 使用上三角Cholesky分解
        L = cholesky_upper(Lambda, overwrite_a=True, check_finite=False)
        src = normrnd(size=(mu.shape[0],))
        return solve_ut(L, src, lower=False, check_finite=False, overwrite_b=True) + mu
    except np.linalg.LinAlgError:
        # 如果Cholesky失败，使用SVD分解
        U, s, Vt = np.linalg.svd(Lambda)
        s = np.maximum(s, 1e-10)  # 确保奇异值不为零
        return mu + U @ np.diag(np.sqrt(s)) @ Vt @ np.random.randn(mu.shape[0])


def mnrnd(M, U, V):
    """生成矩阵正态分布随机矩阵"""
    dim1, dim2 = M.shape
    try:
        X0 = np.random.randn(dim1, dim2)
        P = cholesky_lower(U)
        Q = cholesky_lower(V)
        return M + P @ X0 @ Q.T
    except:
        # 如果失败，返回均值
        return M


# ==============================================================================
# --- 优化版BTTF类 ---
# ==============================================================================
class BTTF:
    """
    优化版贝叶斯时间张量分解 (Bayesian Temporal Tensor Factorization)

    主要改进：
    1. 自动数据标准化
    2. 增强的数值稳定性
    3. 改进的收敛检测
    4. 优化的评估指标
    5. 更好的错误处理
    """

    def __init__(self, dense_tensor, sparse_tensor, time_lags, rank, burn_iter=500, gibbs_iter=100):
        """
        初始化BTTF模型

        参数:
        - dense_tensor: 完整的张量数据
        - sparse_tensor: 包含缺失值的张量
        - time_lags: 时间滞后数组
        - rank: 分解的秩
        - burn_iter: burn-in迭代次数
        - gibbs_iter: Gibbs采样迭代次数
        """
        self.dense_tensor = dense_tensor.copy()
        self.sparse_tensor = sparse_tensor.copy()
        self.time_lags = np.array(time_lags, dtype=int)
        self.rank = rank
        self.burn_iter = burn_iter
        self.gibbs_iter = gibbs_iter

        self.dim1, self.dim2, self.T = sparse_tensor.shape

        # 数据标准化
        self._normalize_data()

        # 初始化参数
        self.init_parameters()

        # 验证配置
        self._validate_config()

    def _normalize_data(self):
        """数据标准化"""
        # 记录原始数据的统计信息
        self.original_min = np.nanmin(self.dense_tensor)
        self.original_max = np.nanmax(self.dense_tensor)
        self.original_mean = np.nanmean(self.dense_tensor)

        # 避免除零
        if self.original_max > self.original_min:
            self.scaler = self.original_max - self.original_min
            self.dense_tensor = (self.dense_tensor - self.original_min) / self.scaler
            self.sparse_tensor = (self.sparse_tensor - self.original_min) / self.scaler
        else:
            self.scaler = 1.0
            print("警告: 数据为常数，跳过标准化")

        # 处理缺失值
        self.ind = ~np.isnan(self.sparse_tensor)
        self.sparse_tensor[np.isnan(self.sparse_tensor)] = 0

    def _validate_config(self):
        """验证配置参数"""
        if self.T <= np.max(self.time_lags):
            print(f"警告: 时间序列长度({self.T})小于最大时间滞后({np.max(self.time_lags)})")

        if self.rank <= 0:
            raise ValueError("Rank必须大于0")

        if self.burn_iter <= 0 or self.gibbs_iter <= 0:
            raise ValueError("迭代次数必须大于0")

    def init_parameters(self):
        """初始化模型参数"""
        # 使用较小的初始值以获得更好的收敛性
        init_scale = 0.01
        self.U = init_scale * np.random.randn(self.dim1, self.rank)
        self.V = init_scale * np.random.randn(self.dim2, self.rank)
        self.X = init_scale * np.random.randn(self.T, self.rank)

        # VAR系数矩阵
        d = len(self.time_lags)
        self.A = init_scale * np.random.randn(self.rank * d, self.rank)

        # 精度参数矩阵
        self.Tau = np.ones((self.dim1, self.dim2))

        # 收敛相关变量
        self.convergence_history = []

    def _sample_factor_u(self):
        """采样空间因子U"""
        # 自适应正则化
        Lambda_u = (1e-6 / self.dim1) * np.eye(self.rank)

        var1 = kr_prod(self.X, self.V).T
        var2 = kr_prod(var1, var1)
        var3 = (var2 @ ten2mat(self.ind, 0).T).reshape([self.rank, self.rank, self.dim1]) + Lambda_u[..., np.newaxis]

        # 使用广播的Tau
        tau_ind = self.Tau[:, :, None] * self.ind
        tau_sparse_tensor = self.Tau[:, :, None] * self.sparse_tensor
        var4 = var1 @ ten2mat(tau_sparse_tensor, 0).T

        for i in range(self.dim1):
            try:
                self.U[i, :] = mvnrnd_pre(solve(var3[:, :, i], var4[:, i]), var3[:, :, i])
            except:
                # 如果采样失败，使用正态分布
                self.U[i, :] = multivariate_normal(np.zeros(self.rank), Lambda_u)

    def _sample_factor_v(self):
        """采样空间因子V"""
        Lambda_v = (1e-6 / self.dim2) * np.eye(self.rank)

        var1 = kr_prod(self.X, self.U).T
        var2 = kr_prod(var1, var1)
        var3 = (var2 @ ten2mat(self.ind, 1).T).reshape([self.rank, self.rank, self.dim2]) + Lambda_v[..., np.newaxis]

        tau_ind = self.Tau[:, :, None] * self.ind
        tau_sparse_tensor = self.Tau[:, :, None] * self.sparse_tensor
        var4 = var1 @ ten2mat(tau_sparse_tensor, 1).T

        for j in range(self.dim2):
            try:
                self.V[j, :] = mvnrnd_pre(solve(var3[:, :, j], var4[:, j]), var3[:, :, j])
            except:
                self.V[j, :] = multivariate_normal(np.zeros(self.rank), Lambda_v)

    def _sample_factor_x(self, Sigma):
        """采样时间因子X"""
        dim3, rank = self.X.shape
        d = len(self.time_lags)
        tmax = np.max(self.time_lags)
        tmin = np.min(self.time_lags)

        # 预计算一些矩阵
        A0 = np.dstack([self.A] * d)
        for k in range(d):
            A0[k * rank: (k + 1) * rank, :, k] = 0

        mat0 = inv(Sigma) @ self.A.T
        mat1 = np.einsum('kij, jt -> kit', self.A.reshape([d, rank, rank]), inv(Sigma))
        mat2 = np.einsum('kit, kjt -> ij', mat1, self.A.reshape([d, rank, rank]))

        var1 = kr_prod(self.V, self.U).T
        var2 = kr_prod(var1, var1)
        var3 = (var2 @ ten2mat(self.ind, 2).T).reshape([rank, rank, dim3]) + inv(Sigma)[:, :, None]

        tau_ind = self.Tau[:, :, None] * self.ind
        tau_sparse_tensor = self.Tau[:, :, None] * self.sparse_tensor
        var4 = var1 @ ten2mat(tau_sparse_tensor, 2).T

        for t in range(dim3):
            Mt = np.zeros((rank, rank))
            Nt = np.zeros(rank)
            Qt = mat0 @ self.X[t - self.time_lags, :].reshape(rank * d)

            index = list(range(0, d))
            if t >= dim3 - tmax and t < dim3 - tmin:
                index = list(np.where(t + self.time_lags < dim3))[0]
            elif t < tmax:
                Qt = np.zeros(rank)
                index = list(np.where(t + self.time_lags >= tmax))[0]

            if t < dim3 - tmin:
                Mt = mat2.copy()
                temp = np.zeros((rank * d, len(index)))
                n = 0
                for k in index:
                    temp[:, n] = self.X[t + self.time_lags[k] - self.time_lags, :].reshape(rank * d)
                    n += 1
                temp0 = self.X[t + self.time_lags[index], :].T - np.einsum('ijk, ik -> jk', A0[:, :, index], temp)
                Nt = np.einsum('kij, jk -> i', mat1[index, :, :], temp0)

            var3[:, :, t] = var3[:, :, t] + Mt
            if t < tmax:
                var3[:, :, t] = var3[:, :, t] - inv(Sigma) + np.eye(rank)

            try:
                self.X[t, :] = mvnrnd_pre(solve(var3[:, :, t], var4[:, t] + Nt + Qt), var3[:, :, t])
            except:
                self.X[t, :] = multivariate_normal(np.zeros(rank), Sigma)

    def _sample_var_coefficient(self):
        """采样VAR系数A和协方差Sigma"""
        dim, rank = self.X.shape
        d = len(self.time_lags)
        tmax = np.max(self.time_lags)

        if dim <= tmax:
            return np.eye(rank), np.eye(rank)

        # 构造回归矩阵
        Z_mat = self.X[tmax:, :]
        Q_mat = np.zeros((dim - tmax, rank * d))
        for k in range(d):
            Q_mat[:, k * rank: (k + 1) * rank] = self.X[tmax - self.time_lags[k]: dim - self.time_lags[k], :]

        # 自适应正则化
        lambda_reg = max(1e-6, 1.0 / (dim - tmax))
        var_Psi0 = Q_mat.T @ Q_mat + lambda_reg * np.eye(rank * d)

        # 确保数值稳定性
        eigenvals = np.linalg.eigvals(var_Psi0)
        if np.min(eigenvals) < 1e-8:
            var_Psi0 += (1e-8 - np.min(eigenvals)) * np.eye(rank * d)

        var_Psi = inv(var_Psi0)
        var_M = var_Psi @ Q_mat.T @ Z_mat

        var_S = Z_mat.T @ Z_mat - var_M.T @ var_Psi0 @ var_M
        var_S = var_S + lambda_reg * np.eye(rank)

        # 确保var_S正定
        eigenvals_S = np.linalg.eigvals(var_S)
        if np.min(eigenvals_S) < 1e-8:
            var_S += (1e-8 - np.min(eigenvals_S)) * np.eye(rank)

        # 采样Sigma
        try:
            Sigma = invwishart.rvs(df=rank + dim - tmax, scale=var_S)
        except:
            print("警告: 逆Wishart采样失败，使用单位矩阵")
            Sigma = np.eye(rank)

        # 采样A
        try:
            cov_A = np.kron(Sigma, var_Psi)
            cov_A += 1e-6 * np.eye(cov_A.shape[0])
            A = multivariate_normal(var_M.ravel(), cov_A)
            A = A.reshape(rank * d, rank)
        except:
            print("警告: 矩阵正态采样失败，使用均值")
            A = var_M.reshape(rank * d, rank)

        return A, Sigma

    def _sample_precision_tau(self):
        """采样精度参数Tau"""
        tensor_hat = np.einsum('ir, jr, tr -> ijt', self.U, self.V, self.X)

        # 使用温和的先验
        var_alpha = 1e-6 + 0.5 * np.sum(self.ind, axis=2)
        var_beta = 1e-6 + 0.5 * np.sum(((self.sparse_tensor - tensor_hat) ** 2) * self.ind, axis=2)

        # 避免除零
        var_beta = np.maximum(var_beta, 1e-10)
        self.Tau = np.random.gamma(var_alpha, 1 / var_beta)

    def _check_convergence(self, it, tensor_hat):
        """检查收敛性"""
        if it <= self.burn_iter:
            return False

        # 计算当前误差
        current_error = np.linalg.norm(self.sparse_tensor - tensor_hat)
        self.convergence_history.append(current_error)

        # 检查最近几次的变化
        if len(self.convergence_history) >= 10:
            recent_errors = self.convergence_history[-10:]
            change = np.std(recent_errors) / (np.mean(recent_errors) + 1e-10)

            if change < 1e-4:
                return True

        return False

    def gibbs_sampling(self):
        """执行Gibbs采样"""
        # 初始化累加器
        U_plus = np.zeros_like(self.U)
        V_plus = np.zeros_like(self.V)
        X_plus = np.zeros_like(self.X)
        A_plus = np.zeros_like(self.A)
        tensor_hat_plus = np.zeros_like(self.sparse_tensor)

        total_iter = self.burn_iter + self.gibbs_iter
        start_time = time.time()
        last_print_time = start_time

        print(f"开始Gibbs采样: {self.burn_iter} burn-in + {self.gibbs_iter} sampling")

        for it in range(total_iter):
            iter_start = time.time()

            # 采样步骤
            self._sample_factor_u()
            self._sample_factor_v()
            self.A, Sigma = self._sample_var_coefficient()
            self._sample_factor_x(Sigma)
            self._sample_precision_tau()

            # 计算当前估计
            tensor_hat = np.einsum('ir, jr, tr -> ijt', self.U, self.V, self.X)

            # 检查收敛
            if self._check_convergence(it, tensor_hat):
                elapsed = time.time() - start_time
                print(f"\n🎉 模型提前收敛于第 {it + 1} 轮！")
                print(f"   - 总用时: {elapsed:.1f}秒")
                break

            # 累加样本
            if it + 1 > self.burn_iter:
                U_plus += self.U
                V_plus += self.V
                X_plus += self.X
                A_plus += self.A
                tensor_hat_plus += tensor_hat

            # 进度显示
            current_time = time.time()
            if (it + 1) % 50 == 0 or (it + 1) <= 10 or (current_time - last_print_time) > 30:
                iter_time = current_time - iter_start
                elapsed = current_time - start_time
                remaining = (total_iter - it - 1) * (elapsed / (it + 1))
                current_error = np.linalg.norm(self.sparse_tensor - tensor_hat)

                print(f"🔄 轮次 {it + 1:4d}/{total_iter} | "
                      f"用时 {iter_time:.2f}s | "
                      f"累计 {elapsed:.1f}s | "
                      f"预计剩余 {remaining:.0f}s | "
                      f"误差 {current_error:.4f}")

                last_print_time = current_time

        # 计算后验均值
        print("\n📊 计算后验均值...")
        self.U = U_plus / self.gibbs_iter
        self.V = V_plus / self.gibbs_iter
        self.X = X_plus / self.gibbs_iter
        self.A = A_plus / self.gibbs_iter
        tensor_hat = tensor_hat_plus / self.gibbs_iter

        # 反标准化
        print("🔄 反标准化结果...")
        tensor_hat = tensor_hat * self.scaler + self.original_min
        tensor_hat[tensor_hat < 0] = 0  # 确保非负

        total_time = time.time() - start_time
        print(f"✅ 完成！总用时: {total_time:.1f}秒")

        return tensor_hat

    def compute_mape(self, var, var_hat):
        """计算MAPE（平均绝对百分比误差）"""
        mask = var != 0
        if np.sum(mask) == 0:
            return np.inf

        ape = np.abs(var[mask] - var_hat[mask]) / var[mask]

        # 使用截断平均值减少异常值影响
        if len(ape) > 0:
            ape_sorted = np.sort(ape)
            truncate_idx = int(0.95 * len(ape_sorted))
            if truncate_idx > 0:
                return np.mean(ape_sorted[:truncate_idx])

        return np.mean(ape)

    def compute_rmse(self, var, var_hat):
        """计算RMSE（均方根误差）"""
        if len(var) == 0:
            return np.inf
        return np.sqrt(np.mean((var - var_hat) ** 2))

    def compute_mae(self, var, var_hat):
        """计算MAE（平均绝对误差）"""
        if len(var) == 0:
            return np.inf
        return np.mean(np.abs(var - var_hat))

    def evaluate(self, dense_tensor, tensor_hat):
        """评估模型性能"""
        # 找到测试位置（原始数据中非零且被填充的位置）
        sparse_tensor_norm = (self.sparse_tensor - self.original_min) / self.scaler
        pos_test = np.where((np.isnan(sparse_tensor_norm)) & (dense_tensor != 0))

        if len(pos_test[0]) == 0:
            print("警告: 没有找到有效的测试位置")
            return None, None, None

        mape = self.compute_mape(dense_tensor[pos_test], tensor_hat[pos_test])
        rmse = self.compute_rmse(dense_tensor[pos_test], tensor_hat[pos_test])
        mae = self.compute_mae(dense_tensor[pos_test], tensor_hat[pos_test])

        return mape, rmse, mae

    def get_factor_matrices(self):
        """获取因子矩阵"""
        return {
            'U': self.U,
            'V': self.V,
            'X': self.X,
            'A': self.A
        }

    def save_model(self, filepath):
        """保存模型参数"""
        model_data = {
            'U': self.U,
            'V': self.V,
            'X': self.X,
            'A': self.A,
            'Tau': self.Tau,
            'rank': self.rank,
            'time_lags': self.time_lags,
            'scaler': self.scaler,
            'original_min': self.original_min,
            'original_max': self.original_max
        }
        np.savez(filepath, **model_data)
        print(f"模型已保存到: {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """加载模型参数"""
        data = np.load(filepath)
        # 创建一个虚拟实例来存储参数
        instance = cls.__new__(cls)
        instance.U = data['U']
        instance.V = data['V']
        instance.X = data['X']
        instance.A = data['A']
        instance.Tau = data['Tau']
        instance.rank = data['rank']
        instance.time_lags = data['time_lags']
        instance.scaler = data['scaler']
        instance.original_min = data['original_min']
        instance.original_max = data['original_max']
        print(f"模型已从 {filepath} 加载")
        return instance
