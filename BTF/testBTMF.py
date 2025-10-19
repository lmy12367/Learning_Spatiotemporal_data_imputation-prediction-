import numpy as np
import time
from BTMF import BTMF


def create_simple_data():
    """创建一个最小的、可控的测试数据集"""
    print("--- 创建简单测试数据 ---")
    dim1, dim2, T, rank = 5, 4, 10, 2
    np.random.seed(0)

    # 创建真实的底层因子
    W_true = np.random.rand(dim1 * dim2, rank)
    X_true = np.random.rand(T, rank)

    # 生成完整的张量
    dense_tensor = (W_true @ X_true.T).reshape(dim1, dim2, T)

    # 创建一个随机缺失的稀疏张量
    sparse_tensor = dense_tensor.copy()
    mask = np.random.rand(dim1, dim2, T) < 0.3  # 30%的缺失率
    sparse_tensor[mask] = np.nan

    print(f"数据形状: {dense_tensor.shape}")
    print(f"真实因子W形状: {W_true.shape}, X形状: {X_true.shape}")
    print(f"缺失值数量: {np.sum(np.isnan(sparse_tensor))}")
    print("-" * 20 + "\n")

    return dense_tensor, sparse_tensor


if __name__ == "__main__":
    # 1. 创建数据
    dense_tensor, sparse_tensor = create_simple_data()

    # 2. 设置模型参数
    time_lags = np.array([1, 2])
    rank = 2
    burn_iter = 5  # 极少的迭代次数
    gibbs_iter = 5

    print("--- 开始BTMF测试 ---")
    try:
        # 3. 初始化并运行模型
        model = BTMF(dense_tensor, sparse_tensor, time_lags, rank, burn_iter, gibbs_iter)

        start_time = time.time()
        tensor_hat = model.gibbs_sampling()
        end_time = time.time()

        print(f"\n测试成功！总耗时: {end_time - start_time:.2f}秒")
        print(f"填充后的张量形状: {tensor_hat.shape}")

        # 4. 简单检查
        pos_nan = np.where(np.isnan(sparse_tensor))
        filled_values = tensor_hat[pos_nan]
        print(f"填充了 {len(filled_values)} 个缺失值，平均值为: {np.mean(filled_values):.4f}")

    except Exception as e:
        print(f"\n测试失败！捕获到错误: {e}")
        import traceback

        traceback.print_exc()

