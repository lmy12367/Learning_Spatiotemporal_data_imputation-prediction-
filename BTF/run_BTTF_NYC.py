from BTTF import BTTF
import numpy as np
import scipy.io
import pandas as pd
import time
import os
from datetime import datetime


def generate_missing_pattern(dense_tensor, missing_type, missing_rate):
    """
    生成不同类型的缺失模式

    参数:
    - dense_tensor: 原始张量
    - missing_type: 'RM'(随机缺失), 'NM'(非随机缺失), 'FM'(纤维缺失)
    - missing_rate: 缺失率

    返回:
    - sparse_tensor: 包含缺失的张量
    """
    dim1, dim2, dim3 = dense_tensor.shape
    sparse_tensor = dense_tensor.copy()

    if missing_type == "RM":
        # 随机缺失
        mask = np.random.rand(dim1, dim2, dim3) < missing_rate
        sparse_tensor[mask] = np.nan

    elif missing_type == "NM":
        # 非随机缺失 - 按天连续缺失
        print(f"  生成非随机缺失模式 (连续缺失天数: {missing_rate * 100:.0f}%)")
        nm_tensor = np.random.rand(dim1, dim2, dim3 // 24)
        binary_tensor = np.zeros((dim1, dim2, dim3))

        for i1 in range(dim1):
            for i2 in range(dim2):
                for i3 in range(nm_tensor.shape[2]):
                    # 决定这一天是否完全缺失
                    if nm_tensor[i1, i2, i3] < missing_rate:
                        binary_tensor[i1, i2, i3 * 24:(i3 + 1) * 24] = 0
                    else:
                        binary_tensor[i1, i2, i3 * 24:(i3 + 1) * 24] = 1

        sparse_tensor[binary_tensor == 0] = np.nan

    elif missing_type == "FM":
        # 纤维缺失 - 整个时间维度缺失
        print(f"  生成纤维缺失模式 (缺失位置: {missing_rate * 100:.0f}%)")
        binary = np.random.rand(dim1, dim2) < missing_rate
        binary_tensor = binary[:, :, np.newaxis] * np.ones((dim1, dim2, dim3))
        sparse_tensor[binary_tensor == 0] = np.nan

    else:
        raise ValueError(f"未知的缺失类型: {missing_type}")

    # 统计缺失信息
    total_missing = np.sum(np.isnan(sparse_tensor))
    total_elements = sparse_tensor.size
    actual_missing_rate = total_missing / total_elements

    print(f"  实际缺失率: {actual_missing_rate:.2%} ({total_missing}/{total_elements})")

    return sparse_tensor


def evaluate_model(model, dense_tensor, sparse_tensor, tensor_hat):
    """
    评估模型性能

    参数:
    - model: BTTF模型实例
    - dense_tensor: 原始完整张量
    - sparse_tensor: 包含缺失的张量
    - tensor_hat: 预测张量

    返回:
    - metrics: 包含各种评估指标的字典
    """
    # 找到测试位置：原始数据不为0且被模型填充的位置
    pos_test = np.where((np.isnan(sparse_tensor)) & (dense_tensor != 0))

    if len(pos_test[0]) == 0:
        print("  警告: 没有找到有效的测试位置")
        return {
            'mape': np.inf,
            'rmse': np.inf,
            'mae': np.inf,
            'correlation': 0,
            'r2': 0,
            'test_points': 0
        }

    # 提取测试数据
    y_true = dense_tensor[pos_test]
    y_pred = tensor_hat[pos_test]

    # 确保y_true和y_pred是1D数组
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # 过滤掉无效值
    valid_mask = (y_true != 0) & ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        print("  警告: 过滤后没有有效的测试点")
        return {
            'mape': np.inf,
            'rmse': np.inf,
            'mae': np.inf,
            'correlation': 0,
            'r2': 0,
            'test_points': 0
        }

    # 计算各种指标
    mape = model.compute_mape(y_true, y_pred)
    rmse = model.compute_rmse(y_true, y_pred)
    mae = model.compute_mae(y_true, y_pred)

    # 计算相关系数
    if len(y_true) > 1:
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        if np.isnan(correlation):
            correlation = 0
    else:
        correlation = 0

    # 计算R²
    if len(y_true) > 1:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        if np.isnan(r2):
            r2 = 0
    else:
        r2 = 0

    metrics = {
        'mape': mape,
        'rmse': rmse,
        'mae': mae,
        'correlation': correlation,
        'r2': r2,
        'test_points': len(y_true)
    }

    return metrics


def run_nyc_experiments_optimized(data_path="./datasets/NYC-data-set/tensor.mat",
                                  output_dir="results",
                                  rank=30,
                                  burn_iter=100,
                                  gibbs_iter=20,
                                  scenarios=None,
                                  time_lags=np.array([1, 2, 24]),
                                  seed=1000):
    """
    运行优化的NYC数据集实验

    参数:
    - data_path: 数据文件路径
    - output_dir: 输出目录
    - rank: 张量分解的秩
    - burn_iter: burn-in迭代次数
    - gibbs_iter: Gibbs采样迭代次数
    - scenarios: 实验场景列表
    - time_lags: 时间滞后
    - seed: 随机种子

    返回:
    - results_df: 结果DataFrame
    """

    # 设置随机种子
    np.random.seed(seed)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 默认场景
    if scenarios is None:
        scenarios = [
            ("RM", 0.4, "Random Missing"),
            ("RM", 0.6, "Random Missing"),
            ("NM", 0.4, "Non-random Missing"),
            ("FM", 0.2, "Fiber Missing")
        ]

    print("=" * 60)
    print("NYC Speed Data - BTTF Experiments")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据路径: {data_path}")
    print(f"输出目录: {output_dir}")
    print(f"参数设置: rank={rank}, burn_iter={burn_iter}, gibbs_iter={gibbs_iter}")
    print(f"时间滞后: {time_lags}")
    print("-" * 60)

    # 加载数据
    try:
        print("\n📂 加载数据...")
        dense_tensor = scipy.io.loadmat(data_path)['tensor'].astype(np.float32)
        print(f"✅ 数据加载成功")
        print(f"   数据形状: {dense_tensor.shape}")
        print(f"   数据范围: [{np.min(dense_tensor):.2f}, {np.max(dense_tensor):.2f}]")
        print(f"   非零值比例: {np.sum(dense_tensor != 0) / dense_tensor.size:.2%}")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None

    # 存储结果
    all_results = []
    detailed_results = []

    # 运行每个场景
    for i, (mtype, mrate, mname) in enumerate(scenarios, 1):
        print(f"\n🔬 场景 {i}/{len(scenarios)}: {mname} ({mrate * 100:.0f}%)")
        print("-" * 40)

        try:
            # 生成缺失模式
            sparse_tensor = generate_missing_pattern(dense_tensor, mtype, mrate)

            # 初始化模型
            print("  🚀 初始化BTTF模型...")
            model = BTTF(dense_tensor, sparse_tensor, time_lags, rank, burn_iter, gibbs_iter)

            # 运行采样
            print("  ⏳ 开始Gibbs采样...")
            start_time = time.time()
            tensor_hat = model.gibbs_sampling()
            end_time = time.time()
            running_time = end_time - start_time

            # 评估模型
            print("  📊 评估模型性能...")
            metrics = evaluate_model(model, dense_tensor, sparse_tensor, tensor_hat)

            # 记录结果
            result = {
                'Scenario': f"{mname} ({mrate * 100:.0f}%)",
                'Type': mtype,
                'Rate': mrate,
                'Rank': rank,
                'MAPE': metrics['mape'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'Correlation': metrics['correlation'],
                'R2': metrics['r2'],
                'Test_Points': metrics['test_points'],
                'Time': running_time
            }
            all_results.append(result)

            # 详细结果
            detailed_result = result.copy()
            detailed_result.update({
                'Timestamp': datetime.now().isoformat(),
                'Data_Shape': dense_tensor.shape,
                'Burn_Iter': burn_iter,
                'Gibbs_Iter': gibbs_iter,
                'Time_Lags': list(time_lags)
            })
            detailed_results.append(detailed_result)

            # 打印结果
            print(f"\n  📈 结果摘要:")
            print(f"     MAPE: {metrics['mape']:.4f}")
            print(f"     RMSE: {metrics['rmse']:.4f}")
            print(f"     MAE: {metrics['mae']:.4f}")
            print(f"     相关系数: {metrics['correlation']:.4f}")
            print(f"     R²: {metrics['r2']:.4f}")
            print(f"     测试点数: {metrics['test_points']}")
            print(f"     运行时间: {running_time:.2f}秒")

            # 保存单个场景的结果
            scenario_file = os.path.join(output_dir, f"NYC_{mtype}_{int(mrate * 100)}_results.csv")
            scenario_df = pd.DataFrame([result])
            scenario_df.to_csv(scenario_file, index=False)
            print(f"  💾 场景结果已保存: {scenario_file}")

        except Exception as e:
            print(f"❌ 场景 {mname} 运行失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 保存所有结果
    if all_results:
        # 简洁结果
        results_df = pd.DataFrame(all_results)
        summary_file = os.path.join(output_dir, "NYC_summary_results.csv")
        results_df.to_csv(summary_file, index=False)

        # 详细结果
        detailed_df = pd.DataFrame(detailed_results)
        detailed_file = os.path.join(output_dir, "NYC_detailed_results.csv")
        detailed_df.to_csv(detailed_file, index=False)

        # 打印汇总
        print("\n" + "=" * 60)
        print("🎉 所有实验完成！")
        print("=" * 60)
        print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总运行时间: {sum(r['Time'] for r in all_results):.2f}秒")
        print(f"结果文件:")
        print(f"  - 汇总结果: {summary_file}")
        print(f"  - 详细结果: {detailed_file}")

        # 性能统计
        print("\n📊 性能统计:")
        print(f"  平均MAPE: {results_df['MAPE'].mean():.4f}")
        print(f"  平均RMSE: {results_df['RMSE'].mean():.4f}")
        print(f"  最佳MAPE场景: {results_df.loc[results_df['MAPE'].idxmin(), 'Scenario']}")
        print(f"  最佳RMSE场景: {results_df.loc[results_df['RMSE'].idxmin(), 'Scenario']}")

        return results_df
    else:
        print("\n❌ 没有成功完成的实验")
        return None


# 快速运行函数
def quick_nyc_test():
    """快速测试NYC实验"""
    return run_nyc_experiments_optimized(
        burn_iter=50,
        gibbs_iter=10,
        rank=20,
        scenarios=[("RM", 0.4, "Random Missing")]  # 只测试一个场景
    )


# 超快速测试函数（用于调试）
def debug_nyc_test():
    """超快速测试NYC实验（用于调试）"""
    return run_nyc_experiments_optimized(
        burn_iter=10,
        gibbs_iter=5,
        rank=10,
        scenarios=[("RM", 0.2, "Random Missing")]  # 只测试一个场景，低缺失率
    )


# 主函数
if __name__ == "__main__":
    # 运行完整实验
    results = run_nyc_experiments_optimized()

    # 或者运行快速测试
    # results = quick_nyc_test()

    # 或者运行调试测试
    # results = debug_nyc_test()
