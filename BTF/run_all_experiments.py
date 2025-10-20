import time
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from BTMF import BTMF


# ==============================================================================
# --- 1. 参数管理模块---
# ==============================================================================
CONFIG = {
    # --- 路径管理 ---
    "paths": {
        "data_root": "./datasets",
        "output_dir": "results"
    },

    # --- 实验通用设置 ---
    "seed": 1000,
    "burn_iter": 50,
    "gibbs_iter": 10,

    # --- 缺失场景配置 ---
    "missing_scenarios": [
        {"rate": 0.4, "type": "RM", "name": "Random Missing (40%)"},
        {"rate": 0.6, "type": "RM", "name": "Random Missing (60%)"},
        {"rate": 0.4, "type": "NM", "name": "Non-random Missing (40%)"}
    ],

    # --- 数据集特定配置 ---
    "datasets": {
        "Guangzhou Speed Data": {
            # 现在路径是动态构建的
            "relative_path": "Guangzhou-data-set/tensor.mat",
            "rank_rm": 20, "rank_nm": 10,
            "time_lags": np.array([1, 2, 144]),
            "init_scale_rm": 0.1, "init_scale_nm": 0.01
        },
        "Birmingham Parking Data": {
            "relative_path": "Birmingham-data-set/tensor.mat",
            "rank_rm": 20, "rank_nm": 20,
            "time_lags": np.array([1, 2, 18]),
            "init_scale_rm": 0.1, "init_scale_nm": 0.01
        },
        "Hangzhou Flow Data": {
            "relative_path": "Hangzhou-data-set/tensor.mat",
            "rank_rm": 30, "rank_nm": 20,
            "time_lags": np.array([1, 2, 108]),
            "init_scale_rm": 0.1, "init_scale_nm": 0.01,
            "option": "pca"
        },
        "Seattle Speed Data": {
            "relative_path": "Seattle-data-set/tensor.npz",
            "rank_rm": 30, "rank_nm": 10,
            "time_lags": np.array([1, 2, 288]),
            "init_scale_rm": 0.1, "init_scale_nm": 0.01
        }
    },

    # --- 绘图和输出设置 ---
    "plotting": {
        "save_plots": True,
        "show_plots": True,
        "summary_file": "experiment_results_summary.txt"
    }
}

# ==============================================================================
# --- 动态路径构建和环境准备 ---
# ==============================================================================
import os

# 获取根路径
DATA_ROOT = CONFIG["paths"]["data_root"]
OUTPUT_DIR = CONFIG["paths"]["output_dir"]

# 创建输出目录
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 为每个数据集构建完整的绝对路径
# 并创建一个统一的加载器
for name, config in CONFIG["datasets"].items():
    full_path = os.path.join(DATA_ROOT, config["relative_path"])
    config["path"] = full_path

    # 根据文件扩展名自动选择加载器
    if full_path.endswith('.mat'):
        config["loader"] = lambda p: scipy.io.loadmat(p)['tensor']
    elif full_path.endswith('.npz'):
        config["loader"] = lambda p: np.load(p)['arr_0']
    else:
        raise ValueError(f"不支持的文件格式: {full_path}")

# 从CONFIG中读取其他参数
np.random.seed(CONFIG["seed"])
BURN_ITER = CONFIG["burn_iter"]
GIBBS_ITER = CONFIG["gibbs_iter"]
MISSING_SCENARIOS = CONFIG["missing_scenarios"]
DATASET_CONFIGS = CONFIG["datasets"]
PLOT_CONFIG = CONFIG["plotting"]
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)
PLOT_CONFIG["plot_dir"] = PLOT_DIR
PLOT_CONFIG["summary_file"] = os.path.join(OUTPUT_DIR, PLOT_CONFIG["summary_file"])


# ==============================================================================
# --- 2. 核心实验函数---
# ==============================================================================
def run_single_experiment(dense_tensor, rank, time_lags, missing_rate, missing_type, init_scale, option="factor"):
    dim = dense_tensor.shape
    if missing_type == "RM":
        mask = np.random.rand(dim[0], dim[1], dim[2]) + 0.5 - missing_rate
        sparse_tensor = dense_tensor * np.round(mask)
    elif missing_type == "NM":
        mask = np.random.rand(dim[0], dim[1])[:, :, np.newaxis] + 0.5 - missing_rate
        sparse_tensor = dense_tensor * np.round(mask)
    else:
        raise ValueError("Invalid missing type")
    init = {"W": init_scale * np.random.randn(dim[0], rank),
            "X": init_scale * np.random.randn(dim[1] * dim[2], rank)}
    model = BTMF(dense_tensor, sparse_tensor, time_lags, rank, burn_iter=BURN_ITER, gibbs_iter=GIBBS_ITER)
    start_time = time.time()
    tensor_hat = model.gibbs_sampling()
    end_time = time.time()
    pos_test = np.where((dense_tensor != 0) & (sparse_tensor == 0))
    mape = model.compute_mape(dense_tensor[pos_test], tensor_hat[pos_test])
    rmse = model.compute_rmse(dense_tensor[pos_test], tensor_hat[pos_test])
    return mape, rmse, end_time - start_time


# ==============================================================================
# --- 3. 绘图函数 ---
# ==============================================================================
def plot_results(df_results):
    if not PLOT_CONFIG["save_plots"] and not PLOT_CONFIG["show_plots"]:
        return
    df_results['MAPE'] = df_results['MAPE'].astype(float)
    df_results['RMSE'] = df_results['RMSE'].astype(float)
    if not os.path.exists(PLOT_CONFIG["plot_dir"]):
        os.makedirs(PLOT_CONFIG["plot_dir"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    fig.suptitle('BTMF Performance Across Datasets and Scenarios', fontsize=16)
    df_results.pivot(index='Dataset', columns='Scenario', values='MAPE').plot(kind='bar', ax=ax1)
    ax1.set_ylabel('MAPE')
    ax1.set_title('Mean Absolute Percentage Error (MAPE)')
    ax1.legend(title='Scenario')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    df_results.pivot(index='Dataset', columns='Scenario', values='RMSE').plot(kind='bar', ax=ax2)
    ax2.set_ylabel('RMSE')
    ax2.set_title('Root Mean Squared Error (RMSE)')
    ax2.legend(title='Scenario')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Dataset')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if PLOT_CONFIG["save_plots"]:
        plt.savefig(f'{PLOT_CONFIG["plot_dir"]}/performance_comparison_bar.png')
        print(f"\n图表已保存: {PLOT_CONFIG['plot_dir']}/performance_comparison_bar.png")
    if PLOT_CONFIG["show_plots"]:
        plt.show()
    else:
        plt.close()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('BTMF Performance Distribution on Each Dataset', fontsize=16)
    df_results.boxplot(column='MAPE', by='Dataset', ax=ax1, grid=False)
    ax1.set_title('MAPE Distribution')
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('MAPE')
    df_results.boxplot(column='RMSE', by='Dataset', ax=ax2, grid=False)
    ax2.set_title('RMSE Distribution')
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('RMSE')
    plt.suptitle('')
    plt.tight_layout()
    if PLOT_CONFIG["save_plots"]:
        plt.savefig(f'{PLOT_CONFIG["plot_dir"]}/performance_distribution_boxplot.png')
        print(f"图表已保存: {PLOT_CONFIG['plot_dir']}/performance_distribution_boxplot.png")
    if PLOT_CONFIG["show_plots"]:
        plt.show()
    else:
        plt.close()


# ==============================================================================
# --- 4. 主函数---
# ==============================================================================
if __name__ == "__main__":
    all_results = []
    print("========================================")
    print("      开始执行所有BTMF对比实验")
    print("========================================\n")
    for data_name, config in DATASET_CONFIGS.items():
        print(f"--- 正在处理数据集: {data_name} ---")
        try:
            dense_tensor = config["loader"](config["path"])
        except FileNotFoundError:
            print(f"[警告] 数据文件未找到: {config['path']}，跳过此数据集。\n")
            continue
        for scenario in MISSING_SCENARIOS:
            print(f"  > 场景: {scenario['name']}")
            if scenario['type'] == 'RM':
                rank = config['rank_rm']
                init_scale = config['init_scale_rm']
            else:
                rank = config['rank_nm']
                init_scale = config['init_scale_nm']
            option = config.get("option", "factor")
            mape, rmse, running_time = run_single_experiment(
                dense_tensor, rank, config['time_lags'],
                scenario['rate'], scenario['type'], init_scale, option
            )
            result = {
                "Dataset": data_name,
                "Scenario": scenario['name'],
                "Rank": rank,
                "MAPE": f"{mape:.6f}",
                "RMSE": f"{rmse:.6f}",
                "Time (s)": f"{running_time:.2f}"
            }
            all_results.append(result)
            print(f"    - MAPE: {result['MAPE']}, RMSE: {result['RMSE']}, 运行时间: {result['Time (s)']}秒")
        print("-" * (len(data_name) + 20) + "\n")
    print("========================================")
    print("           所有实验完成！")
    print("========================================\n")
    df_results = pd.DataFrame(all_results)
    results_str = df_results.to_string(index=False)
    with open(PLOT_CONFIG["summary_file"], "w") as f:
        f.write("BTMF Model Evaluation Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(results_str)
    print(f"结果汇总表已保存到: {PLOT_CONFIG['summary_file']}")
    print("\n结果汇总表:")
    print(results_str)
    plot_results(df_results)
