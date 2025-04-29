import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import allantools
import os

# 1. 加载CSV文件
file_path = r"D:\06_engineering\03_analysis\pj_gptp_simulation\version\20250429\output_data\data_simulation_1-7hops.csv"
try:
    df = pd.read_csv(file_path)
    print("数据加载成功！前5行示例：")
    print(df.head())
except Exception as e:
    print(f"文件加载失败: {e}")
    exit()


# 2. 数据预处理函数
def preprocess_data(hop_data):
    time_errors = hop_data.values
    fs = 1  # 默认采样率1Hz

    if len(time_errors) < 10:
        raise ValueError("数据点不足（至少需要10个点）")
    if np.all(np.isnan(time_errors)):
        raise ValueError("时间误差数据全为NaN")

    time_errors = np.nan_to_num(time_errors, nan=np.nanmean(time_errors))
    return time_errors, fs


# 3. 创建输出目录
output_data_dir =  "output_data"
output_image_dir =  "output_image"
os.makedirs(output_data_dir, exist_ok=True)
os.makedirs(output_image_dir, exist_ok=True)
print(f"数据将保存到: {output_data_dir}")
print(f"图片将保存到: {output_image_dir}")

# 4. 准备存储所有TDEV结果的DataFrame
all_tdev_results = pd.DataFrame()
hop_columns = [f"Hop_{i}" for i in range(1, 8)]  # 列名列表

# 5. 创建组合图
plt.figure(figsize=(14, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(hop_columns)))  # 不同颜色

# 6. 遍历每一跳
for idx, hop_col in enumerate(hop_columns):
    try:
        # 处理数据
        hop_data = df[hop_col]
        time_errors, fs = preprocess_data(hop_data)
        print(f"\n处理 {hop_col}: 有效数据点={len(time_errors)}")

        # 计算TDEV
        taus, tdev, _, _ = allantools.tdev(
            data=time_errors,
            rate=fs,
            data_type="phase",
            taus="decade"
        )

        # 存储结果
        all_tdev_results[f"tau_{hop_col}"] = taus
        all_tdev_results[f"tdev_{hop_col}"] = tdev

        # 绘制到组合图中
        plt.loglog(taus, tdev, 'o-', color=colors[idx],
                   linewidth=2, markersize=6, label=hop_col)

    except Exception as e:
        print(f"处理 {hop_col} 时出错: {e}")
        continue

# 7. 保存所有TDEV结果到CSV
output_csv_path = os.path.join(output_data_dir, "all_tdev_results.csv")
all_tdev_results.to_csv(output_csv_path, index=False)
print(f"\n所有TDEV数据已保存到: {output_csv_path}")

# 8. 完善并保存组合图
plt.xlabel('Averaging Time $\\tau$ (s)', fontsize=12)
plt.ylabel('Time Deviation (s)', fontsize=12)
plt.title('Combined TDEV Analysis (All Hops)', fontsize=14)
plt.grid(True, which="both", linestyle='--', alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在右侧

plt.tight_layout()

# 保存组合图
output_plot_path = os.path.join(output_image_dir, "combined_tdev_plot.png")
plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')  # bbox_inches确保图例完整保存
plt.close()
print(f"组合TDEV图已保存到: {output_plot_path}")

print("\n所有处理完成！")
