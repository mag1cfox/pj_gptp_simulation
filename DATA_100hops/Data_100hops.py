"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/4/17 15:27
*  @Project :   pj_gptp_simulation
*  @Description :   仿真100跳的数据并保存结果
*  @FileName:   Data_100hops.py
**************************************
"""

"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/4/12 20:11
*  @Project :   pj_gptp_simulation
*  @Description :   从ieee8021as_simulation.py 拷贝来的版本
*  @FileName:   main.py
**************************************
""""""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/4/12 20:11
*  @Project :   pj_gptp_simulation
*  @Description :   从ieee8021as_simulation.py 拷贝来的版本
*  @FileName:   main.py
**************************************
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math

# Create the output directory
os.makedirs("output_data", exist_ok=True)
os.makedirs("output_image", exist_ok=True)

# Parameters from the paper
PHY_JITTER_MAX = 8e-9  # 8 ns
CLOCK_GRANULARITY = 8e-9  # 8 ns
MAX_DRIFT_RATE = 10e-6  # 10 ppm
NR_ERROR = 0.1e-6  # 0.1 ppm
RESIDENCE_TIME_MAX = 1e-3  # 1 ms
DRIFT_RATE_CHANGE = 1e-6  # 漂移率变化范围：±1 ppm/s
PROPAGATION_DELAY = 25e-9  # 62 ns
SYNC_INTERVAL = 31.25e-3  # 31.25 ms
NUM_SAMPLES = 1000  # Samples per hop


def calculate_time_error(hops, sync_interval=SYNC_INTERVAL):
    """
    Calculate time synchronization error based on the paper's equations.
    """
    # 初始漂移率
    gm_drift_initial = np.random.uniform(-MAX_DRIFT_RATE, MAX_DRIFT_RATE)
    node_drift_initial = np.random.uniform(-MAX_DRIFT_RATE, MAX_DRIFT_RATE)

    # 漂移率变化率 (ppm/s)
    gm_drift_change_rate = np.random.uniform(-DRIFT_RATE_CHANGE, DRIFT_RATE_CHANGE)
    node_drift_change_rate = np.random.uniform(-DRIFT_RATE_CHANGE, DRIFT_RATE_CHANGE)

    # 计算漂移率变化量 (变化率 * 同步间隔)
    gm_drift_change = gm_drift_change_rate * sync_interval  # 单位：秒
    node_drift_change = node_drift_change_rate * sync_interval

    # 更新后的漂移率
    gm_drift = gm_drift_initial + gm_drift_change
    node_drift = node_drift_initial + node_drift_change

    # 基本误差计算(使用更新后的漂移率)
    basic_error = (node_drift - gm_drift) * sync_interval

    # 添加 NR_ERROR 影响 (±0.1 ppm)
    nr_error_contribution = np.random.uniform(-NR_ERROR, NR_ERROR) * sync_interval

    # Errors due to PHY jitter and clock granularity
    if hops > 1:
        # Timestamp error
        timestamp_error = np.random.uniform(0, PHY_JITTER_MAX) + np.random.uniform(0, CLOCK_GRANULARITY)

        # Error propagation factor based on equations (16)-(23)
        error_factor = np.random.uniform(0.5, 1.0)

        # Calculate accumulated error (grows with hop count)
        accumulated_error = timestamp_error * error_factor * hops

        # From Figure 10 - error grows faster after ~30 hops
        if hops > 30:
            accumulated_error *= 1 + (hops - 30) / 100
    else:
        accumulated_error = 0

    # Total error (now including nr_error_contribution)
    total_error = basic_error + accumulated_error + nr_error_contribution

    # Randomize sign (error can be positive or negative)
    if np.random.random() < 0.5:
        total_error = -total_error

    return total_error


# Generate data for all hops (1-100)
all_data = {}
for h in range(1, 101):
    print(f"Simulating hop {h}...")
    hop_errors = [calculate_time_error(h) for _ in range(NUM_SAMPLES)]
    all_data[h] = hop_errors

# Save to CSV with each column representing one hop
df = pd.DataFrame({hop: all_data[hop] for hop in range(1, 101)})
df.to_csv("output_data/te_data_3125ms_v2.csv", index=False)

# 设置全局字体和字号
plt.rcParams.update({
    'font.size': 20,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.weight': 'bold',  # 设置字体加粗
    'axes.labelweight': 'bold',  # 坐标轴标签加粗
    'mathtext.fontset': 'stix'
})

# Target hops for plotting
target_hops = [1, 7, 10, 25, 50, 75, 100]
colors = {
    '1': '#003f5c',  # 深蓝色
    '2': '#58508d',  # 紫蓝色
    '3': '#bc5090',  # 紫红色
    '4': '#ff6361',  # 珊瑚红
    '5': '#ffa600',  # 橙黄色
    '6': '#2c9c4c',  # 翠绿色
    '7': '#5e6472'  # 石板灰
}

# Create plot of time error distributions with modified styling
plt.figure(figsize=(8, 6), facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')

for i, hop in enumerate(target_hops):
    # Extract data for this hop and convert to microseconds
    errors = np.array(all_data[hop])
    errors_us = errors * 1e6

    # Sort errors for CDF plot
    sorted_errors = np.sort(errors_us)
    cumulative_prob = np.linspace(0, 1, len(sorted_errors))

    plt.plot(sorted_errors, cumulative_prob,
             label=f'Hop {hop}',
             color=colors[str(i + 1)],
             linewidth=2)

# Configure plot appearance
plt.grid(True, linestyle='--', alpha=0.7, color='lightgray')  # 灰色虚线网格

# 设置坐标轴为黑色
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')

plt.xlabel('Time Error (μs)', fontname='Times New Roman', fontsize=20, fontweight='bold', color='black')
plt.ylabel('CDF', fontname='Times New Roman', fontsize=20, fontweight='bold', color='black')
plt.xlim(-3,3)
plt.legend(fontsize=20, frameon=True, shadow=False, edgecolor='black', prop={'family': 'Times New Roman'})

plt.xticks(fontname='Times New Roman', fontsize=20, color='black')
plt.yticks(fontname='Times New Roman', fontsize=20, color='black')

plt.tight_layout()
plt.savefig("output_image/time_error_cdf_3125ms_v3.png", dpi=600, facecolor='white')
plt.show()
