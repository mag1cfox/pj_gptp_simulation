"""
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
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
import matplotlib as mpl

# 设置全局字体为Times New Roman, 字号20，加粗
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 20
rcParams['font.weight'] = 'bold'
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'
rcParams['figure.titleweight'] = 'bold'
rcParams['axes.linewidth'] = 2
rcParams['lines.linewidth'] = 2

# 创建输出目录
if not os.path.exists('version/20250413/output_data'):
    os.makedirs('version/20250413/output_data')

# 参数设置
sync_intervals = [0.03125, 0.125, 1.0]  # 同步间隔(秒): 31.25ms, 125ms, 1s
interval_names = ["Sync Interval = 31_25ms", "Sync Interval = 125ms", "Sync Interval = 1s"]
max_hops = 100
simulation_time = 60  # 仿真时长(秒)
time_step = 0.001  # 时间步长(秒)

# 时钟参数
max_drift_rate = 10e-6  # 最大漂移率: 10 ppm
phy_jitter_max = 8e-9  # 最大PHY抖动: 8 ns
clock_granularity = 8e-9  # 时钟粒度: 8 ns
residence_time_max = 1e-3  # 最大留存时间: 1 ms
neighbor_rate_ratio_error = 0.1e-6  # 邻居速率比误差: 0.1 ppm

# 颜色定义
colors = {
    '1': '#054e97',  # 深蓝
    '7': '#70a3c4',  # 中蓝
    '10': '#c7e5ec',  # 浅蓝
    '25': '#f5b46f',  # 浅橙
    '50': '#df5b3f',  # 深红
    '75': '#4696a0',  # 蓝绿
    '100': '#e6785a'  # 珊瑚
}

class ClockNode:
    def __init__(self, node_id, initial_drift=None):
        self.node_id = node_id
        # 随机初始化时钟漂移率，范围为±max_drift_rate
        if initial_drift is None:
            self.drift_rate = np.random.uniform(-max_drift_rate, max_drift_rate)
        else:
            self.drift_rate = initial_drift
        self.local_time = 0  # 本地时钟时间
        self.offset = 0  # 与主时钟的偏移量
        self.last_sync_time = 0  # 上次同步的时间

    def update_time(self, real_time, time_step):
        # 更新本地时钟时间(考虑漂移率)
        delta_time = time_step * (1 + self.drift_rate)
        self.local_time += delta_time
        return self.local_time

    def get_time(self):
        # 获取当前修正后的时间
        return self.local_time - self.offset

    def sync_with_master(self, master_time, hop_count):
        # 模拟与主时钟的同步过程
        # 1. 测量与主时钟的偏移
        # 2. 考虑PHY抖动、时钟粒度和留存时间等累积误差

        # 模拟PHY抖动的影响(随机±phy_jitter_max, 累积hop_count次)
        phy_jitter = np.sum(np.random.uniform(-phy_jitter_max, phy_jitter_max, hop_count))

        # 模拟时钟粒度的影响(随机0到clock_granularity, 累积hop_count次)
        granularity_error = np.sum(np.random.uniform(0, clock_granularity, hop_count))

        # 模拟留存时间的影响(随机0到residence_time_max, 累积hop_count次)
        residence_time_error = np.sum(np.random.uniform(0, residence_time_max, hop_count))

        # 模拟邻居速率比的累积误差
        rate_ratio_cumulative_error = 0
        if hop_count > 1:
            for i in range(1, hop_count):
                # 错误累积是非线性的，与路径长度相关
                rate_ratio_cumulative_error += i * neighbor_rate_ratio_error * residence_time_max

        # 总累积误差
        cumulative_error = phy_jitter + granularity_error + rate_ratio_cumulative_error

        # 当前实际偏移量 = 本地时间 - 主时钟时间
        current_offset = self.local_time - master_time

        # 更新偏移量(考虑累积误差)
        self.offset = current_offset - cumulative_error

        # 记录同步时间
        self.last_sync_time = self.local_time

        # 返回同步后的时间误差(绝对值)
        return abs(self.get_time() - master_time)


def run_simulation(sync_interval, max_hops, sim_time, time_step):
    # 创建主时钟(grandmaster)
    grandmaster = ClockNode(0, initial_drift=0)  # 假设主时钟无漂移

    # 创建网络中的所有节点
    nodes = [ClockNode(i + 1) for i in range(max_hops)]

    # 时间序列和结果存储
    time_series = np.arange(0, sim_time, time_step)
    results = {}

    # 初始化结果字典，为每个hop保存时间误差序列
    for hop in range(1, max_hops + 1):
        results[hop] = []

    # 主循环:模拟时间进程
    for t in time_series:
        # 更新grandmaster时钟
        gm_time = grandmaster.update_time(t, time_step)

        # 对每个节点:
        for hop in range(1, max_hops + 1):
            node = nodes[hop - 1]

            # 更新节点时钟
            node.update_time(t, time_step)

            # 检查是否需要同步(基于同步间隔)
            if t - node.last_sync_time / node.drift_rate >= sync_interval:
                # 同步节点和记录同步误差
                sync_error = node.sync_with_master(gm_time, hop)
                results[hop].append((t, sync_error))

    return results


# 运行所有同步间隔的仿真并保存结果
for idx, sync_interval in enumerate(sync_intervals):
    print(f"Running simulation for sync interval: {sync_interval}s...")

    # 运行仿真
    simulation_results = run_simulation(sync_interval, max_hops, simulation_time, time_step)

    # 将结果保存到CSV文件
    for hop, error_data in simulation_results.items():
        if len(error_data) > 0:
            df = pd.DataFrame(error_data, columns=['Time', 'Error'])
            df.to_csv(f'version/20250413/output_data/sync_{interval_names[idx]}_hop_{hop}.csv', index=False)

    print(f"Simulation completed for {interval_names[idx]}")

# 可视化特定跳数的时间误差
hops_to_visualize = [1, 7, 10, 25, 50, 75, 100]

for idx, interval_name in enumerate(interval_names):
    plt.figure(figsize=(12, 8))

    for hop in hops_to_visualize:
        file_path = f'version/20250413/output_data/sync_{interval_name}_hop_{hop}.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # 将误差转换为微秒并保留3位小数
            df['Error_us'] = (df['Error'] * 1e6).round(3)
            plt.plot(df['Time'], df['Error_us'], label=f'Hop {hop}', color=colors[str(hop)])

    plt.xlabel('Time (s)')
    plt.ylabel('Time Error (μs)')
    plt.title(f'Time Synchronization Error - {interval_name}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'version/20250413/output_data/sync_error_{interval_name}.png', dpi=300)
    plt.close()

# 创建汇总图表(所有同步间隔的特定跳数)
for hop in hops_to_visualize:
    plt.figure(figsize=(12, 8))

    for idx, interval_name in enumerate(interval_names):
        file_path = f'version/20250413/output_data/sync_{interval_name}_hop_{hop}.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Error_us'] = (df['Error'] * 1e6).round(3)
            plt.plot(df['Time'], df['Error_us'],
                     label=f'{interval_name}',
                     color=list(colors.values())[idx % len(colors)])

    plt.xlabel('Time (s)')
    plt.ylabel('Time Error (μs)')
    plt.title(f'Time Synchronization Error - Hop {hop}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'version/20250413/output_data/sync_error_hop_{hop}.png', dpi=300)
    plt.close()

print("Visualization completed. Results are saved in the 'results' directory.")
