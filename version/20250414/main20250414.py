"""
**************************************
*  @Author  ：   mag1cfox (modified by Claude)
*  @Time    ：   2025/4/14
*  @Project :   pj_gptp_simulation
*  @Description :   增加分域功能的gPTP时间同步误差仿真
*  @FileName:   main_with_domains.py
**************************************
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math

# 创建输出目录
os.makedirs("output_data", exist_ok=True)
os.makedirs("output_image", exist_ok=True)

# 来自论文的参数
PHY_JITTER_MAX = 8e-9  # 8 ns
CLOCK_GRANULARITY = 8e-9  # 8 ns
MAX_DRIFT_RATE = 10e-6  # 10 ppm
NR_ERROR = 0.1e-6  # 0.1 ppm
RESIDENCE_TIME_MAX = 1e-3  # 1 ms
PROPAGATION_DELAY = 62e-9  # 62 ns
SYNC_INTERVAL = 125.0e-3  # 125 ms
NUM_SAMPLES = 1000  # 每跳采样数
SAMPLE_TIME = 24  # 模拟24小时的网络运行


def calculate_time_error(hops, sync_interval=SYNC_INTERVAL):
    """
    基于论文中的公式计算时间同步误差。
    """
    # 基础误差来自时钟漂移（公式11）
    gm_drift = np.random.uniform(-MAX_DRIFT_RATE, MAX_DRIFT_RATE)
    node_drift = np.random.uniform(-MAX_DRIFT_RATE, MAX_DRIFT_RATE)
    basic_error = (node_drift - gm_drift) * sync_interval

    # 物理层抖动和时钟粒度导致的误差
    if hops > 1:
        # 时间戳误差
        timestamp_error = np.random.uniform(0, PHY_JITTER_MAX) + np.random.uniform(0, CLOCK_GRANULARITY)

        # 基于公式(16)-(23)的误差传播因子
        error_factor = np.random.uniform(0.5, 1.0)

        # 计算累积误差（随跳数增加而增长）
        accumulated_error = timestamp_error * error_factor * hops

        # 从图10可知 - 30跳后误差增长更快
        if hops > 30:
            accumulated_error *= 1 + (hops - 30) / 100
    else:
        accumulated_error = 0

    # 总误差
    total_error = basic_error + accumulated_error

    # 随机化符号（误差可正可负）
    if np.random.random() < 0.5:
        total_error = -total_error

    return total_error


def calculate_overlapping_domains_error(total_hops, hops_per_domain):
    """
    计算重叠域（时间感知域）模式下的时间同步误差。

    Args:
        total_hops: 网络总跳数
        hops_per_domain: 每个域内的跳数

    Returns:
        端到端的时间同步误差
    """
    # 如果总跳数小于等于每域跳数，直接作为单域处理
    if total_hops <= hops_per_domain:
        return calculate_time_error(total_hops)

    # 计算需要的完整域数量
    full_domains = total_hops // hops_per_domain

    # 计算最后一个域的剩余跳数
    remaining_hops = total_hops % hops_per_domain

    # 初始化累积误差为0
    accumulated_error = 0

    # 计算每个完整域的误差并累积
    for i in range(full_domains):
        # 当前域内的误差 - 只考虑域内的跳数
        domain_error = calculate_time_error(hops_per_domain)

        # 累积误差
        accumulated_error += domain_error

    # 如果有剩余跳数，计算最后一个不完整域的误差
    if remaining_hops > 0:
        last_domain_error = calculate_time_error(remaining_hops)
        accumulated_error += last_domain_error

    # 应用域间误差缩减因子
    domain_count = full_domains + (1 if remaining_hops > 0 else 0)
    if domain_count > 1:
        # 域间误差缩减因子 - 域越多，误差相对缩减越多
        reduction_factor = 1.0 / (1 + 0.15 * (domain_count - 1))
        accumulated_error *= reduction_factor

    return accumulated_error


# 生成100跳网络的时间误差数据序列
def generate_time_error_sequence(total_hops=100, samples=NUM_SAMPLES, duration_hours=SAMPLE_TIME,
                                domain_size=None):
    """
    生成一段时间内100跳网络的时间误差序列

    Args:
        total_hops: 网络跳数，默认100
        samples: 样本数量
        duration_hours: 模拟时长（小时）
        domain_size: 如果不为None，表示使用域划分，值为每个域的大小

    Returns:
        时间点列表和对应的误差值列表
    """
    # 计算每个样本之间的时间间隔（小时）
    time_interval = duration_hours / (samples - 1)

    # 生成时间点序列（小时）
    time_points = [i * time_interval for i in range(samples)]

    # 生成误差序列
    error_values = []
    for _ in range(samples):
        if domain_size is None:
            # 单域模式
            error = calculate_time_error(total_hops)
        else:
            # 多域模式
            error = calculate_overlapping_domains_error(total_hops, domain_size)

        error_values.append(error)

    return time_points, error_values


# 生成100跳网络中不同配置的误差序列数据
print("生成100跳网络的时间误差序列数据...")

# 单域配置
time_points, single_domain_errors = generate_time_error_sequence(total_hops=100)

# 多域配置
domain_configs = {
    "2 domains (50 hops each)": 50,
    "4 domains (25 hops each)": 25,
    "10 domains (10 hops each)": 10
}

multi_domain_errors = {}
for config_name, domain_size in domain_configs.items():
    print(f"生成 {config_name} 配置的误差序列...")
    _, errors = generate_time_error_sequence(total_hops=100, domain_size=domain_size)
    multi_domain_errors[config_name] = errors

# 将误差从秒转换为微秒
single_domain_errors_us = [e * 1e6 for e in single_domain_errors]
for config in multi_domain_errors:
    multi_domain_errors[config] = [e * 1e6 for e in multi_domain_errors[config]]

# 保存数据到CSV
df = pd.DataFrame({
    'Time(hours)': time_points,
    'Single Domain Error(μs)': single_domain_errors_us
})

for config in multi_domain_errors:
    df[f'{config} Error(μs)'] = multi_domain_errors[config]

df.to_csv("output_data/time_errors_100hops_comparison_v1.csv", index=False)

# =========== 图1: 100跳网络中不同域配置的时间误差折线图 ===========
plt.figure(figsize=(14, 8))

# 绘制单域误差
plt.plot(time_points, single_domain_errors_us, label='Single Domain', linewidth=2, color='#E41A1C')

# 绘制多域误差
colors = ['#377EB8', '#4DAF4A', '#984EA3']
for i, config_name in enumerate(domain_configs):
    plt.plot(time_points, multi_domain_errors[config_name],
             label=config_name, linewidth=2, color=colors[i])

# 添加1μs工业要求的水平线
plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='1μs Requirement')
plt.axhline(y=-1.0, color='black', linestyle='--', alpha=0.7)

# 配置图表外观
plt.grid(True)
plt.xlabel('Time (hours)', fontname='Times New Roman', fontsize=20, fontweight='bold')
plt.ylabel('Time Error (μs)', fontname='Times New Roman', fontsize=20, fontweight='bold')
plt.title('Node 100 Time Error vs. GM in 100-hop Network', fontname='Times New Roman', fontsize=22, fontweight='bold')
plt.legend(fontsize=16, frameon=True, shadow=False, edgecolor='black', prop={'family': 'Times New Roman'})
plt.xticks(fontname='Times New Roman', fontsize=18)
plt.yticks(fontname='Times New Roman', fontsize=18)
plt.tight_layout()
plt.savefig("output_image/time_error_sequence_100hops_v1.png", dpi=600)
plt.show()

# =========== 图2: 100跳网络中不同域配置的时间误差箱形图 ===========
plt.figure(figsize=(12, 8))

# 准备箱形图数据
boxplot_data = [single_domain_errors_us]
boxplot_labels = ['Single Domain']

for config_name in domain_configs:
    boxplot_data.append(multi_domain_errors[config_name])
    boxplot_labels.append(config_name)

# 绘制箱形图
box_colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']
box = plt.boxplot(boxplot_data, patch_artist=True, labels=boxplot_labels, showfliers=False)

# 设置箱形图颜色
for patch, color in zip(box['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# 添加1μs工业要求的水平线
plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7)
plt.axhline(y=-1.0, color='black', linestyle='--', alpha=0.7)

# 配置图表外观
plt.grid(True, axis='y')
plt.ylabel('Time Error (μs)', fontname='Times New Roman', fontsize=20, fontweight='bold')
plt.title('Time Error Distribution in 100-hop Network', fontname='Times New Roman', fontsize=22, fontweight='bold')
plt.xticks(fontname='Times New Roman', fontsize=18, rotation=15)
plt.yticks(fontname='Times New Roman', fontsize=18)
plt.tight_layout()
plt.savefig("output_image/time_error_boxplot_100hops_v1.png", dpi=600)
plt.show()

# =========== 图3: 各配置误差CDF对比 ===========
plt.figure(figsize=(12, 8))

# 单域（100跳）
sorted_errors = np.sort(single_domain_errors_us)
cumulative_prob = np.linspace(0, 1, len(sorted_errors))
plt.plot(sorted_errors, cumulative_prob, label='Single Domain', linewidth=2, color='#E41A1C')

# 多域方法
for i, config_name in enumerate(domain_configs):
    sorted_errors = np.sort(multi_domain_errors[config_name])
    cumulative_prob = np.linspace(0, 1, len(sorted_errors))
    plt.plot(sorted_errors, cumulative_prob, label=config_name, linewidth=2, color=colors[i])

# 添加1μs的垂直参考线
plt.axvline(x=1.0, color='black', linestyle='--', alpha=0.7)
plt.axvline(x=-1.0, color='black', linestyle='--', alpha=0.7)

# 配置图表外观
plt.grid(True)
plt.xlabel('Time Error (μs)', fontname='Times New Roman', fontsize=20, fontweight='bold')
plt.ylabel('CDF', fontname='Times New Roman', fontsize=20, fontweight='bold')
plt.title('Error CDF for 100-hop Network', fontname='Times New Roman', fontsize=22, fontweight='bold')
plt.legend(fontsize=16, frameon=True, shadow=False, edgecolor='black', prop={'family': 'Times New Roman'})
plt.xticks(fontname='Times New Roman', fontsize=18)
plt.yticks(fontname='Times New Roman', fontsize=18)
plt.tight_layout()
plt.savefig("output_image/time_error_cdf_100hops_v1.png", dpi=600)
plt.show()