"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/4/23 9:31
*  @Project :   pj_gptp_simulation
*  @Description :   Description
*  @FileName:   main_test20250423.py
**************************************
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
import random
import time

# 确保输出目录存在
os.makedirs("output_data_v2", exist_ok=True)
os.makedirs("output_image_v2", exist_ok=True)


class Clock:
    """时钟类，模拟节点的物理时钟"""

    def __init__(self):
        """初始化时钟"""
        self.time = 0
        # 初始漂移率 [-10, 10] ppm
        self.drift_rate = np.random.uniform(-10, 10)
        # 漂移率变化率 [0, 1] ppm/s
        self.drift_rate_change = np.random.uniform(0, 1)
        # 时钟粒度 8 ns
        self.granularity = 8e-9

    def update(self, elapsed_time):
        """
        更新时钟，考虑漂移率和粒度

        参数:
            elapsed_time: 物理时间流逝，单位秒
        返回:
            更新后的时间
        """
        # 更新漂移率 (考虑随时间的变化)
        drift_change = np.random.normal(0, self.drift_rate_change * elapsed_time)
        self.drift_rate += drift_change

        # 计算时间前进量 (考虑漂移率)
        # 漂移率单位是ppm，所以需要乘以1e-6转换
        time_advance = elapsed_time * (1 + self.drift_rate * 1e-6)

        # 考虑时钟粒度的影响 (时间只能以时钟粒度为单位增加)
        ticks = np.round(time_advance / self.granularity)
        time_increase = ticks * self.granularity

        self.time += time_increase
        return self.time

    def get_time(self):
        """获取当前时钟时间"""
        return self.time

    def adjust_time(self, offset):
        """
        调整时钟时间

        参数:
            offset: 时间偏移量，正值表示向前调整，负值表示向后调整
        """
        # 考虑时钟粒度的影响
        ticks = np.round(offset / self.granularity)
        actual_offset = ticks * self.granularity
        self.time += actual_offset


class TimeAwareSystem:
    """时间感知系统，实现IEEE 802.1AS的功能"""

    def __init__(self, node_id, is_gm=False):
        """
        初始化时间感知系统

        参数:
            node_id: 节点ID
            is_gm: 是否为主时钟(Grand Master)
        """
        self.node_id = node_id
        self.is_gm = is_gm
        self.clock = Clock()
        self.sync_locked = True  # 同步锁定模式
        self.parent = None
        self.children = []

        # IEEE 802.1AS 参数
        self.sync_interval = 31.25e-3  # 同步间隔 31.25 ms
        self.pdelay_interval = 1.0  # 传播延迟测量间隔 1 s
        self.max_freq_error = 0.1e-6  # 邻居频率比率误差上限 ±0.1 ppm

        # 时间同步状态
        self.last_sync_time = 0
        self.last_pdelay_time = 0
        self.propagation_delay = 62e-9  # 初始传播延迟估计值

        # 添加correction field属性
        self.correction_field = 0
        self.rate_ratio = 1.0  # 当前节点的频率比例 (相对于GM)
        self.neighbor_rate_ratio = 1.0  # 邻居频率比例

        # 接收和发送时间戳
        self.receive_timestamp = 0
        self.send_timestamp = 0

        # 性能指标
        self.time_errors = []
        self.time_stamps = []

    def add_child(self, child_node):
        """添加子节点"""
        self.children.append(child_node)
        child_node.parent = self

    def get_time(self):
        """获取当前节点的时间"""
        return self.clock.get_time()

    def update(self, elapsed_real_time, real_time, gm_time):
        """
        更新节点状态

        参数:
            elapsed_real_time: 物理时间流逝
            real_time: 当前物理时间
            gm_time: 主时钟当前时间
        """
        # 更新物理时钟
        self.clock.update(elapsed_real_time)

        # 记录与主时钟的误差
        error = self.get_time() - gm_time
        self.time_errors.append(error)
        self.time_stamps.append(real_time)

        # 如果是主时钟，无需同步
        if self.is_gm:
            return

        # 检查是否需要执行同步
        if real_time - self.last_sync_time >= self.sync_interval:
            # 执行同步
            self.synchronize()
            self.last_sync_time = real_time

        # 检查是否需要测量传播延迟
        if real_time - self.last_pdelay_time >= self.pdelay_interval:
            self.measure_propagation_delay()
            self.last_pdelay_time = real_time

    def synchronize(self):
        """执行时间同步过程"""
        if not self.parent or not self.sync_locked:
            return

        # 获取父节点时间和父节点的同步信息
        parent_time = self.parent.get_time()

        # 记录接收时间戳
        self.receive_timestamp = self.get_time()

        # 计算邻居频率比率 (允许±0.1 ppm的误差)
        error = np.random.uniform(-0.1e-6, 0.1e-6)
        self.neighbor_rate_ratio = (1 + self.parent.clock.drift_rate * 1e-6) / (1 + self.clock.drift_rate * 1e-6) + error

        # 更新频率比率
        self.rate_ratio = self.parent.rate_ratio * self.neighbor_rate_ratio

        # 模拟处理时间
        processing_delay = np.random.uniform(0, 1e-3)

        # 计算发送时间戳
        self.send_timestamp = self.receive_timestamp + processing_delay

        # 计算驻留时间
        residence_time = self.send_timestamp - self.receive_timestamp

        # 更新correction field - 遵循IEEE 802.1AS公式
        self.correction_field = self.parent.correction_field + self.propagation_delay + (residence_time * self.rate_ratio)

        # 考虑PHY抖动的影响 (0-8 ns均匀分布)
        phy_jitter = np.random.uniform(0, 8e-9)

        # 计算正确的时间 - 按照论文中的公式
        corrected_time = parent_time + self.parent.correction_field + self.propagation_delay + phy_jitter

        # 计算时间调整量
        current_time = self.get_time()
        time_adjustment = corrected_time - current_time

        # 应用时间调整
        self.clock.adjust_time(time_adjustment)

    def measure_propagation_delay(self):
        """测量链路传播延迟"""
        if not self.parent:
            return

        # 实际传播延迟为50ns
        actual_delay = 50e-9

        # PHY抖动 (0-8 ns均匀分布)
        phy_jitter_1 = np.random.uniform(0, 8e-9)
        phy_jitter_2 = np.random.uniform(0, 8e-9)

        # 传播延迟变化 (±3 ns)
        delay_variation = np.random.uniform(-3e-9, 3e-9)

        # 最终测量的传播延迟 (考虑双向抖动和变化)
        measured_delay = actual_delay + (phy_jitter_1 + phy_jitter_2) / 2 + delay_variation

        # 由于时钟粒度的影响，延迟会被量化
        measured_delay = np.round(measured_delay / self.clock.granularity) * self.clock.granularity

        # 更新传播延迟估计
        self.propagation_delay = measured_delay


class NetworkSimulation:
    """网络仿真类，实现整个网络拓扑和仿真逻辑"""

    def __init__(self, num_hops=100):
        """
        初始化网络仿真

        参数:
            num_hops: 网络跳数，默认为100
        """
        self.num_hops = num_hops
        self.nodes = []
        self.real_time = 0
        self.simulation_time = 100  # 仿真时长 100 秒
        self.time_step = 0.001  # 时间步长 1 ms

        # 创建网络拓扑
        self.create_topology()

    def create_topology(self):
        """创建线性链状拓扑"""
        # 创建主时钟节点 (GM)
        gm_node = TimeAwareSystem(0, is_gm=True)
        self.nodes.append(gm_node)

        # 创建其他节点，形成链状拓扑
        for i in range(1, self.num_hops + 1):
            node = TimeAwareSystem(i)
            self.nodes.append(node)

            # 建立父子关系 (线性链状)
            self.nodes[i - 1].add_child(node)

    def run_simulation(self):
        """运行一次完整的仿真"""
        # 重置仿真状态
        self.real_time = 0

        # 清空所有节点的历史数据
        for node in self.nodes:
            node.time_errors = []
            node.time_stamps = []

        # 运行仿真
        steps = int(self.simulation_time / self.time_step)
        for _ in tqdm(range(steps), desc="仿真进度"):
            # 首先获取GM时间
            gm_time = self.nodes[0].get_time()

            # 更新所有节点
            for node in self.nodes:
                node.update(self.time_step, self.real_time, gm_time)

            # 前进仿真时间
            self.real_time += self.time_step

    def collect_results(self):
        """收集仿真结果"""
        results = {}
        for node in self.nodes:
            results[node.node_id] = {
                'time_errors': node.time_errors,
                'time_stamps': node.time_stamps
            }
        return results


def run_single_simulation(num_hops=100):
    """
    运行单次仿真

    参数:
        num_hops: 网络跳数

    返回:
        results: 仿真结果
    """
    # 设置随机种子
    np.random.seed(42)
    random.seed(42)

    # 创建并运行一次仿真
    print(f"开始仿真 {num_hops}跳网络...")
    simulation = NetworkSimulation(num_hops)
    simulation.run_simulation()

    # 收集结果
    results = simulation.collect_results()

    # 保存所有跳数的结果到一个CSV文件
    save_all_results_to_csv(results)

    return results


def save_all_results_to_csv(results):
    """
    保存所有跳数的结果到一个CSV文件

    参数:
        results: 仿真结果，格式为{node_id: {'time_errors': [...], 'time_stamps': [...]}}
    """
    print("保存仿真结果...")

    # 创建一个包含所有时间点的数据框
    time_stamps = results[0]['time_stamps']  # 使用GM的时间戳

    # 创建一个字典，用于保存所有节点的误差数据
    data = {'time': time_stamps}

    # 添加每个节点的误差数据 - 修改后以秒为单位保存
    for node_id, node_data in results.items():
        # 不转换单位，直接使用秒
        data[f'hop_{node_id}_error_s'] = node_data['time_errors']

    # 创建DataFrame并保存，使用float_format参数启用科学计数法
    df = pd.DataFrame(data)
    df.to_csv("output_data_v2/all_nodes_error_v2.csv", index=False, float_format='%.8e')

    print(f"已保存仿真结果到 output_data_v2/all_nodes_error_v2.csv")


def analyze_results(results, hops_to_analyze=None):
    """
    分析仿真结果并生成图表

    参数:
        results: 仿真结果
        hops_to_analyze: 需要分析的跳数列表，如果为None则分析所有跳数
    """
    if hops_to_analyze is None:
        # 默认分析1, 10, 20, 50, 100跳(如果存在)
        max_hop = max(results.keys())
        hops_to_analyze = [1]
        if max_hop >= 10:
            hops_to_analyze.append(10)
        if max_hop >= 20:
            hops_to_analyze.append(20)
        if max_hop >= 50:
            hops_to_analyze.append(50)
        if max_hop >= 100:
            hops_to_analyze.append(100)

    # 确保我们只分析存在的跳数
    valid_hops = []
    for hop in hops_to_analyze:
        if hop in results:
            valid_hops.append(hop)
        else:
            print(f"警告: 跳数 {hop} 在结果中不存在，将被忽略")

    # 提取时间戳和每个跳数的误差数据
    time_stamps = results[0]['time_stamps']  # 使用GM的时间戳
    error_data = {}

    for hop in valid_hops:
        # 将误差转换为微秒
        error_data[hop] = [e * 1e6 for e in results[hop]['time_errors']]

    # 绘制时间同步误差折线图
    plt.figure(figsize=(12, 8))

    for hop in valid_hops:
        plt.plot(time_stamps, error_data[hop], label=f'Hop {hop}')

    plt.xlabel('Time (s)')
    plt.ylabel('Synchronization Error (μs)')
    plt.title('Time Synchronization Error vs GM')
    plt.legend()
    plt.grid(True)
    plt.savefig('output_image_v2/time_sync_error_v2.png', dpi=300)

    # 绘制最大误差与跳数的关系
    plt.figure(figsize=(12, 8))

    # 获取所有可用的跳数
    all_hops = sorted(list(results.keys()))
    max_abs_errors = []
    min_errors = []
    max_errors = []

    for hop in all_hops:
        # 计算每个跳数的误差统计 (微秒)
        errors_us = [e * 1e6 for e in results[hop]['time_errors']]
        max_abs_error = np.max(np.abs(errors_us))
        min_error = np.min(errors_us)
        max_error = np.max(errors_us)

        max_abs_errors.append(max_abs_error)
        min_errors.append(min_error)
        max_errors.append(max_error)

    plt.plot(all_hops, max_abs_errors, 'o-', label='Max Absolute Error')
    plt.plot(all_hops, min_errors, 'v-', label='Minimum Error')
    plt.plot(all_hops, max_errors, '^-', label='Maximum Error')

    plt.xlabel('Hop Count')
    plt.ylabel('Error (μs)')
    plt.title('Synchronization Error vs Hop Count')
    plt.legend()
    plt.grid(True)
    plt.savefig('output_image_v2/error_vs_hop_v2.png', dpi=300)

    # 计算不同精度阈值下的同步概率
    thresholds = [0.2, 0.5, 0.75, 1.0, 1.5, 2.0]  # 微秒
    sync_probabilities = {hop: [] for hop in valid_hops}

    for hop in valid_hops:
        for threshold in thresholds:
            # 计算误差在阈值内的百分比
            errors_us = np.abs(error_data[hop])
            probability = np.mean(errors_us < threshold) * 100
            sync_probabilities[hop].append(probability)

    # 绘制同步概率图
    plt.figure(figsize=(12, 8))

    for hop in valid_hops:
        plt.plot(thresholds, sync_probabilities[hop], 'o-', label=f'Hop {hop}')

    plt.xlabel('Error Threshold (μs)')
    plt.ylabel('Synchronization Probability (%)')
    plt.title('Synchronization Probability vs Error Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig('output_image_v2/sync_probability_v2.png', dpi=300)


if __name__ == "__main__":
    # 运行单次仿真
    num_hops = 100

    print("开始运行仿真...")
    start_time = time.time()
    results = run_single_simulation(num_hops)
    end_time = time.time()
    print(f"仿真完成，耗时: {end_time - start_time:.2f}秒")

    # 分析结果
    analyze_results(results)

    print("分析完成，结果已保存到output_data_v2和output_image_v2文件夹")