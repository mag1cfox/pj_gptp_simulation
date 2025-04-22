"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/4/22 20:35
*  @Project :   pj_gptp_simulation
*  @Description :   Description
*  @FileName:   main_test20250422.py
**************************************
"""
"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/4/21 23:23
*  @Project :   pj_gptp_simulation
*  @Description :   Description
*  @FileName:   main_test20250421.py
**************************************
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os
import csv
import pandas as pd

# 常量定义
NUM_NODES = 101  # 总节点数 (1个GM + 99个桥接器 + 1个终端)
SIMULATION_TIME = 10  # 仿真总时间(秒)
SYNC_INTERVAL = 0.03125  # 同步间隔31.25ms
PDELAY_INTERVAL = 1.0  # 链路延迟测量间隔1s
CLOCK_GRANULARITY = 8e-9  # 时钟粒度8ns
PHY_JITTER_MAX = 8e-9  # 物理层抖动最大值8ns
MAX_DRIFT_RATE = 10e-6  # 最大漂移率10ppm
MAX_DRIFT_RATE_CHANGE = 0.1e-6  # 最大漂移率变化0.1ppm/s (改小提高稳定性)
LINK_DELAY = 50e-9  # 链路延迟50ns
RESIDENCE_TIME = 1e-3  # 最大驻留时间1ms

# 创建输出目录
os.makedirs('output_data', exist_ok=True)
os.makedirs('output_image', exist_ok=True)  # 创建图片保存目录


class Clock:
    """时钟模型"""

    def __init__(self, is_ideal=False, drift_rate=0, drift_rate_change=0):
        self.is_ideal = is_ideal
        self.drift_rate = 0 if is_ideal else drift_rate
        self.drift_rate_change = 0 if is_ideal else drift_rate_change
        self.time = 0
        self.granularity = CLOCK_GRANULARITY
        self.last_update_time = 0  # 上次更新的仿真时间
        self.last_drift_update_second = -1  # 上次更新漂移率的整数秒

    def update(self, elapsed_sim_time):
        """更新时钟，考虑漂移率变化和粒度"""
        if self.is_ideal:
            self.time = elapsed_sim_time
        else:
            # 检查是否过了一个新的整数秒
            current_second = int(elapsed_sim_time)
            if current_second > self.last_drift_update_second:
                # 每秒更新一次漂移率
                drift_change = np.random.uniform(-MAX_DRIFT_RATE_CHANGE, MAX_DRIFT_RATE_CHANGE)
                self.drift_rate += drift_change

                # 限制漂移率不超过最大值
                self.drift_rate = max(min(self.drift_rate, MAX_DRIFT_RATE), -MAX_DRIFT_RATE)

                # 更新上次漂移率更新时间
                self.last_drift_update_second = current_second

            # 计算此次更新的实际时间间隔
            delta_time = elapsed_sim_time - self.last_update_time

            # 带漂移的时钟更新
            real_elapsed = delta_time * (1 + self.drift_rate)
            self.time += real_elapsed

            # 考虑时钟粒度
            self.time = int(self.time / self.granularity) * self.granularity

            # 更新上次更新时间
            self.last_update_time = elapsed_sim_time

    def get_time(self):
        """获取当前时钟时间"""
        return self.time

    def adjust(self, offset, rate_ratio=None):
        """调整时钟偏移和频率"""
        if not self.is_ideal:
            # 调整时钟偏移
            self.time += offset

            # 添加频率调整功能 - 关键修改点
            if rate_ratio is not None and abs(rate_ratio) > 1e-10:
                # 根据速率比率调整漂移率
                new_drift = (1 / rate_ratio) - 1

                # 平滑调整，避免突变
                adjustment_weight = 0.3  # 调整权重
                self.drift_rate = (1 - adjustment_weight) * self.drift_rate + adjustment_weight * new_drift

                # 确保不超出最大漂移限制
                self.drift_rate = max(min(self.drift_rate, MAX_DRIFT_RATE), -MAX_DRIFT_RATE)


class TimeAwareNode:
    """时间感知节点基类"""

    def __init__(self, node_id, is_gm=False, drift_rate=0, drift_rate_change=0):
        self.node_id = node_id
        self.is_gm = is_gm
        self.clock = Clock(is_ideal=is_gm, drift_rate=drift_rate, drift_rate_change=drift_rate_change)

        # 链路延迟测量
        self.measured_link_delay = LINK_DELAY  # 初始假设的链路延迟
        self.neighbor_rate_ratio = 1.0
        self.delay_history = deque(maxlen=10)  # 存储最近10次测量的延迟值

        # 同步状态
        self.last_sync_receive_time = 0
        self.last_sync_send_time = 0
        self.correction_field = 0
        self.rate_ratio = 1.0

        # 统计数据
        self.clock_offsets = []
        self.true_offsets = []
        self.clock_times = []  # 记录时钟时间
        self.drift_rates = []  # 记录漂移率

    def process_sync_message(self, origin_timestamp, correction_field, rate_ratio, sim_time):
        """处理同步消息和后续Follow_Up消息的信息"""
        self.last_sync_receive_time = self.clock.get_time()

        # 计算时钟偏移并调整
        offset = origin_timestamp + correction_field - self.last_sync_receive_time
        self.clock.adjust(offset, rate_ratio)  # 传递速率比率进行频率调整

        # 更新本地状态
        self.correction_field = correction_field
        self.rate_ratio = rate_ratio

        # 返回接收时间(本地时钟)
        return self.last_sync_receive_time

    def forward_sync_message(self, origin_timestamp, correction_field, rate_ratio, sim_time):
        """转发同步消息(仅桥节点需要实现)"""
        pass

    def measure_link_delay(self, sim_time):
        """模拟对等延迟测量过程"""
        # 简化模型：添加随机误差的链路延迟测量
        jitter1 = np.random.uniform(0, PHY_JITTER_MAX)
        jitter2 = np.random.uniform(0, PHY_JITTER_MAX)
        measured_delay = LINK_DELAY + jitter1 + jitter2 + CLOCK_GRANULARITY

        # 模拟对等延迟测量的随机误差
        error = np.random.normal(0, 3e-9)  # 3ns标准差
        current_measurement = measured_delay + error

        # 使用累积平均滤波提高稳定性
        self.delay_history.append(current_measurement)
        self.measured_link_delay = sum(self.delay_history) / len(self.delay_history)

        # 返回测量的链路延迟
        return self.measured_link_delay


class Grandmaster(TimeAwareNode):
    """GM节点模型"""

    def __init__(self, node_id):
        super().__init__(node_id, is_gm=True)

    def generate_sync_message(self, sim_time):
        """生成同步消息"""
        # 获取精确发送时间戳
        precise_origin_timestamp = self.clock.get_time()
        return precise_origin_timestamp, 0, 1.0  # 原始时间戳，修正域为0，速率比率为1


class Bridge(TimeAwareNode):
    """桥节点(交换机)模型"""

    def __init__(self, node_id, drift_rate, drift_rate_change=0):
        super().__init__(node_id, is_gm=False, drift_rate=drift_rate, drift_rate_change=drift_rate_change)

    def forward_sync_message(self, origin_timestamp, correction_field, rate_ratio, sim_time):
        """转发同步消息"""
        # 计算驻留时间(随机化，最大为RESIDENCE_TIME)
        residence_time = np.random.uniform(0, RESIDENCE_TIME)

        # 更新修正域，确保正确考虑链路延迟和驻留时间
        new_correction = correction_field + self.measured_link_delay + (residence_time * rate_ratio)

        # 更新速率比率，减小误差范围，提高稳定性
        neighbor_rate_error = np.random.uniform(-0.05e-6, 0.05e-6)  # 减小误差范围
        self.neighbor_rate_ratio = (1 + self.clock.drift_rate) / (1 + (self.clock.drift_rate - neighbor_rate_error))
        new_rate_ratio = rate_ratio * self.neighbor_rate_ratio

        # 更新发送时间
        self.last_sync_send_time = self.clock.get_time() + residence_time

        return origin_timestamp, new_correction, new_rate_ratio


class Simulator:
    """仿真器类"""

    def __init__(self):
        # 创建节点
        self.nodes = []

        # 创建GM节点
        self.nodes.append(Grandmaster(0))

        # 创建桥节点和终端节点，每个节点具有独立的漂移率
        for i in range(1, NUM_NODES):
            # 随机生成初始漂移率 (-10ppm to 10ppm)
            drift_rate = np.random.uniform(-MAX_DRIFT_RATE, MAX_DRIFT_RATE)

            # drift_rate_change参数不再使用，因为我们在Clock类中每秒更新漂移率
            # 使用0作为占位符
            node = Bridge(i, drift_rate, 0)
            self.nodes.append(node)

            print(f"节点 {i} - 初始漂移率: {drift_rate * 1e6:.3f} ppm")

        # 仿真时间
        self.sim_time = 0

        # 记录仿真时间点
        self.sim_time_points = []

    def run(self):
        """运行仿真"""
        # 时间步进仿真
        time_step = min(SYNC_INTERVAL, PDELAY_INTERVAL) / 10  # 选择一个合适的时间步长

        next_sync_time = 0
        next_pdelay_time = 0

        while self.sim_time < SIMULATION_TIME:
            # 更新所有节点的时钟
            for node in self.nodes:
                node.clock.update(self.sim_time)

            # 处理同步消息
            if self.sim_time >= next_sync_time:
                # GM生成同步消息
                origin_timestamp, correction, rate_ratio = self.nodes[0].generate_sync_message(self.sim_time)

                # 逐跳传递同步消息
                for i in range(1, NUM_NODES):
                    # 添加物理层抖动
                    jitter = np.random.uniform(0, PHY_JITTER_MAX)

                    # 处理同步消息
                    self.nodes[i].process_sync_message(origin_timestamp, correction, rate_ratio, self.sim_time)

                    # 如果不是最后一个节点，则继续转发
                    if i < NUM_NODES - 1:
                        origin_timestamp, correction, rate_ratio = self.nodes[i].forward_sync_message(
                            origin_timestamp, correction, rate_ratio, self.sim_time)

                next_sync_time = self.sim_time + SYNC_INTERVAL

            # 处理链路延迟测量
            if self.sim_time >= next_pdelay_time:
                for i in range(1, NUM_NODES):
                    self.nodes[i].measure_link_delay(self.sim_time)
                next_pdelay_time = self.sim_time + PDELAY_INTERVAL

            # 记录统计数据
            if self.sim_time % (SYNC_INTERVAL / 2) < time_step:  # 每半个同步周期记录一次数据
                self.sim_time_points.append(self.sim_time)
                gm_time = self.nodes[0].clock.get_time()

                for i in range(NUM_NODES):
                    local_time = self.nodes[i].clock.get_time()
                    self.nodes[i].clock_times.append(local_time)

                    # 记录当前漂移率
                    if not self.nodes[i].is_gm:
                        self.nodes[i].drift_rates.append(self.nodes[i].clock.drift_rate)
                    else:
                        self.nodes[i].drift_rates.append(0)

                    if i > 0:  # 非GM节点
                        offset = local_time - gm_time
                        self.nodes[i].clock_offsets.append(offset)

                        # 记录相对于理想时间的真实偏差
                        true_offset = local_time - self.sim_time
                        self.nodes[i].true_offsets.append(true_offset)

            # 推进仿真时间
            self.sim_time += time_step

    def save_data_to_csv(self):
        """保存数据到CSV文件"""
        # 保存时钟时间
        clock_times_data = {}
        clock_times_data['sim_time'] = [round(t * 1e6, 3) for t in self.sim_time_points]  # 转换为微秒并保留3位小数

        for i in range(NUM_NODES):
            # 转换为微秒并保留3位小数
            clock_times_data[f'node_{i}'] = [round(t * 1e6, 3) for t in self.nodes[i].clock_times]

        # 创建DataFrame并保存
        df_times = pd.DataFrame(clock_times_data)
        df_times.to_csv('output_data/clock_times_v2.csv', index=False)

        # 保存时钟偏差 (TE - Time Error)
        offsets_data = {}
        offsets_data['sim_time'] = [round(t * 1e6, 3) for t in self.sim_time_points]  # 转换为微秒并保留3位小数

        for i in range(1, NUM_NODES):
            # 转换为微秒并保留3位小数
            offsets_data[f'node_{i}'] = [round(offset * 1e6, 3) for offset in self.nodes[i].clock_offsets]

        # 创建DataFrame并保存
        df_offsets = pd.DataFrame(offsets_data)
        df_offsets.to_csv('output_data/time_errors_v2.csv', index=False)

        # 保存真实偏差
        true_offsets_data = {}
        true_offsets_data['sim_time'] = [round(t * 1e6, 3) for t in self.sim_time_points]  # 转换为微秒并保留3位小数

        for i in range(1, NUM_NODES):
            # 转换为微秒并保留3位小数
            true_offsets_data[f'node_{i}'] = [round(offset * 1e6, 3) for offset in self.nodes[i].true_offsets]

        # 创建DataFrame并保存
        df_true = pd.DataFrame(true_offsets_data)
        df_true.to_csv('output_data/true_offsets_v2.csv', index=False)

        # 保存漂移率数据
        drift_rates_data = {}
        drift_rates_data['sim_time'] = [round(t * 1e6, 3) for t in self.sim_time_points]  # 转换为微秒并保留3位小数

        for i in range(NUM_NODES):
            # 转换为ppm并保留3位小数
            drift_rates_data[f'node_{i}'] = [round(rate * 1e6, 3) for rate in self.nodes[i].drift_rates]

        # 创建DataFrame并保存
        df_drift = pd.DataFrame(drift_rates_data)
        df_drift.to_csv('output_data/drift_rates_v2.csv', index=False)

        print("数据已保存到output_data文件夹中")

    def analyze_results(self):
        """分析仿真结果"""
        # 计算每个节点的同步精度
        max_offsets = []
        avg_offsets = []

        for i in range(1, NUM_NODES):
            # 跳过前2秒的数据，等待系统稳定
            stable_offsets = [abs(offset) for t_idx, offset in enumerate(self.nodes[i].clock_offsets)
                              if self.sim_time_points[t_idx] > 2.0]

            if stable_offsets:
                max_offsets.append(max(stable_offsets))
                avg_offsets.append(sum(stable_offsets) / len(stable_offsets))
            else:
                max_offsets.append(0)
                avg_offsets.append(0)

        # 保存同步精度结果
        precision_data = {
            'node_id': list(range(1, NUM_NODES)),
            'max_offset_us': [round(o * 1e6, 3) for o in max_offsets],  # 转换为微秒并保留3位小数
            'avg_offset_us': [round(o * 1e6, 3) for o in avg_offsets]  # 转换为微秒并保留3位小数
        }

        df_precision = pd.DataFrame(precision_data)
        df_precision.to_csv('output_data/sync_precision_v2.csv', index=False)

        return max_offsets, avg_offsets

    def plot_results(self):
        """绘制仿真结果"""
        # 绘制TE(时间误差)折线图 - 选择几个代表性节点
        plt.figure(figsize=(12, 6))

        # 选择几个代表性节点绘制
        nodes_to_plot = [1, 10, 50, 100]

        for i in nodes_to_plot:
            times = self.sim_time_points
            # 转换为微秒并保留3位小数
            offsets = [round(o * 1e6, 3) for o in self.nodes[i].clock_offsets]
            plt.plot(times, offsets, label=f"Node {i}")

        plt.xlabel("Simulation Time (s)")
        plt.ylabel("Time Error (μs)")
        plt.title("Time Error (TE) Relative to GM")
        plt.legend()
        plt.grid(True)
        plt.savefig('output_image/time_errors_v2.png', dpi=300)

        # 绘制所有节点的TE热图
        plt.figure(figsize=(12, 8))

        te_data = np.zeros((NUM_NODES - 1, len(self.sim_time_points)))
        for i in range(1, NUM_NODES):
            # 转换为微秒
            te_data[i - 1, :] = [o * 1e6 for o in self.nodes[i].clock_offsets]

        im = plt.imshow(te_data, aspect='auto', cmap='coolwarm',
                        extent=[0, SIMULATION_TIME, NUM_NODES, 1],
                        vmin=-2.0, vmax=2.0)  # 限制色标范围，单位为微秒

        plt.colorbar(im, label="Time Error (μs)")
        plt.xlabel("Simulation Time (s)")
        plt.ylabel("Node ID")
        plt.title("Time Error (TE) Heatmap for All Nodes")
        plt.savefig('output_image/time_error_heatmap_v2.png', dpi=300)

        # 绘制同步精度与跳数的关系
        max_offsets, avg_offsets = self.analyze_results()

        plt.figure(figsize=(12, 6))
        # 转换为微秒
        plt.plot(range(1, NUM_NODES), [round(o * 1e6, 3) for o in max_offsets], 'r-', label="Max Offset")
        plt.plot(range(1, NUM_NODES), [round(o * 1e6, 3) for o in avg_offsets], 'b-', label="Average Offset")
        plt.xlabel("Node ID (Hop Count)")
        plt.ylabel("Clock Offset (μs)")
        plt.title("Synchronization Precision vs. Hop Count")
        plt.legend()
        plt.grid(True)
        plt.savefig('output_image/sync_precision_v2.png', dpi=300)

        # 绘制最后一个节点的偏差随时间的变化
        plt.figure(figsize=(12, 6))
        times = self.sim_time_points
        # 转换为微秒并保留3位小数
        offsets = [round(o * 1e6, 3) for o in self.nodes[-1].clock_offsets]
        plt.plot(times, offsets)
        plt.xlabel("Simulation Time (s)")
        plt.ylabel("Time Error (μs)")
        plt.title(f"Time Error of Last Node (Node {NUM_NODES - 1})")
        plt.grid(True)
        plt.savefig('output_image/last_node_offset_v2.png', dpi=300)

        # 绘制几个代表节点的漂移率变化
        plt.figure(figsize=(12, 6))
        for i in nodes_to_plot:
            times = self.sim_time_points
            drift_rates = [rate * 1e6 for rate in self.nodes[i].drift_rates]  # 转换为ppm
            plt.plot(times, drift_rates, label=f"Node {i}")

        plt.xlabel("Simulation Time (s)")
        plt.ylabel("Drift Rate (ppm)")
        plt.title("Drift Rate Changes Over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig('output_image/drift_rates_v2.png', dpi=300)

        plt.show()


# 运行仿真
sim = Simulator()
sim.run()
sim.save_data_to_csv()
sim.plot_results()