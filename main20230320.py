"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/3/20 10:30
*  @Project :   pj_gptp_simulation
*  @Description :   Description
*  @FileName:   main20230320.py
**************************************
"""
import random
import numpy as np
import matplotlib.pyplot as plt

# 常量配置
NUM_NODES = 4  # 网络节点数
SYNC_INTERVAL = 0.03125  # 同步间隔（秒）
PDELAY_INTERVAL = 1.0  # Pdelay测量间隔（秒）
PHY_JITTER = 8e-9  # PHY抖动（秒）
CLOCK_GRANULARITY = 8e-9  # 时钟粒度（秒）
MAX_RESIDENCE_TIME = 1e-3  # 最大驻留时间（秒）
CLOCK_DRIFT_RANGE = (-10e-6, 10e-6)  # 时钟漂移率范围（ppm）

# 节点类
class Node:
    def __init__(self, node_id):
        self.id = node_id
        self.clock = 0.0  # 本地时钟
        self.clock_drift = random.uniform(*CLOCK_DRIFT_RANGE)  # 时钟漂移率
        self.rate_ratio = 1.0  # 速率比
        self.correction_field = 0.0  # 校正字段
        self.neighbor_rate_ratio = 1.0  # 邻居速率比
        self.propagation_delay = 50e-9  # 传播延迟（秒）
        self.sync_locked = True  # 同步锁定模式

    def update_clock(self, elapsed_time):
        """更新本地时钟，考虑漂移率"""
        self.clock += elapsed_time * (1 + self.clock_drift)

    def calculate_neighbor_rate_ratio(self, neighbor):
        """计算邻居速率比"""
        self.neighbor_rate_ratio = (1 + neighbor.clock_drift) / (1 + self.clock_drift)

    def update_correction_field(self, neighbor):
        """更新校正字段"""
        residence_time = random.uniform(0, MAX_RESIDENCE_TIME)
        self.correction_field = (
            neighbor.correction_field
            + neighbor.propagation_delay
            + residence_time * self.rate_ratio
        )

    def measure_propagation_delay(self, neighbor):
        """测量传播延迟（模拟Pdelay机制）"""
        t1 = self.clock
        t2 = neighbor.clock + random.uniform(-PHY_JITTER, PHY_JITTER)
        t3 = neighbor.clock + random.uniform(-PHY_JITTER, PHY_JITTER)
        t4 = self.clock + random.uniform(-PHY_JITTER, PHY_JITTER)
        self.propagation_delay = 0.5 * (
            (t4 - t1) - self.neighbor_rate_ratio * (t3 - t2)
        ) + random.uniform(-CLOCK_GRANULARITY, CLOCK_GRANULARITY)

    def synchronize(self, neighbor):
        """同步本地时钟"""
        self.calculate_neighbor_rate_ratio(neighbor)
        self.update_correction_field(neighbor)
        self.rate_ratio = neighbor.rate_ratio * self.neighbor_rate_ratio
        self.clock = (
            neighbor.clock
            + neighbor.correction_field
            + neighbor.propagation_delay
        )

# 仿真主函数
def simulate():
    # 初始化节点
    nodes = [Node(i) for i in range(NUM_NODES)]
    grandmaster = nodes[0]  # 第一个节点为Grandmaster

    # 仿真参数
    simulation_time = 100.0  # 仿真时间（秒）
    current_time = 0.0
    sync_times = []
    deviations = []

    # 仿真循环
    while current_time < simulation_time:
        # 更新所有节点的时钟
        for node in nodes:
            node.update_clock(SYNC_INTERVAL)

        # Grandmaster发送Sync消息
        for i in range(1, NUM_NODES):
            nodes[i].synchronize(nodes[i - 1])

        # 测量传播延迟（每隔PDELAY_INTERVAL秒）
        if current_time % PDELAY_INTERVAL == 0:
            for i in range(1, NUM_NODES):
                nodes[i].measure_propagation_delay(nodes[i - 1])

        # 记录末端节点与Grandmaster的偏差
        deviation = nodes[-1].clock - grandmaster.clock
        sync_times.append(current_time)
        deviations.append(deviation)

        # 更新时间
        current_time += SYNC_INTERVAL

    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.plot(sync_times, deviations, label="Deviation from Grandmaster")
    # plt.axhline(y=1e-6, color="r", linestyle="--", label="1μs Threshold")
    # plt.axhline(y=2e-6, color="g", linestyle="--", label="2μs Threshold")
    plt.xlabel("Time (s)")
    plt.ylabel("Clock Deviation (us)")
    plt.title("IEEE 802.1AS Synchronization Precision in a 100-Hop Network")
    plt.legend()
    plt.grid(True)
    plt.show()

# 运行仿真
if __name__ == "__main__":
    simulate()
