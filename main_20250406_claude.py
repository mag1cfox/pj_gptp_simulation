"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/4/6 21:11
*  @Project :   pj_gptp_simulation
*  @Description :   尝试poe的claude 读取论文得到代码
*  @FileName:   main_20250406_claude.py
**************************************
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 系统模型参数
NUM_HOPS = 100  # 跳数
SIM_TIME = 100  # 模拟时间(秒)
SYNC_INTERVAL = 0.03125  # 同步间隔(秒)
PDELAY_INTERVAL = 1.0  # 传播延迟测量间隔(秒)
CLOCK_GRANULARITY = 8e-9  # 时钟粒度(8纳秒)
MAX_PHY_JITTER = 8e-9  # 最大PHY抖动(8纳秒)
MAX_DRIFT_RATE = 10e-6  # 最大漂移率(10ppm)
MAX_DRIFT_CHANGE = 1e-6  # 最大漂移变化率(1ppm/s)
PROPAGATION_DELAY = 50e-9  # 链路传播延迟(50纳秒)
RESIDENCE_TIME = 1e-3  # 停留时间(1毫秒)
MAX_NR_ERROR = 0.1e-6  # 邻居率比最大误差(0.1ppm)


class TimeAwareSystem:
    def __init__(self, position, initial_drift_rate=0):
        self.position = position
        self.drift_rate = initial_drift_rate
        self.local_time = 0
        self.last_sync_time = 0
        self.last_correction = 0
        self.correction_field = 0
        self.rate_ratio = 1.0
        self.neighbor_rate_ratio = 1.0
        self.propagation_delay = 0
        self.time_deviations = []

    def update_clock(self, time_step):
        # 更新漂移率 (随机变化但保持限制)
        drift_change = np.random.uniform(-MAX_DRIFT_CHANGE * time_step,
                                         MAX_DRIFT_CHANGE * time_step)
        self.drift_rate = np.clip(self.drift_rate + drift_change,
                                  -MAX_DRIFT_RATE, MAX_DRIFT_RATE)

        # 更新本地时钟
        self.local_time += time_step * (1 + self.drift_rate)

    def measure_propagation_delay(self, previous_tas):
        """测量与前一个时间感知系统之间的传播延迟"""
        if previous_tas is None:
            return 0

        # 模拟传播延迟测量的时间戳
        t1 = self.local_time
        # 添加PHY抖动和传播延迟
        phy_jitter1 = np.random.uniform(0, MAX_PHY_JITTER)

        # 时间戳t2在前一个系统中
        t2 = previous_tas.local_time + phy_jitter1 + PROPAGATION_DELAY
        # 前一系统的处理时间
        t3 = t2 + RESIDENCE_TIME

        # 时间戳t4回到当前系统
        phy_jitter2 = np.random.uniform(0, MAX_PHY_JITTER)
        t4 = self.local_time + RESIDENCE_TIME + phy_jitter2 + PROPAGATION_DELAY

        # 计算邻居率比 (带误差)
        true_nr = (1 + previous_tas.drift_rate) / (1 + self.drift_rate)
        nr_error = np.random.uniform(-MAX_NR_ERROR, MAX_NR_ERROR)
        self.neighbor_rate_ratio = true_nr + nr_error

        # 计算传播延迟
        propagation_delay = 0.5 * ((t4 - t1) - self.neighbor_rate_ratio * (t3 - t2))

        # 加入时钟粒度效应 (四舍五入到最近的CLOCK_GRANULARITY)
        propagation_delay = np.round(propagation_delay / CLOCK_GRANULARITY) * CLOCK_GRANULARITY

        self.propagation_delay = propagation_delay
        return propagation_delay

    def receive_sync(self, previous_tas, grandmaster_time):
        """接收同步消息并更新同步信息"""
        if previous_tas is None:
            # 如果是grandmaster，不需要接收同步
            return

        # 接收同步消息时的时间戳
        phy_jitter = np.random.uniform(0, MAX_PHY_JITTER)
        receive_time = self.local_time + phy_jitter

        # 更新速率比
        self.rate_ratio = previous_tas.rate_ratio * self.neighbor_rate_ratio

        # 更新修正字段
        residence_time = previous_tas.last_sync_time - previous_tas.last_correction
        self.correction_field = (previous_tas.correction_field +
                                 self.propagation_delay +
                                 residence_time * previous_tas.rate_ratio)

        # 计算与grandmaster的偏差
        gm_time = previous_tas.last_correction + self.correction_field
        time_deviation = receive_time - gm_time
        self.time_deviations.append(time_deviation)

        # 更新同步时间
        self.last_correction = receive_time
        self.last_sync_time = self.local_time

    def send_sync(self):
        """发送同步消息"""
        # 记录发送同步消息的时间
        self.last_sync_time = self.local_time


def run_simulation():
    # 创建时间感知系统
    systems = []
    for i in range(NUM_HOPS + 1):  # GM + NUM_HOPS时间感知系统
        initial_drift = np.random.uniform(-MAX_DRIFT_RATE, MAX_DRIFT_RATE)
        systems.append(TimeAwareSystem(i, initial_drift))

    # 模拟时间步
    time_step = 0.001  # 1ms
    total_steps = int(SIM_TIME / time_step)

    # 初始化测量
    for i in range(1, NUM_HOPS + 1):
        systems[i].measure_propagation_delay(systems[i - 1])

    # 主模拟循环
    for step in tqdm(range(total_steps)):
        current_time = step * time_step

        # 更新所有时钟
        for system in systems:
            system.update_clock(time_step)

        # 检查是否需要同步
        if current_time % SYNC_INTERVAL < time_step:
            # 从grandmaster开始同步
            systems[0].send_sync()

            # 依次同步每个系统
            for i in range(1, NUM_HOPS + 1):
                systems[i].receive_sync(systems[i - 1], systems[0].local_time)
                systems[i].send_sync()

        # 检查是否需要测量传播延迟
        if current_time % PDELAY_INTERVAL < time_step:
            for i in range(1, NUM_HOPS + 1):
                systems[i].measure_propagation_delay(systems[i - 1])

    return systems


def analyze_results(systems):
    # 分析每个系统的同步精度
    precisions = []
    for i in range(1, NUM_HOPS + 1):
        deviations = np.abs(systems[i].time_deviations)
        max_deviation = np.max(deviations)
        precisions.append(max_deviation)

    # 绘制结果
    plt.figure(figsize=(12, 6))

    # 绘制每个系统的最大偏差
    plt.subplot(1, 2, 1)
    plt.plot(range(1, NUM_HOPS + 1), precisions)
    plt.xlabel('Time-Aware System Position')
    plt.ylabel('Maximum Time Deviation (s)')
    plt.title('Maximum Time Deviation vs. Position')
    plt.grid(True)

    # 绘制最后一个系统的时间偏差分布
    plt.subplot(1, 2, 2)
    plt.hist(systems[NUM_HOPS].time_deviations, bins=50)
    plt.xlabel('Time Deviation (s)')
    plt.ylabel('Frequency')
    plt.title(f'Time Deviation Distribution for System {NUM_HOPS}')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 计算1μs内同步的概率
    sync_prob = []
    thresholds = [1e-6, 1.5e-6, 2e-6]

    for threshold in thresholds:
        probs = []
        for i in range(1, NUM_HOPS + 1):
            deviations = np.abs(systems[i].time_deviations)
            in_sync = np.sum(deviations < threshold) / len(deviations)
            probs.append(in_sync)
        sync_prob.append(probs)

    # 绘制同步概率
    plt.figure(figsize=(10, 6))
    for i, threshold in enumerate(thresholds):
        plt.plot(range(1, NUM_HOPS + 1), sync_prob[i],
                 label=f'Precision < {threshold * 1e6} μs')

    plt.xlabel('Time-Aware System Position')
    plt.ylabel('Probability of Synchronization')
    plt.title('Probability of Synchronization vs. Position')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.show()

    # 报告最终结果
    print(f"System at 100 hops away from GM:")
    print(f"  Maximum time deviation: {precisions[-1] * 1e6:.3f} μs")
    print(f"  Probability of sync within 1 μs: {sync_prob[0][-1] * 100:.2f}%")
    print(f"  Probability of sync within 1.5 μs: {sync_prob[1][-1] * 100:.2f}%")
    print(f"  Probability of sync within 2 μs: {sync_prob[2][-1] * 100:.2f}%")

    # 找出保证1μs同步精度的最大跳数
    max_hops_1us = np.argmax(np.array(sync_prob[0]) < 0.95) + 1
    print(f"Maximum hops for 95% probability of 1 μs precision: {max_hops_1us}")


# 运行模拟
np.random.seed(42)
systems = run_simulation()
analyze_results(systems)