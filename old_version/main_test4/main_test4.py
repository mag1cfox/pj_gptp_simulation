"""
修正版 main_test4.py - 修复时钟同步间隔对误差影响的问题
"""
import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os
import csv

# 常量配置
NUM_NODES = 102  # 链式网络中的节点数
SYNC_INTERVAL = 0.03125  # 同步间隔 (31.25 ms)
# SYNC_INTERVAL = 1  # 同步间隔 (1s)
PHY_JITTER = 8e-9  # PHY抖动范围 (8 ns)
CLOCK_GRANULARITY = 8e-9  # 时钟粒度 (8 ns)
MAX_DRIFT_RATE = 10e-6  # 最大漂移率 (±10 ppm)
SIM_TIME = 60.0  # 仿真总时长 (秒)
PDELAY_INTERVAL = 1.0  # 传播延迟测量间隔 (1 s)
DRIFT_RATE_CHANGE = 1e-6  # 漂移率每秒变化范围 [0, 1] ppm/s

# 新增配置参数
CLOCK_UPDATE_INTERVAL = 0.001  # 时钟更新间隔 (1 ms)
MAX_ADJUSTMENT_PER_SYNC = 0.5  # 每次同步最大调整比例 (50%)


class Clock:
    def __init__(self):
        self.drift_rate = np.random.uniform(-MAX_DRIFT_RATE, MAX_DRIFT_RATE)
        self.offset = 0.0  # 相对于主时钟的偏移
        self.time = 0.0  # 本地时间
        self.last_update_time = 0.0  # 上次更新时间
        self.accumulated_error = 0.0  # 累积误差

    def update(self, current_sim_time):
        # 计算自上次更新后的时间间隔
        delta_t = current_sim_time - self.last_update_time

        # 随机小幅调整漂移率
        self.drift_rate += np.random.uniform(-DRIFT_RATE_CHANGE, DRIFT_RATE_CHANGE) * delta_t
        self.drift_rate = np.clip(self.drift_rate, -MAX_DRIFT_RATE, MAX_DRIFT_RATE)

        # 累积漂移误差
        drift_effect = delta_t * self.drift_rate
        self.accumulated_error += drift_effect

        # 更新本地时间
        self.time += delta_t * (1.0 + self.drift_rate)
        self.last_update_time = current_sim_time

        return self.time

    def adjust(self, error):
        # 限制单次调整量，更符合实际时钟行为
        adjustment = np.sign(error) * min(abs(error), abs(error * MAX_ADJUSTMENT_PER_SYNC))
        self.time -= adjustment

        # 记录剩余误差
        remaining_error = error - adjustment
        return remaining_error


class Node:
    def __init__(self, node_id):
        self.id = node_id
        self.clock = Clock()
        self.residence_time = 1e-3  # 驻留时间 (1 ms)
        self.propagation_delay = 50e-9  # 固定传播延迟 (50 ns)
        self.asymmetry = 0.0  # 链路非对称性 (默认对称)
        self.rate_ratio = 1.0  # 率比 (初始为1)
        self.neighbor_rate_ratio = 1.0  # 邻居率比 (初始为1)

        # 记录误差数据
        self.pre_sync_errors = []  # 同步前误差（微秒）
        self.post_sync_errors = []  # 同步后误差（微秒）

    def receive_sync(self, sync_time, correction_field, sim_time):
        # 接收Sync消息时添加PHY抖动和时钟粒度
        actual_receive_time = sync_time + self.propagation_delay + \
                              np.random.uniform(0, PHY_JITTER) + \
                              np.random.uniform(0, CLOCK_GRANULARITY)

        # 计算本地时间与主时钟的偏差
        local_time = self.clock.time
        gm_time = sync_time + correction_field
        error = local_time - gm_time

        # 记录同步前的误差（微秒）
        pre_sync_error = error * 1e6
        self.pre_sync_errors.append((sim_time, pre_sync_error))

        # 调整本地时钟，获取剩余误差
        remaining_error = self.clock.adjust(error)

        # 记录同步后的误差（微秒）
        post_sync_error = remaining_error * 1e6
        self.post_sync_errors.append((sim_time, post_sync_error))

        # 添加驻留时间并转发Sync消息
        forward_time = actual_receive_time + self.residence_time
        return forward_time

    def measure_pdelay(self, neighbor):
        # 发送Pdelay_Req消息
        t1 = self.clock.time
        t2 = neighbor.clock.time + np.random.uniform(0, PHY_JITTER)

        # 发送Pdelay_Resp消息
        t3 = neighbor.clock.time
        t4 = self.clock.time + np.random.uniform(0, PHY_JITTER)

        # 计算传播延迟（考虑非对称性）
        propagation_delay = ((t4 - t1) - self.neighbor_rate_ratio * (t3 - t2)) / 2
        self.propagation_delay = propagation_delay + self.asymmetry


class Network:
    def __init__(self):
        self.nodes = [Node(i) for i in range(NUM_NODES)]
        self.grandmaster = self.nodes[0]
        self.event_queue = []  # 事件队列
        self.current_time = 0.0
        self.event_counter = 0  # 事件计数器

        # 初始化事件队列
        self.initialize_events()

    def initialize_events(self):
        # 安排首次同步消息
        self.schedule_event(0.0, self.send_sync, self.grandmaster)

        # 安排首次传播延迟测量
        for i in range(1, NUM_NODES):
            self.schedule_event(0.0, self.measure_pdelay, self.nodes[i], self.nodes[i-1])

        # 安排时钟更新
        self.schedule_event(0.0, self.update_clocks)

    def schedule_event(self, time, callback, *args):
        # 将事件加入队列，使用事件计数器避免时间相同时的顺序问题
        heapq.heappush(self.event_queue, (time, self.event_counter, callback, args))
        self.event_counter += 1

    def run_simulation(self):
        # 事件驱动仿真
        while self.event_queue and self.current_time < SIM_TIME:
            time, _, callback, args = heapq.heappop(self.event_queue)
            self.current_time = time
            callback(*args)

    def update_clocks(self):
        # 更新所有节点的时钟
        for node in self.nodes:
            node.clock.update(self.current_time)

        # 安排下一次时钟更新
        next_update = self.current_time + CLOCK_UPDATE_INTERVAL
        if next_update < SIM_TIME:
            self.schedule_event(next_update, self.update_clocks)

    def send_sync(self, node):
        # 发送同步消息
        sync_time = self.current_time
        correction_field = 0.0

        # 消息逐跳传播
        for i in range(1, NUM_NODES):
            forward_time = self.nodes[i].receive_sync(sync_time, correction_field, self.current_time)
            correction_field += self.nodes[i].propagation_delay * self.nodes[i].rate_ratio
            sync_time = forward_time

        # 安排下一次同步消息
        next_sync = self.current_time + SYNC_INTERVAL
        if next_sync < SIM_TIME:
            self.schedule_event(next_sync, self.send_sync, node)

    def measure_pdelay(self, node, neighbor):
        # 测量传播延迟
        node.measure_pdelay(neighbor)

        # 安排下一次测量
        next_measure = self.current_time + PDELAY_INTERVAL
        if next_measure < SIM_TIME:
            self.schedule_event(next_measure, self.measure_pdelay, node, neighbor)

    def save_results(self, filename_prefix):
        """保存所有节点的同步前误差到CSV文件"""
        results_dir = "simulation_results"
        os.makedirs(results_dir, exist_ok=True)

        hops_to_save = [1, 10, 50, 100]  # 要保存的跳数

        for hop in hops_to_save:
            if hop >= NUM_NODES:
                continue

            filename = f"{results_dir}/{filename_prefix}_hop{hop}.csv"

            # 提取误差数据
            node = self.nodes[hop]
            times, errors = zip(*node.pre_sync_errors) if node.pre_sync_errors else ([], [])

            # 写入CSV
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Time (s)", "Error (us)"])
                for t, e in zip(times, errors):
                    writer.writerow([t, e])

    def plot_results(self, hop):
        """绘制特定跳数节点的时间误差"""
        if hop < 1 or hop >= NUM_NODES:
            print(f"错误: 无效的跳数 {hop}，应在1和{NUM_NODES-1}之间。")
            return

        node = self.nodes[hop]

        # 提取同步前误差数据
        times, errors = zip(*node.pre_sync_errors) if node.pre_sync_errors else ([], [])

        # 绘制时间误差图
        plt.figure(figsize=(10, 6))
        plt.plot(times, errors, 'b-', linewidth=1.5, label=f'跳数={hop}')
        plt.xlabel('时间 (s)')
        plt.ylabel('时间误差 (微秒)')
        plt.title(f'时间误差随时间变化 (跳数={hop}, 同步间隔={SYNC_INTERVAL}s)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 打印误差统计
        if errors:
            error_min = min(errors)
            error_max = max(errors)
            error_avg = sum(errors) / len(errors)
            error_std = np.std(errors)
            print(f"跳数 {hop} 的统计数据:")
            print(f"  最小误差: {error_min:.3f} 微秒")
            print(f"  最大误差: {error_max:.3f} 微秒")
            print(f"  平均误差: {error_avg:.3f} 微秒")
            print(f"  标准差: {error_std:.3f} 微秒")
            print(f"  误差范围: [{error_min:.3f}, {error_max:.3f}] 微秒")

            # 添加统计信息到图表
            plt.figtext(0.15, 0.85, f"最小值: {error_min:.3f} μs\n"
                                  f"最大值: {error_max:.3f} μs\n"
                                  f"均值: {error_avg:.3f} μs\n"
                                  f"标准差: {error_std:.3f} μs",
                      bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.show()


def run_comparison():
    """运行两种同步间隔的对比仿真"""
    # 运行31.25ms间隔仿真
    print("\n=== 运行31.25ms同步间隔仿真 ===")
    SYNC_INTERVAL = 0.03125
    network1 = Network()
    network1.run_simulation()
    network1.save_results("interval_31_25ms")

    # 显示关键节点的结果
    for hop in [1, 10, 50, 100]:
        if hop < NUM_NODES:
            network1.plot_results(hop)

    # 运行1s间隔仿真
    print("\n=== 运行1s同步间隔仿真 ===")
    SYNC_INTERVAL = 1.0
    network2 = Network()
    network2.run_simulation()
    network2.save_results("interval_1s")

    # 显示关键节点的结果
    for hop in [1, 10, 50, 100]:
        if hop < NUM_NODES:
            network2.plot_results(hop)


if __name__ == "__main__":
    run_comparison()