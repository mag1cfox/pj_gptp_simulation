"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/4/6 21:17
*  @Project :   pj_gptp_simulation
*  @Description :   Description
*  @FileName:   main_20250406_deepseek.py
**************************************
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class Clock:
    def __init__(self, drift_rate=0, granularity=8e-9):
        # 初始化时钟参数
        self.drift_rate = drift_rate * 1e-6  # 转换为比例值(ppm转小数)
        self.granularity = granularity  # 时钟粒度(8纳秒)
        self.local_time = 0  # 本地时钟时间
        self.offset = 0  # 与主时钟的偏移量修正值
        self.rate_ratio = 1.0  # 时钟频率比(相对于主时钟)

    def advance(self, duration):
        # 时钟推进，考虑时钟漂移
        self.local_time += duration * (1 + self.drift_rate)

    def get_time(self):
        # 获取当前时间，考虑时钟粒度
        return self.local_time + np.random.uniform(0, self.granularity)

    def correct_clock(self, new_offset, new_rate_ratio):
        # 修正时钟偏移和频率比
        self.offset = new_offset
        self.rate_ratio = new_rate_ratio


class NetworkNode:
    def __init__(self, node_id, is_grandmaster=False):
        self.id = node_id
        self.is_grandmaster = is_grandmaster
        # 初始化时钟(-10到10ppm的随机漂移)
        self.clock = Clock(drift_rate=np.random.uniform(-10, 10))

        # 网络参数
        self.propagation_delay = 50e-9  # 50纳秒基础传播延迟
        self.phy_jitter = np.random.uniform(0, 8e-9)  # PHY抖动(0-8纳秒)
        self.residence_time = np.random.uniform(10e-6, 1e-3)  # 驻留时间(10μs-1ms)
        self.upstream_node = None  # 上游节点
        self.downstream_node = None  # 下游节点

        # 邻居时钟频率比(相对于上游节点)
        self.neighbor_rate_ratio = 1.0

    def measure_propagation_delay(self):
        # 测量传播延迟(Pdelay机制)

        # 四个时间戳测量
        t1 = self.clock.get_time()  # 本地发送Pdelay_Req时间
        t2 = self.upstream_node.clock.get_time() + self.phy_jitter  # 上游接收时间
        t3 = self.upstream_node.clock.get_time() + self.upstream_node.phy_jitter  # 上游响应时间
        t4 = self.clock.get_time() + self.phy_jitter  # 本地接收Pdelay_Resp时间

        # 计算传播延迟(考虑频率比)
        measured_delay = 0.5 * ((t4 - t1) - self.neighbor_rate_ratio * (t3 - t2))
        return max(measured_delay, 0)  # 延迟不能为负

    def process_sync_message(self, sync_time, origin_timestamp, correction_field, rate_ratio):
        # 处理Sync消息并转发给下游

        # 考虑PHY抖动的影响
        sync_receive_time = sync_time + self.phy_jitter

        # 如果不是主时钟，计算邻居频率比
        if not self.is_grandmaster:
            upstream_drift = 1 + self.upstream_node.clock.drift_rate
            local_drift = 1 + self.clock.drift_rate
            self.neighbor_rate_ratio = upstream_drift / local_drift

        # 计算主时钟时间
        gm_time = (origin_timestamp +
                   correction_field +
                   self.measure_propagation_delay())

        # 计算本地需要修正的偏移量
        local_time = sync_receive_time
        clock_offset = gm_time - local_time
        new_rate_ratio = rate_ratio * self.neighbor_rate_ratio
        self.clock.correct_clock(clock_offset, new_rate_ratio)

        # 更新校正字段
        residence_delay = self.residence_time * self.clock.rate_ratio
        new_correction = (correction_field +
                          self.measure_propagation_delay() +
                          residence_delay)

        # 返回更新后的Sync消息(添加PHY抖动)
        return (
            self.clock.get_time() + self.phy_jitter,  # 发送时间
            origin_timestamp,  # 原始时间戳不变
            new_correction,  # 更新后的校正字段
            self.clock.rate_ratio  # 更新后的频率比
        )


class GPTPNetworkSimulator:
    def __init__(self, num_hops=100, sync_interval=31.25e-3, duration=100):
        self.num_hops = num_hops
        self.sync_interval = sync_interval  # 31.25ms同步间隔
        self.duration = duration  # 仿真时长100秒
        self.nodes = []  # 网络节点列表
        self.time_errors = defaultdict(list)  # 记录各节点的时间误差
        self.setup_network()

    def setup_network(self):
        # 创建主时钟节点
        grandmaster = NetworkNode(0, is_grandmaster=True)
        self.nodes.append(grandmaster)

        # 创建从时钟节点并连接成链
        for i in range(1, self.num_hops + 1):
            node = NetworkNode(i)
            node.upstream_node = self.nodes[i - 1]
            self.nodes[i - 1].downstream_node = node
            self.nodes.append(node)

    def run_simulation(self):
        num_syncs = int(self.duration / self.sync_interval)

        for _ in range(num_syncs):
            # 主时钟生成Sync消息(时间原点标记、校正字段初始为0,频率比初始为1.0)
            sync_time = self.nodes[0].clock.get_time()
            sync_msg = (sync_time, sync_time, 0, 1.0)

            # 消息通过每一跳传播
            for hop in range(1, len(self.nodes)):
                sync_msg = self.nodes[hop].process_sync_message(*sync_msg)

                # 计算当前节点与主时钟的时间误差(纳秒)
                corrected_time = (self.nodes[hop].clock.get_time() +
                                  self.nodes[hop].clock.offset)
                error = corrected_time - self.nodes[0].clock.get_time()
                self.time_errors[hop].append(abs(error))

            # 所有节点时钟推进一个同步间隔
            for node in self.nodes:
                node.clock.advance(self.sync_interval)

    def analyze_results(self):
        # 计算每跳的平均和最大时间误差(转换为微秒)
        avg_errors = []
        max_errors = []

        for hop in range(1, self.num_hops + 1):
            hop_errors = np.array(self.time_errors[hop]) * 1e6  # 转换为微秒
            avg_errors.append(np.mean(hop_errors))
            max_errors.append(np.max(hop_errors))

        return avg_errors, max_errors

    def plot_results(self):
        avg_errors, max_errors = self.analyze_results()

        plt.figure(figsize=(14, 7))

        # 绘制每跳的平均和最大误差
        plt.plot(range(1, self.num_hops + 1), avg_errors,
                 'b-', label='平均误差')
        plt.plot(range(1, self.num_hops + 1), max_errors,
                 'r--', label='最大误差')

        # 绘制论文中的理论值
        theoretical = [0.625 + 0.0625 * hop for hop in range(self.num_hops)]
        plt.plot(range(1, self.num_hops + 1), theoretical,
                 'g-.', label='理论最差值')

        plt.xlabel('跳数')
        plt.ylabel('时间误差(μs)')
        plt.title(f'100跳网络gPTP时间同步误差分析(仿真时长:{self.duration}秒)')
        plt.grid(True)
        plt.legend()
        plt.show()

        # 绘制第100跳的误差分布
        plt.figure(figsize=(14, 7))
        plt.hist(np.array(self.time_errors[100]) * 1e6, bins=50)
        plt.xlabel('时间误差(μs)')
        plt.ylabel('出现次数')
        plt.title('第100跳节点的时间误差分布')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # 创建并运行仿真
    simulator = GPTPNetworkSimulator(num_hops=100, duration=100)
    simulator.run_simulation()

    # 分析并绘制结果
    simulator.plot_results()

    # 输出部分节点的典型误差值
    hop_numbers = [1, 10, 30, 50, 100]
    avg_errors, max_errors = simulator.analyze_results()
    print("\n典型跳数的时间误差(μs):")
    print("跳数\t平均误差\t最大误差")
    for hop in hop_numbers:
        print(f"{hop}\t{avg_errors[hop - 1]:.3f}\t\t{max_errors[hop - 1]:.3f}")
