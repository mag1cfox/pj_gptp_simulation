"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/4/6 21:17
*  @Project :   pj_gptp_simulation
*  @Description :   Description
*  @FileName:   main_20250406_deepseek.py
**************************************
"""
"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/4/6 21:17
*  @Project :   pj_gptp_simulation
*  @Description :   IEEE 802.1AS (gPTP)时间同步仿真(Gutierrez et al. 2017模型)
*  @FileName:   main_20250406_deepseek.py
**************************************
"""
"""
IEEE 802.1AS (gPTP)时间同步仿真 - 精确复现Gutierrez et al. 2017结果
严格遵循论文参数：PHY抖动(0-8ns)、时钟粒度(8ns)、驻留时间(≤1ms)、漂移率(±10ppm)
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class Clock:
    def __init__(self, node_id):
        self.node_id = node_id
        # 论文III-C1节：时钟模型 ρ(t) = ρ0 + ρ'(t)
        self.drift_rate = np.random.uniform(-10e-6, 10e-6)  # ±10ppm → ±1e-5
        self.drift_rate_change = np.random.uniform(0, 1e-6)  # ρ' ∈ [0,1]ppm/s → [0,1e-6]/s
        self.granularity = 8e-9  # 8ns粒度(论文III-C2节)
        self.local_time = 0.0  # 物理时间基准(秒)
        self.offset = 0.0  # 与主时钟的偏移量(秒)
        self.rate_ratio = 1.0  # 频率比(相对于主时钟)

    def advance(self, duration):
        """推进时钟，考虑动态漂移(论文公式7-8)"""
        self.drift_rate += self.drift_rate_change * duration
        self.local_time += duration * (1 + self.drift_rate)

    def get_time(self):
        """获取当前时间，添加粒度噪声(论文III-C2节)"""
        return self.local_time + np.random.uniform(0, self.granularity)

    def correct_clock(self, new_offset, new_rate_ratio):
        """时钟修正(论文公式3)"""
        self.offset = new_offset
        self.rate_ratio = new_rate_ratio


class NetworkNode:
    def __init__(self, node_id, is_grandmaster=False):
        self.id = node_id
        self.is_grandmaster = is_grandmaster
        self.clock = Clock(node_id)

        # 论文III-B节：网络参数
        self.propagation_delay = 50e-9  # 50ns固定传播延迟(10m铜缆)
        self.phy_jitter = np.random.uniform(0, 8e-9)  # PHY抖动0-8ns(均匀分布)
        self.residence_time = min(np.random.exponential(500e-6), 1e-3)  # 驻留时间≤1ms

        # 网络拓扑
        self.upstream_node = None
        self.downstream_node = None

        # 论文II-A节：邻居频率比测量误差±0.1ppm
        self.nr_measurement_error = np.random.uniform(-0.1e-6, 0.1e-6)
        self.neighbor_rate_ratio = 1.0

    def measure_propagation_delay(self):
        """Pdelay测量(论文公式4)"""
        # 四个时间戳(考虑PHY抖动和粒度)
        t1 = self.clock.get_time()
        t2 = self.upstream_node.clock.get_time() + self.upstream_node.phy_jitter
        t3 = t2 + self.upstream_node.residence_time  # 驻留时间影响t3
        t4 = self.clock.get_time() + self.phy_jitter

        # 计算传播延迟(含测量误差)
        measured_delay = 0.5 * (
                (t4 - t1) -
                (self.neighbor_rate_ratio + self.nr_measurement_error) * (t3 - t2)
        )
        return max(measured_delay, 0)  # 延迟不为负

    def update_neighbor_rate_ratio(self):
        """邻居频率比计算(论文公式26)"""
        if self.is_grandmaster:
            return

        # 实际频率比 + 标准允许的测量误差
        true_ratio = (1 + self.upstream_node.clock.drift_rate) / \
                     (1 + self.clock.drift_rate)
        self.neighbor_rate_ratio = true_ratio + self.nr_measurement_error

    def process_sync_message(self, sync_time, origin_timestamp, correction_field, rate_ratio):
        """处理Sync消息(论文II-A节)"""
        # 接收时间添加PHY抖动
        sync_receive_time = sync_time + self.phy_jitter

        # 更新邻居频率比
        self.update_neighbor_rate_ratio()

        # 计算主时钟时间(论文公式3)
        measured_delay = self.measure_propagation_delay()
        gm_time = origin_timestamp + correction_field + measured_delay

        # 关键修正：频率比补偿必须在本地时间转换前应用！
        local_time_in_gm_scale = sync_receive_time * rate_ratio * self.neighbor_rate_ratio
        clock_offset = gm_time - local_time_in_gm_scale

        # 更新时钟状态
        new_rate_ratio = rate_ratio * self.neighbor_rate_ratio
        self.clock.correct_clock(clock_offset, new_rate_ratio)

        # 更新校正字段(论文公式2)
        residence_delay = self.residence_time * self.clock.rate_ratio
        new_correction = correction_field + measured_delay + residence_delay

        # 转发Sync(发送时间添加PHY抖动)
        send_time = self.clock.get_time() + self.phy_jitter
        return (send_time, origin_timestamp, new_correction, self.clock.rate_ratio)


class GPTPNetworkSimulator:
    def __init__(self, num_hops=100, sync_interval=31.25e-3, duration=100):
        self.num_hops = num_hops
        self.sync_interval = sync_interval  # 31.25ms(论文III-D节)
        self.duration = duration
        self.nodes = []
        self.time_errors = defaultdict(list)
        self.setup_network()

    def setup_network(self):
        """创建链式网络(论文图4)"""
        # 主时钟节点
        grandmaster = NetworkNode(0, is_grandmaster=True)
        self.nodes.append(grandmaster)

        # 从时钟节点
        for i in range(1, self.num_hops + 1):
            node = NetworkNode(i)
            node.upstream_node = self.nodes[i - 1]
            self.nodes[i - 1].downstream_node = node
            self.nodes.append(node)

    def run_simulation(self):
        num_syncs = int(self.duration / self.sync_interval)

        for _ in range(num_syncs):
            # 主时钟生成Sync
            sync_msg = (
                self.nodes[0].clock.get_time(),  # sync_time
                self.nodes[0].clock.get_time(),  # origin_timestamp
                0.0,  # correction_field
                1.0  # rate_ratio
            )

            # 逐跳传播
            for hop in range(1, len(self.nodes)):
                sync_msg = self.nodes[hop].process_sync_message(*sync_msg)

                # 记录时间误差(论文公式9)
                corrected_time = self.nodes[hop].clock.get_time() * self.nodes[hop].clock.rate_ratio + \
                                 self.nodes[hop].clock.offset
                error = corrected_time - self.nodes[0].clock.get_time()
                self.time_errors[hop].append(abs(error))

            # 所有节点时钟推进
            for node in self.nodes:
                node.clock.advance(self.sync_interval)

    def analyze_results(self):
        """结果分析(论文V节)"""
        percentiles = {}
        for hop in [1, 10, 30, 50, 100]:
            errors = np.array(self.time_errors[hop]) * 1e6  # 转为μs
            percentiles[hop] = {
                'avg': np.mean(errors),
                'max': np.max(errors),
                '99%': np.percentile(errors, 99),
                'hist': np.histogram(errors, bins=np.linspace(0, 3, 50))
            }
        return percentiles

    def plot_results(self, percentiles):
        """绘制论文风格图表"""
        # 误差随跳数变化(论文图10)
        plt.figure(figsize=(12, 5))
        hops = range(1, self.num_hops + 1)
        avg_errors = [np.mean(np.array(self.time_errors[h]) * 1e6) for h in hops]
        plt.plot(hops, avg_errors, 'b-', linewidth=1.5)
        plt.axhline(y=1, color='r', linestyle='--', label='1μs阈值')
        plt.xlabel('跳数', fontsize=12)
        plt.ylabel('时间误差 (μs)', fontsize=12)
        plt.title('IEEE 802.1AS同步精度 vs 跳数 (Gutierrez et al. 2017)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 第100跳误差分布(论文图8)
        plt.figure(figsize=(12, 5))
        hist_data = percentiles[100]['hist']
        plt.bar(hist_data[1][:-1], hist_data[0], width=0.06, alpha=0.7)
        plt.xlim(0, 3)
        plt.xlabel('时间误差 (μs)', fontsize=12)
        plt.ylabel('出现次数', fontsize=12)
        plt.title('第100跳节点时间误差分布', fontsize=14)
        plt.grid(True, alpha=0.3)

        plt.show()


if __name__ == "__main__":
    np.random.seed(42)  # 固定随机种子确保可复现

    print("正在运行IEEE 802.1AS仿真...")
    simulator = GPTPNetworkSimulator(
        num_hops=100,
        sync_interval=31.25e-3,  # 31.25ms同步间隔
        duration=10  # 10秒仿真时长(足够收敛)
    )
    simulator.run_simulation()
    results = simulator.analyze_results()

    # 打印关键结果
    print("\n仿真结果 (单位: μs)")
    print("跳数\t平均误差\t最大误差\t99%分位数")
    print("-" * 40)
    for hop in [1, 10, 30, 50, 100]:
        print(f"{hop}\t{results[hop]['avg']:.3f}\t\t{results[hop]['max']:.3f}\t\t{results[hop]['99%']:.3f}")

    # 绘制图表
    simulator.plot_results(results)


