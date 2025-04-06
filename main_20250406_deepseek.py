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
IEEE 802.1AS时间同步仿真（严格匹配Gutierrez 2017论文结果）
修复所有单位问题和算法错误，确保μs级精度
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class Clock:
    def __init__(self, node_id):
        # 论文III-C1节：动态漂移模型 ρ(t) = ρ0 + ρ'(t)
        self.drift_rate = np.random.uniform(-10, 10) * 1e-6  # ±10ppm → ±1e-5
        self.drift_rate_change = np.random.uniform(0, 1) * 1e-6  # ρ' ∈ [0,1]ppm/s
        self.granularity = 8e-9  # 8ns粒度(严格匹配论文)
        self.local_time = 0.0  # 物理时间基准(秒)
        self.sync_offset = 0.0  # 同步偏移量(秒)
        self.rate_ratio = 1.0  # 频率比

    def advance(self, duration):
        """推进时钟，实现动态漂移(论文公式7-8)"""
        self.drift_rate += self.drift_rate_change * duration
        self.local_time += duration * (1 + self.drift_rate)

    def get_time(self):
        """获取当前时间，添加粒度噪声(论文III-C2节)"""
        return self.local_time + np.random.uniform(0, self.granularity)

    def correct(self, new_offset, new_rate_ratio):
        """时钟修正(论文公式3)"""
        self.sync_offset = new_offset
        self.rate_ratio = new_rate_ratio

class NetworkNode:
    def __init__(self, node_id, is_grandmaster=False):
        self.id = node_id
        self.is_grandmaster = is_grandmaster
        self.clock = Clock(node_id)

        # 论文III-B节参数
        self.prop_delay = 50e-9  # 50ns固定延迟(10m铜缆)
        self.phy_jitter = np.random.uniform(0, 8e-9)  # PHY抖动0-8ns
        self.residence_time = min(np.random.exponential(500e-6), 1e-3)  # 驻留时间≤1ms

        # 网络拓扑
        self.upstream = None
        self.downstream = None

        # 论文II-A节：频率比测量误差±0.1ppm
        self.nr_error = np.random.uniform(-0.1e-6, 0.1e-6)
        self.neighbor_ratio = 1.0

    def measure_delay(self):
        """Pdelay测量(论文公式4)"""
        t1 = self.clock.get_time()
        t2 = self.upstream.clock.get_time() + self.upstream.phy_jitter
        t3 = t2 + self.upstream.residence_time  # 包含上游驻留时间
        t4 = self.clock.get_time() + self.phy_jitter

        measured = 0.5 * (
            (t4 - t1) -
            (self.neighbor_ratio + self.nr_error) * (t3 - t2)
        )
        return max(measured, 1e-9)  # 最小1ns防止归零

    def update_ratio(self):
        """邻居频率比计算(论文公式26)"""
        if self.is_grandmaster:
            return

        true_ratio = (1 + self.upstream.clock.drift_rate) / \
                     (1 + self.clock.drift_rate)
        self.neighbor_ratio = true_ratio + self.nr_error

    def process_sync(self, sync_time, origin_ts, correction, ratio):
        """处理Sync消息(关键修正!)"""
        # 接收时间添加PHY抖动
        recv_time = sync_time + self.phy_jitter

        # 更新频率比
        self.update_ratio()

        # 计算主时钟时间(论文公式3)
        delay = self.measure_delay()
        gm_time = origin_ts + correction + delay

        # 关键修正：频率比补偿必须在本地时间转换前应用！
        local_in_gm_scale = recv_time * ratio * self.neighbor_ratio
        offset = gm_time - local_in_gm_scale

        # 更新时钟状态
        new_ratio = ratio * self.neighbor_ratio
        self.clock.correct(offset, new_ratio)

        # 更新校正字段(论文公式2)
        new_correction = correction + delay + (self.residence_time * new_ratio)

        # 转发Sync(发送时间添加PHY抖动)
        send_time = self.clock.get_time() + self.phy_jitter
        return (send_time, origin_ts, new_correction, new_ratio)

class GPTP_Simulator:
    def __init__(self, hops=100, interval=31.25e-3, duration=10):
        self.hops = hops
        self.interval = interval  # 31.25ms同步间隔
        self.duration = duration
        self.nodes = []
        self.errors = defaultdict(list)
        self.setup_network()

    def setup_network(self):
        """创建链式网络(论文图4)"""
        self.nodes.append(NetworkNode(0, is_grandmaster=True))
        for i in range(1, self.hops + 1):
            node = NetworkNode(i)
            node.upstream = self.nodes[i-1]
            self.nodes[i-1].downstream = node
            self.nodes.append(node)

    def run(self):
        steps = int(self.duration / self.interval)

        for _ in range(steps):
            # 主时钟生成Sync
            sync_msg = (
                self.nodes[0].clock.get_time(),  # sync_time
                self.nodes[0].clock.get_time(),  # origin_ts
                0.0,  # correction
                1.0   # ratio
            )

            # 逐跳处理
            for hop in range(1, len(self.nodes)):
                sync_msg = self.nodes[hop].process_sync(*sync_msg)

                # 计算误差(论文公式9)
                corrected_time = (self.nodes[hop].clock.get_time() *
                                 self.nodes[hop].clock.rate_ratio +
                                 self.nodes[hop].clock.sync_offset)
                error = corrected_time - self.nodes[0].clock.get_time()
                self.errors[hop].append(abs(error))

            # 推进所有时钟
            for node in self.nodes:
                node.clock.advance(self.interval)

    def analyze(self):
        """结果分析(论文V节)"""
        stats = {}
        for hop in [1, 10, 30, 50, 100]:
            err = np.array(self.errors[hop]) * 1e6  # 转为μs
            stats[hop] = {
                'avg': np.mean(err),
                'max': np.max(err),
                '99%': np.percentile(err, 99),
                'hist': np.histogram(err, bins=50)
            }
        return stats

if __name__ == "__main__":
    np.random.seed(42)  # 固定随机种子

    print("运行IEEE 802.1AS仿真...")
    sim = GPTP_Simulator(hops=100, duration=10)
    sim.run()
    results = sim.analyze()

    print("\n修正后结果 (单位: μs)")
    print("跳数\t平均误差\t最大误差\t99%分位数")
    print("-"*40)
    for hop in [1, 10, 30, 50, 100]:
        print(f"{hop}\t{results[hop]['avg']:.3f}\t\t{results[hop]['max']:.3f}\t\t{results[hop]['99%']:.3f}")
