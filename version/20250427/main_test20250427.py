"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/4/27 13:36
*  @Project :   pj_gptp_simulation
*  @Description :   Description
*  @FileName:   main_test20250427.py
**************************************
"""

# IEEE 802.1AS 在 IEC/IEEE 60802 中的时间同步仿真
# 基于 McCall 等人文档(2021-2022)的分析

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import random


@dataclass
class SimulationParameters:
    """时间同步仿真的参数"""
    # 网络配置
    num_hops: int = 100  # 链中的跳数
    num_runs: int = 10000  # 蒙特卡洛运行次数

    # 时钟特性
    gm_clock_drift_max: float = 1.5  # 最大GM时钟漂移（ppm/s）
    gm_clock_drift_min: float = -1.5  # 最小GM时钟漂移（ppm/s）
    gm_clock_drift_fraction: float = 0.8  # 具有漂移的GM节点比例

    clock_drift_max: float = 1.5  # 最大非GM时钟漂移（ppm/s）
    clock_drift_min: float = -1.5  # 最小非GM时钟漂移（ppm/s）
    clock_drift_fraction: float = 0.8  # 具有漂移的非GM节点比例

    # 时间戳误差特性
    tsge_tx: float = 4.0  # TX时间戳粒度误差（±ns）
    tsge_rx: float = 4.0  # RX时间戳粒度误差（±ns）
    dtse_tx: float = 4.0  # TX动态时间戳误差（±ns）
    dtse_rx: float = 4.0  # RX动态时间戳误差（±ns）

    # 消息间隔
    pdelay_interval: float = 125.0  # pDelay消息间隔（ms）
    sync_interval: float = 125.0  # 同步消息间隔（ms）
    pdelay_turnaround: float = 10.0  # pDelay响应时间（ms）
    residence_time: float = 10.0  # 节点内驻留时间（ms）

    # 校正因子
    mean_link_delay_correction: float = 0.98  # 平均链路延迟平均的有效性
    nrr_drift_correction: float = 0.90  # NRR漂移校正有效性
    rr_drift_correction: float = 0.90  # RR漂移校正有效性
    pdelayresp_sync_correction: float = 0.0  # pDelay响应到同步的对齐因子

    # NRR平滑参数
    mnrr_smoothing_n: int = 3  # 使用的先前pDelayResp数量
    mnrr_smoothing_m: int = 1  # 用于中值计算（在推荐设置中未使用）


@dataclass
class NodeState:
    """链中节点的状态"""
    # 时钟相关状态
    clock_drift: float = 0.0  # 时钟漂移率（ppm/s）

    # 时间戳误差
    t1_pderror: float = 0.0  # pDelay请求的TX时间戳误差
    t2_pderror: float = 0.0  # pDelay请求的RX时间戳误差
    t3_pderror: float = 0.0  # pDelay响应的TX时间戳误差
    t4_pderror: float = 0.0  # pDelay响应的RX时间戳误差
    t3_pderror_prev: List[float] = field(default_factory=list)  # 用于NRR计算的先前t3误差
    t4_pderror_prev: List[float] = field(default_factory=list)  # 用于NRR计算的先前t4误差

    t2_sinerror: float = 0.0  # 同步的RX时间戳误差
    t1_souterror: float = 0.0  # 同步的TX时间戳误差

    # 误差累积
    mnrr_error: float = 0.0  # 邻居速率比误差
    mnrr_error_ts: float = 0.0  # 由时间戳误差导致的NRR误差
    mnrr_error_cd: float = 0.0  # 由时钟漂移导致的NRR误差

    rr_error: float = 0.0  # 速率比误差
    rr_error_sum: float = 0.0  # 累积的RR误差

    mean_link_delay_error: float = 0.0  # 链路延迟测量误差
    residence_time_error: float = 0.0  # 驻留时间测量误差

    dte: float = 0.0  # 该节点的动态时间误差


class TimeSyncSimulation:
    """IEEE 802.1AS 在 IEC/IEEE 60802 中的时间同步仿真"""

    def __init__(self, params: SimulationParameters):
        self.params = params
        self.results = {
            'dte_max': [],  # 所有运行中的最大DTE
            'dte_7sigma': [],  # DTE的7-sigma值
            'dte_per_hop': np.zeros((params.num_runs, params.num_hops))  # 每次运行中每个跳的DTE
        }

    def generate_timestamp_error(self, is_tx: bool) -> float:
        """根据参数生成随机时间戳误差"""
        if is_tx:
            tsge = np.random.uniform(-self.params.tsge_tx, self.params.tsge_tx)
            dtse = np.random.uniform(-self.params.dtse_tx, self.params.dtse_tx)
        else:
            tsge = np.random.uniform(-self.params.tsge_rx, self.params.tsge_rx)
            dtse = np.random.uniform(-self.params.dtse_rx, self.params.dtse_rx)
        return tsge + dtse

    def generate_clock_drift(self, is_gm: bool) -> float:
        """根据参数生成随机时钟漂移"""
        if is_gm:
            if np.random.random() <= self.params.gm_clock_drift_fraction:
                return np.random.uniform(self.params.gm_clock_drift_min, self.params.gm_clock_drift_max)
            return 0.0
        else:
            if np.random.random() <= self.params.clock_drift_fraction:
                return np.random.uniform(self.params.clock_drift_min, self.params.clock_drift_max)
            return 0.0

    def generate_pdelay_interval(self) -> float:
        """在规格范围内生成随机pDelay间隔"""
        return np.random.uniform(0.9 * self.params.pdelay_interval,
                                 1.3 * self.params.pdelay_interval)

    def run_simulation(self):
        """运行时间同步仿真"""
        for run in range(self.params.num_runs):
            # 为新的运行重置
            nodes = [NodeState() for _ in range(self.params.num_hops + 1)]  # +1 为GM

            # 为所有节点生成时钟漂移
            nodes[0].clock_drift = self.generate_clock_drift(is_gm=True)  # GM
            for i in range(1, self.params.num_hops + 1):
                nodes[i].clock_drift = self.generate_clock_drift(is_gm=False)

            # 计算所有跳的误差
            dte = 0.0
            for hop in range(1, self.params.num_hops + 1):
                # 生成时间戳误差
                nodes[hop].t1_pderror = self.generate_timestamp_error(is_tx=True)
                nodes[hop].t2_pderror = self.generate_timestamp_error(is_tx=False)
                nodes[hop].t3_pderror = self.generate_timestamp_error(is_tx=True)
                nodes[hop].t4_pderror = self.generate_timestamp_error(is_tx=False)
                nodes[hop].t1_souterror = self.generate_timestamp_error(is_tx=True)
                nodes[hop].t2_sinerror = self.generate_timestamp_error(is_tx=False)

                # 为NRR计算生成先前的时间戳
                for n in range(1, self.params.mnrr_smoothing_n):
                    nodes[hop].t3_pderror_prev.append(self.generate_timestamp_error(is_tx=True))
                    nodes[hop].t4_pderror_prev.append(self.generate_timestamp_error(is_tx=False))

                # 计算NRR误差组件
                self.calculate_mnrr_errors(nodes, hop)

                # 计算RR误差
                if hop == 1:
                    nodes[hop].rr_error = nodes[hop].mnrr_error
                    nodes[hop].rr_error_sum = nodes[hop].rr_error
                else:
                    # 添加由NRR测量到同步之间的时钟漂移引起的RR误差
                    pdelay_to_sync = np.random.uniform(0, self.params.pdelay_interval) * (
                                1.0 - self.params.pdelayresp_sync_correction)
                    rr_error_cd_nrr2sync = (pdelay_to_sync * (
                                nodes[hop].clock_drift - nodes[hop - 1].clock_drift) / 1000) * (
                                                       1.0 - self.params.nrr_drift_correction)

                    # 添加由上游RR计算到同步之间的时钟漂移引起的RR误差
                    rr_error_cd_rr2sync = (self.params.residence_time * (
                                nodes[hop - 1].clock_drift - nodes[0].clock_drift) / 1000) * (
                                                      1.0 - self.params.rr_drift_correction)

                    # 累积RR误差
                    nodes[hop].rr_error = nodes[hop - 1].rr_error + nodes[
                        hop].mnrr_error + rr_error_cd_nrr2sync + rr_error_cd_rr2sync
                    nodes[hop].rr_error_sum = nodes[hop].rr_error

                # 计算平均链路延迟误差
                pdelay_error_ts = (nodes[hop].t4_pderror - nodes[hop].t1_pderror - nodes[hop].t3_pderror + nodes[
                    hop].t2_pderror) / 2
                pdelay_error_ts *= (1.0 - self.params.mean_link_delay_correction)

                pdelay_error_nrr = -self.params.pdelay_turnaround * nodes[hop].mnrr_error / 2
                pdelay_error_nrr *= (1.0 - self.params.mean_link_delay_correction)

                nodes[hop].mean_link_delay_error = pdelay_error_ts + pdelay_error_nrr

                # 计算驻留时间误差或终端站误差
                if hop < self.params.num_hops:  # 不是最后一跳
                    # 驻留时间误差组件
                    rt_error_ts_direct = nodes[hop].t1_souterror - nodes[hop].t2_sinerror
                    rt_error_rr = self.params.residence_time * nodes[hop].rr_error
                    rt_error_cd_direct = (self.params.residence_time ** 2 * (
                                nodes[hop].clock_drift - nodes[0].clock_drift) / (2 * 1000)) * (
                                                     1.0 - self.params.rr_drift_correction)

                    nodes[hop].residence_time_error = rt_error_ts_direct + rt_error_rr + rt_error_cd_direct
                    nodes[hop].dte = dte + nodes[hop].mean_link_delay_error + nodes[hop].residence_time_error
                else:  # 最后一跳（终端站）
                    # 终端站误差组件
                    sync_interval = self.params.sync_interval
                    es_error_rr = sync_interval * nodes[hop].rr_error
                    es_error_cd_direct = (sync_interval / 2 * (
                                nodes[hop].clock_drift - nodes[0].clock_drift) / 1000) * (
                                                     1.0 - self.params.rr_drift_correction)

                    end_station_error = es_error_rr + es_error_cd_direct
                    nodes[hop].dte = dte + nodes[hop].mean_link_delay_error + end_station_error

                # 更新下一跳的累积DTE
                dte = nodes[hop].dte

                # 存储结果
                self.results['dte_per_hop'][run, hop - 1] = dte

            # 计算完所有跳后，存储此次运行的最大DTE
            if run == 0 or abs(dte) > self.results['dte_max'][-1]:
                self.results['dte_max'].append(abs(dte))

        # 计算7-sigma DTE（比最大值更具统计代表性）
        for hop in range(self.params.num_hops):
            dte_at_hop = self.results['dte_per_hop'][:, hop]
            self.results['dte_7sigma'].append(np.std(dte_at_hop) * 7)

    def calculate_mnrr_errors(self, nodes: List[NodeState], hop: int):
        """计算给定跳的mNRR误差组件"""
        # 基于mNRR平滑计算有效pDelay间隔
        tpdelay2pdelay = 0
        for n in range(self.params.mnrr_smoothing_n):
            tpdelay2pdelay += self.generate_pdelay_interval()

        # 计算由时间戳引起的mNRR误差
        if self.params.mnrr_smoothing_n > 1 and len(nodes[hop].t3_pderror_prev) >= self.params.mnrr_smoothing_n - 1:
            # 使用先前的时间戳进行NRR计算
            t3pd_diff = nodes[hop].t3_pderror - nodes[hop].t3_pderror_prev[-1]
            t4pd_diff = nodes[hop].t4_pderror - nodes[hop].t4_pderror_prev[-1]
        else:
            # 使用最近的时间戳进行默认计算
            t3pd_diff = nodes[hop].t3_pderror - 0  # 假设先前样本的误差为0（简化）
            t4pd_diff = nodes[hop].t4_pderror - 0

        nodes[hop].mnrr_error_ts = (t3pd_diff - t4pd_diff) / tpdelay2pdelay

        # 计算由时钟漂移引起的mNRR误差
        nodes[hop].mnrr_error_cd = (tpdelay2pdelay * (nodes[hop].clock_drift - nodes[hop - 1].clock_drift) / (
                    2 * 1000)) * (1.0 - self.params.nrr_drift_correction)

        # 总mNRR误差
        nodes[hop].mnrr_error = nodes[hop].mnrr_error_ts + nodes[hop].mnrr_error_cd

    def plot_results(self):
        """绘制仿真结果"""
        plt.figure(figsize=(12, 8))

        # 绘制最终跳的DTE分布
        plt.subplot(2, 1, 1)
        final_hop_dte = self.results['dte_per_hop'][:, -1]
        plt.hist(final_hop_dte, bins=50, alpha=0.7)
        plt.axvline(x=self.results['dte_7sigma'][-1], color='r', linestyle='--',
                    label=f'7σ: {self.results["dte_7sigma"][-1]:.1f} ns')
        plt.axvline(x=-self.results['dte_7sigma'][-1], color='r', linestyle='--')
        plt.axvline(x=1000, color='g', linestyle=':', label='±1μs 目标')
        plt.axvline(x=-1000, color='g', linestyle=':')
        plt.xlabel('动态时间误差 (ns)')
        plt.ylabel('计数')
        plt.title(f'跳数 {self.params.num_hops} 处的DTE分布')
        plt.legend()

        # 绘制DTE随跳数的增长
        plt.subplot(2, 1, 2)
        hops = np.arange(1, self.params.num_hops + 1)
        plt.plot(hops, self.results['dte_7sigma'], 'b-', label='7σ DTE')
        plt.axhline(y=1000, color='g', linestyle=':', label='±1μs 目标')
        plt.xlabel('跳数')
        plt.ylabel('动态时间误差 (ns)')
        plt.title('DTE随跳数的增长（7σ值）')
        plt.legend()

        plt.tight_layout()
        plt.show()


# 使用推荐参数运行仿真
def main():
    # 使用推荐设置创建参数
    params = SimulationParameters(
        num_hops=100,
        num_runs=10000,  # 为演示减少

        # 时钟特性
        gm_clock_drift_max=1.5,
        gm_clock_drift_min=-1.5,
        gm_clock_drift_fraction=0.8,
        clock_drift_max=1.5,
        clock_drift_min=-1.5,
        clock_drift_fraction=0.8,

        # 时间戳误差
        tsge_tx=4.0,
        tsge_rx=4.0,
        dtse_tx=4.0,
        dtse_rx=4.0,

        # 消息间隔
        pdelay_interval=125.0,
        sync_interval=125.0,
        pdelay_turnaround=10.0,
        residence_time=10.0,

        # 校正因子 - 推荐设置
        mean_link_delay_correction=0.98,
        nrr_drift_correction=0.90,
        rr_drift_correction=0.90,
        pdelayresp_sync_correction=0.0,
        mnrr_smoothing_n=3,
        mnrr_smoothing_m=1
    )

    # 创建并运行仿真
    sim = TimeSyncSimulation(params)
    print("使用推荐参数运行仿真...")
    sim.run_simulation()

    # 输出结果
    max_dte = max(sim.results['dte_max'])
    final_7sigma = sim.results['dte_7sigma'][-1]

    print(f"仿真完成！")
    print(f"最大DTE: {max_dte:.1f} ns")
    print(f"跳数 {params.num_hops} 处的7-sigma DTE: {final_7sigma:.1f} ns")
    print(f"目标 (<1000 ns): {'通过' if final_7sigma < 1000 else '失败'}")

    # 绘制结果
    sim.plot_results()


if __name__ == "__main__":
    main()