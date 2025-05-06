"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/5/6 20:46
*  @Project :   pj_gptp_simulation
*  @Description :   Description
*  @FileName:   main_v1.py
**************************************
"""

# Time Synchronization Simulation for IEEE 802.1AS in IEC/IEEE 60802
# Based on analysis of McCall et al. documents (2021-2022)

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import random
import os
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter


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

    # 终端站计算方法使用的特定跳
    end_station_hops: List[int] = field(default_factory=lambda: [10, 25, 50, 75, 100])

    # 下一次同步消息相关参数
    consider_next_sync: bool = True  # 是否考虑下一次同步消息的影响
    time_to_next_sync: float = None  # 到下一次同步消息的时间(ms), None表示使用sync_interval


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

    te: float = 0.0  # 该节点的动态时间误差


class TimeSyncSimulation:
    """IEEE 802.1AS 在 IEC/IEEE 60802 中的时间同步仿真"""

    def __init__(self, params: SimulationParameters):
        self.params = params
        # 设置默认的下一次同步时间，如果未指定
        if self.params.time_to_next_sync is None:
            self.params.time_to_next_sync = self.params.sync_interval

        self.results = {
            'te_max': [],  # 所有运行中的最大te
            'te_7sigma': [],  # te的7-sigma值
            'te_per_hop': np.zeros((params.num_runs, params.num_hops))  # 每次运行中每个跳的te
        }

        # 创建输出目录
        self.output_data_dir = 'output_data_v2'
        self.output_image_dir = 'output_image_v2'
        os.makedirs(self.output_data_dir, exist_ok=True)
        os.makedirs(self.output_image_dir, exist_ok=True)

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
            te = 0.0
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

                # 修改: 对所有节点使用相同的误差计算方法（普通桥接设备方法）
                rt_error_ts_direct = nodes[hop].t1_souterror - nodes[hop].t2_sinerror
                rt_error_rr = self.params.residence_time * nodes[hop].rr_error
                rt_error_cd_direct = (self.params.residence_time ** 2 * (
                            nodes[hop].clock_drift - nodes[0].clock_drift) / (2 * 1000)) * (
                                                 1.0 - self.params.rr_drift_correction)

                nodes[hop].residence_time_error = rt_error_ts_direct + rt_error_rr + rt_error_cd_direct
                nodes[hop].te = te + nodes[hop].mean_link_delay_error + nodes[hop].residence_time_error

                # 如果需要考虑下一次同步消息的影响并且是指定跳或最后一跳，则添加额外的时钟漂移误差
                if self.params.consider_next_sync and (
                        hop == self.params.num_hops or hop in self.params.end_station_hops):
                    # 计算到下一次同步到达之前积累的额外时钟漂移误差
                    additional_drift_error = (
                                self.params.time_to_next_sync * (nodes[hop].clock_drift - nodes[0].clock_drift) / 1000)
                    nodes[hop].te += additional_drift_error

                # 更新下一跳的累积te
                te = nodes[hop].te

                # 存储结果
                self.results['te_per_hop'][run, hop - 1] = te

            # 计算完所有跳后，存储此次运行的最大te
            if run == 0 or abs(te) > self.results['te_max'][-1]:
                self.results['te_max'].append(abs(te))

        # 计算7-sigma te（比最大值更具统计代表性）
        for hop in range(self.params.num_hops):
            te_at_hop = self.results['te_per_hop'][:, hop]
            self.results['te_7sigma'].append(np.std(te_at_hop) * 7)

        # 保存数据到CSV文件
        self.save_results_to_csv()

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

    def save_results_to_csv(self):
        """将所有节点的时间误差结果保存到CSV文件中"""
        # 创建一个包含所有跳的te数据的DataFrame
        all_te_data = {}
        for hop in range(1, self.params.num_hops + 1):
            hop_data = self.results['te_per_hop'][:, hop - 1]
            all_te_data[f'Hop_{hop}'] = hop_data

        df = pd.DataFrame(all_te_data)

        # 保存到CSV文件
        df.to_csv(os.path.join(self.output_data_dir, 'te_all_hops_v5.csv'), index=False)

        # 保存7-sigma和统计数据
        stats_data = {
            'Hop': list(range(1, self.params.num_hops + 1)),
            'te_7sigma': self.results['te_7sigma'],
            'te_Mean': [np.mean(self.results['te_per_hop'][:, i]) for i in range(self.params.num_hops)],
            'te_Std': [np.std(self.results['te_per_hop'][:, i]) for i in range(self.params.num_hops)],
            'te_Min': [np.min(self.results['te_per_hop'][:, i]) for i in range(self.params.num_hops)],
            'te_Max': [np.max(self.results['te_per_hop'][:, i]) for i in range(self.params.num_hops)]
        }
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(os.path.join(self.output_data_dir, 'te_statistics_v5.csv'), index=False)

    def plot_results(self):
        """绘制仿真结果并保存图像"""
        # 1. 绘制最终跳的te分布
        self.plot_final_hop_distribution()

        # 2. 绘制te随跳数的增长
        self.plot_te_growth()

        # 3. 绘制1-7跳的时间误差数据
        self.plot_first_seven_hops()

        # 4. 绘制特定跳数(10,25,50,75,100)的时间误差折线图和CDF图
        self.plot_specific_hops_line_and_cdf()

    def plot_final_hop_distribution(self):
        """绘制最终跳的te分布"""
        plt.figure(figsize=(10, 6))
        final_hop_te = self.results['te_per_hop'][:, -1]
        plt.hist(final_hop_te, bins=50, alpha=0.7)
        # plt.axvline(x=self.results['te_7sigma'][-1], color='r', linestyle='--',
        #            label=f'7σ: {self.results["te_7sigma"][-1]:.1f} ns')
        # plt.axvline(x=-self.results['te_7sigma'][-1], color='r', linestyle='--')
        plt.axvline(x=1000, color='g', linestyle=':', label='±1μs target')
        plt.axvline(x=-1000, color='g', linestyle=':')
        plt.xlabel('Time Error (ns)')
        plt.ylabel('Count')
        plt.title(f'te Distribution at Hop {self.params.num_hops}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 保存图像
        plt.savefig(os.path.join(self.output_image_dir, 'final_hop_te_distribution_v5.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()

    def plot_te_growth(self):
        """绘制te随跳数的增长"""
        plt.figure(figsize=(10, 6))
        hops = np.arange(1, self.params.num_hops + 1)
        # plt.plot(hops, self.results['te_7sigma'], 'b-', label='7σ te')
        plt.axhline(y=1000, color='g', linestyle=':', label='±1μs target')
        plt.xlabel('Hop Number')
        plt.ylabel('Time Error (ns)')
        # plt.title('te Growth Across Hops (7σ values)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 保存图像
        plt.savefig(os.path.join(self.output_image_dir, 'te_growth_across_hops_v5.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_first_seven_hops(self):
        """绘制1-7跳的时间误差数据"""
        plt.figure(figsize=(12, 8))

        # 绘制箱形图
        first_seven_hops_data = [self.results['te_per_hop'][:, i] for i in range(7)]
        plt.boxplot(first_seven_hops_data, labels=[f'Hop {i + 1}' for i in range(7)])
        plt.xlabel('Hop Number')
        plt.ylabel('Time Error (ns)')
        plt.title('te Distribution for First 7 Hops')
        plt.grid(True, alpha=0.3)

        # 保存图像
        plt.savefig(os.path.join(self.output_image_dir, 'first_seven_hops_te_v5.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 绘制小提琴图
        plt.figure(figsize=(12, 8))
        first_seven_df = pd.DataFrame({f'Hop {i + 1}': self.results['te_per_hop'][:, i] for i in range(7)})
        sns.violinplot(data=first_seven_df)
        plt.xlabel('Hop Number')
        plt.ylabel('Time Error (ns)')
        plt.title('te Distribution for First 7 Hops (Violin Plot)')
        plt.grid(True, alpha=0.3)

        # 保存图像
        plt.savefig(os.path.join(self.output_image_dir, 'first_seven_hops_te_violin_v5.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()

    def plot_specific_hops_line_and_cdf(self):
        """绘制特定跳数(10,25,50,75,100)的时间误差折线图和CDF图"""
        specific_hops = [10, 25, 50, 75, 100]
        specific_hops = [h for h in specific_hops if h <= self.params.num_hops]

        # 折线图 - 每次运行的te随时间的变化
        plt.figure(figsize=(12, 8))
        for hop in specific_hops:
            # 选择100次运行进行可视化，否则图会太乱
            sample_runs = np.random.choice(self.params.num_runs, size=min(100, self.params.num_runs), replace=False)
            for run in sample_runs:
                if hop == specific_hops[0]:  # 只对第一个跳添加标签，避免重复
                    plt.plot(self.results['te_per_hop'][run, :hop], alpha=0.1, color=f'C{specific_hops.index(hop)}')

            # 绘制平均值线
            mean_values = np.mean(self.results['te_per_hop'][:, :hop], axis=0)
            plt.plot(mean_values, linewidth=2, label=f'Hop {hop} (avg)', color=f'C{specific_hops.index(hop)}')

        plt.xlabel('Hop Number')
        plt.ylabel('Time Error (ns)')
        plt.title('te Development for Specific Hops')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 保存图像
        plt.savefig(os.path.join(self.output_image_dir, 'specific_hops_te_line_v5.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # CDF图 - 最终te的累积分布
        plt.figure(figsize=(12, 8))
        for hop in specific_hops:
            hop_data = self.results['te_per_hop'][:, hop - 1]
            sorted_data = np.sort(hop_data)
            # 计算每个值的CDF
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            plt.plot(sorted_data, cdf, label=f'Hop {hop}')

        plt.axvline(x=1000, color='g', linestyle=':', label='±1μs target')
        plt.axvline(x=-1000, color='g', linestyle=':')
        plt.xlabel('Time Error (ns)')
        plt.ylabel('Cumulative Probability')
        plt.title('CDF of te for Specific Hops')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 保存图像
        plt.savefig(os.path.join(self.output_image_dir, 'specific_hops_te_cdf_v5.png'), dpi=300, bbox_inches='tight')
        plt.close()


# 使用推荐参数运行仿真
def main():
    # 使用推荐设置创建参数
    params = SimulationParameters(
        num_hops=100,
        num_runs=1000,  # 为演示减少

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
        pdelay_interval=1000.0,
        sync_interval=125.0,
        pdelay_turnaround=0.5,
        residence_time=0.5,

        # 校正因子 - 推荐设置
        mean_link_delay_correction=0.0,
        nrr_drift_correction=0.0,
        rr_drift_correction=0.0,
        pdelayresp_sync_correction=0.0,
        mnrr_smoothing_n=3,
        mnrr_smoothing_m=1,

        # 终端站计算使用的特定跳
        end_station_hops=[10, 25, 50, 75, 100],

        # 下一次同步消息相关设置
        consider_next_sync=True,
        time_to_next_sync=125.0  # 默认使用sync_interval
    )

    # 创建并运行仿真
    sim = TimeSyncSimulation(params)
    print("Running simulation with recommended parameters...")
    sim.run_simulation()

    # 输出结果
    max_te = max(sim.results['te_max'])
    final_7sigma = sim.results['te_7sigma'][-1]

    print(f"Simulation complete!")
    print(f"Maximum te: {max_te:.1f} ns")
    print(f"7-sigma te at hop {params.num_hops}: {final_7sigma:.1f} ns")
    print(f"Target (<1000 ns): {'PASSED' if final_7sigma < 1000 else 'FAILED'}")

    # 绘制结果
    print("Generating plots...")
    sim.plot_results()
    print(f"Results saved to '{sim.output_data_dir}' and '{sim.output_image_dir}' directories.")


if __name__ == "__main__":
    main()