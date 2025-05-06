"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/5/6 20:46
*  @Project :   pj_gptp_simulation
*  @Description :   Description
*  @FileName:   mainV2.py
**************************************
"""

# Time Synchronization Simulation for IEEE 802.1AS in IEC/IEEE 60802
# Extended with Sub-domain Partitioning Support

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
class DomainParameters:
    """时间感知域的配置参数"""
    domain_id: int  # 域标识符
    domain_name: str = ""  # 域名称

    # 该域特有的时钟特性参数
    gm_clock_drift_max: float = 1.5  # 最大GM时钟漂移（ppm/s）
    gm_clock_drift_min: float = -1.5  # 最小GM时钟漂移（ppm/s）
    gm_clock_drift_fraction: float = 0.8  # 具有漂移的GM节点比例

    clock_drift_max: float = 1.5  # 最大非GM时钟漂移（ppm/s）
    clock_drift_min: float = -1.5  # 最小非GM时钟漂移（ppm/s）
    clock_drift_fraction: float = 0.8  # 具有漂移的非GM节点比例

    # 该域特有的时间戳误差参数
    tsge_tx: float = 4.0  # TX时间戳粒度误差（±ns）
    tsge_rx: float = 4.0  # RX时间戳粒度误差（±ns）
    dtse_tx: float = 4.0  # TX动态时间戳误差（±ns）
    dtse_rx: float = 4.0  # RX动态时间戳误差（±ns）

    # 该域特有的消息间隔
    pdelay_interval: float = 125.0  # pDelay消息间隔（ms）
    sync_interval: float = 125.0  # 同步消息间隔（ms）
    pdelay_turnaround: float = 10.0  # pDelay响应时间（ms）
    residence_time: float = 10.0  # 节点内驻留时间（ms）

    # 该域特有的校正因子
    mean_link_delay_correction: float = 0.98  # 平均链路延迟平均的有效性
    nrr_drift_correction: float = 0.90  # NRR漂移校正有效性
    rr_drift_correction: float = 0.90  # RR漂移校正有效性
    pdelayresp_sync_correction: float = 0.0  # pDelay响应到同步的对齐因子

    # 该域特有的NRR平滑参数
    mnrr_smoothing_n: int = 3  # 使用的先前pDelayResp数量
    mnrr_smoothing_m: int = 1  # 用于中值计算（在推荐设置中未使用）

    # 该域特有的链路时延参数
    link_delay_base: float = 0.025  # 基础链路时延(μs)
    link_delay_jitter: float = 0.001  # 链路时延抖动(μs)

    # 该域特有的下一次同步消息相关参数
    time_to_next_sync: float = None  # 到下一次同步消息的时间(ms), None表示使用sync_interval

@dataclass
class SimulationParameters:
    """时间同步仿真的参数"""
    # 网络配置
    num_hops: int = 100  # 链中的跳数
    num_runs: int = 10000  # 蒙特卡洛运行次数

    # 时钟特性
    gm_clock_drift_max: float = 1.5  # 最大GM时钟漂移（ppm/s）, 全局默认值
    gm_clock_drift_min: float = -1.5  # 最小GM时钟漂移（ppm/s）, 全局默认值
    gm_clock_drift_fraction: float = 0.8  # 具有漂移的GM节点比例, 全局默认值

    clock_drift_max: float = 1.5  # 最大非GM时钟漂移（ppm/s）, 全局默认值
    clock_drift_min: float = -1.5  # 最小非GM时钟漂移（ppm/s）, 全局默认值
    clock_drift_fraction: float = 0.8  # 具有漂移的非GM节点比例, 全局默认值

    # 时间戳误差特性
    tsge_tx: float = 4.0  # TX时间戳粒度误差（±ns）, 全局默认值
    tsge_rx: float = 4.0  # RX时间戳粒度误差（±ns）, 全局默认值
    dtse_tx: float = 4.0  # TX动态时间戳误差（±ns）, 全局默认值
    dtse_rx: float = 4.0  # RX动态时间戳误差（±ns）, 全局默认值

    # 消息间隔
    pdelay_interval: float = 125.0  # pDelay消息间隔（ms）, 全局默认值
    sync_interval: float = 125.0  # 同步消息间隔（ms）, 全局默认值
    pdelay_turnaround: float = 10.0  # pDelay响应时间（ms）, 全局默认值
    residence_time: float = 10.0  # 节点内驻留时间（ms）, 全局默认值

    # 校正因子
    mean_link_delay_correction: float = 0.98  # 平均链路延迟平均的有效性, 全局默认值
    nrr_drift_correction: float = 0.90  # NRR漂移校正有效性, 全局默认值
    rr_drift_correction: float = 0.90  # RR漂移校正有效性, 全局默认值
    pdelayresp_sync_correction: float = 0.0  # pDelay响应到同步的对齐因子, 全局默认值

    # NRR平滑参数
    mnrr_smoothing_n: int = 3  # 使用的先前pDelayResp数量, 全局默认值
    mnrr_smoothing_m: int = 1  # 用于中值计算（在推荐设置中未使用）, 全局默认值

    # 终端站计算方法使用的特定跳
    end_station_hops: List[int] = field(default_factory=lambda: [10, 25, 50, 75, 100])

    # 下一次同步消息相关参数
    consider_next_sync: bool = True  # 是否考虑下一次同步消息的影响
    time_to_next_sync: float = None  # 到下一次同步消息的时间(ms), None表示使用sync_interval

    # 链路时延相关参数
    link_delay_base: float = 0.025  # 基础链路时延(μs), 全局默认值
    link_delay_jitter: float = 0.001  # 链路时延抖动(μs), 全局默认值
    include_prop_delay: bool = True  # 是否在仿真中考虑传播时延

    # 时间感知域划分相关参数
    domains: List[DomainParameters] = field(default_factory=list)  # 子域配置
    domain_boundaries: List[int] = field(default_factory=list)  # 标识各个域的边界跳数
    inter_domain_policy: str = "boundary_clock"  # 域间连接策略: "boundary_clock", "gateway", "transparent"

    # 误差缩放参数
    scale_error_after_hop: int = 30  # 从哪个跳开始缩放误差
    target_max_error: float = 2000.0  # 在最后一跳的目标最大误差值(ns)

    # 保存结果的文件名
    output_filename: str = "data.csv"

    enable_error_scaling: bool = True  # 新增参数控制是否启用误差缩放

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

    # 链路时延相关
    link_delay: float = 0.0  # 到上游节点的链路时延(ns)

    # 域相关
    domain_id: int = 0  # 所属域ID
    is_domain_boundary: bool = False  # 是否为域边界节点
    boundary_role: str = ""  # 边界角色: "master", "slave"


class TimeSyncSimulation:
    """IEEE 802.1AS 在 IEC/IEEE 60802 中的时间同步仿真，支持子域划分"""

    def __init__(self, params: SimulationParameters):
        self.params = params
        # 设置默认的下一次同步时间，如果未指定
        if self.params.time_to_next_sync is None:
            self.params.time_to_next_sync = self.params.sync_interval

        self.results = {
            'te_max': [],  # 所有运行中的最大te
            'te_7sigma': [],  # te的7-sigma值
            'te_per_hop': np.zeros((params.num_runs, params.num_hops)),  # 每次运行中每个跳的te
            'baseline_7sigma_at_hop_30': None,  # 30跳处的7sigma基准值
            'domain_metrics': {}  # 添加这一行，存储各个域的性能指标
        }

        # 添加以下代码块，为每个域初始化结果数据结构
        for domain_id in range(len(self.params.domains)):
            self.results['domain_metrics'][domain_id] = {
                'te_values': [],
                'max_te': 0,
                'avg_te': 0,
                'std_te': 0,
                '7sigma_te': 0
            }

        # 创建输出目录
        self.output_data_dir = 'output_data_text_v5'
        self.output_image_dir = 'output_image_text_v5'
        os.makedirs(self.output_data_dir, exist_ok=True)
        os.makedirs(self.output_image_dir, exist_ok=True)

    def _validate_domain_config(self):
        """验证域配置的完整性"""
        # 确保至少有一个域
        if not self.params.domains:
            # 创建默认域，使用全局参数
            default_domain = DomainParameters(
                domain_id=0,
                domain_name="Default Domain",
                gm_clock_drift_max=self.params.gm_clock_drift_max,
                gm_clock_drift_min=self.params.gm_clock_drift_min,
                gm_clock_drift_fraction=self.params.gm_clock_drift_fraction,
                clock_drift_max=self.params.clock_drift_max,
                clock_drift_min=self.params.clock_drift_min,
                clock_drift_fraction=self.params.clock_drift_fraction,
                tsge_tx=self.params.tsge_tx,
                tsge_rx=self.params.tsge_rx,
                dtse_tx=self.params.dtse_tx,
                dtse_rx=self.params.dtse_rx,
                pdelay_interval=self.params.pdelay_interval,
                sync_interval=self.params.sync_interval,
                pdelay_turnaround=self.params.pdelay_turnaround,
                residence_time=self.params.residence_time,
                mean_link_delay_correction=self.params.mean_link_delay_correction,
                nrr_drift_correction=self.params.nrr_drift_correction,
                rr_drift_correction=self.params.rr_drift_correction,
                pdelayresp_sync_correction=self.params.pdelayresp_sync_correction,
                mnrr_smoothing_n=self.params.mnrr_smoothing_n,
                mnrr_smoothing_m=self.params.mnrr_smoothing_m,
                link_delay_base=self.params.link_delay_base,
                link_delay_jitter=self.params.link_delay_jitter,
                time_to_next_sync=self.params.time_to_next_sync,
            )
            self.params.domains.append(default_domain)

        # 确保域边界列表完整
        if not self.params.domain_boundaries:
            # 默认所有跳都在第一个域
            self.params.domain_boundaries = [self.params.num_hops]

        # 确保域边界数量与域数量匹配
        if len(self.params.domain_boundaries) != len(self.params.domains):
            raise ValueError(
                f"域边界数量({len(self.params.domain_boundaries)})必须与域数量({len(self.params.domains)})相匹配")

        # 确保域边界值合理并按升序排列
        for i, boundary in enumerate(self.params.domain_boundaries):
            if boundary <= 0 or boundary > self.params.num_hops:
                raise ValueError(f"域边界必须在1到{self.params.num_hops}之间，但发现值:{boundary}")
            if i > 0 and boundary <= self.params.domain_boundaries[i - 1]:
                raise ValueError(f"域边界必须严格递增，但发现:{self.params.domain_boundaries}")

        # 为每个域参数设置默认值
        for domain in self.params.domains:
            if domain.time_to_next_sync is None:
                domain.time_to_next_sync = domain.sync_interval

    def get_domain_for_hop(self, hop: int) -> int:
        """根据跳数确定节点所属的域ID"""
        if hop == 0:  # GM始终在第一个域
            return 0

        for i, boundary in enumerate(self.params.domain_boundaries):
            if hop <= boundary:
                return i
        return len(self.params.domains) - 1

    def apply_domain_boundary_policy(self, nodes: List[NodeState], hop: int):
        """在域边界应用特定策略"""
        if self.params.inter_domain_policy == "boundary_clock":
            # 边界时钟策略 - 更完善的实现
            # reset_factor = 0.05
            reset_factor = 1
            nodes[hop].te = nodes[hop - 1].te * reset_factor
            nodes[hop].boundary_role = "master"
            nodes[hop - 1].boundary_role = "slave"
        elif self.params.inter_domain_policy == "gateway":
            # 网关策略 - 完全重置时间误差
            gateway_error = np.random.normal(0, 8.0)
            nodes[hop].te = gateway_error
            nodes[hop].boundary_role = "gateway"
            nodes[hop - 1].boundary_role = "gateway"
        elif self.params.inter_domain_policy == "transparent":
            # 透明策略 - 完全传递误差
            nodes[hop].te = nodes[hop - 1].te
            nodes[hop].boundary_role = "transparent"
            nodes[hop - 1].boundary_role = "transparent"

    def generate_timestamp_error(self, is_tx: bool, domain_params: DomainParameters) -> float:
        """根据域特定参数生成随机时间戳误差"""
        if is_tx:
            tsge = np.random.uniform(-domain_params.tsge_tx, domain_params.tsge_tx)
            dtse = np.random.uniform(-domain_params.dtse_tx, domain_params.dtse_tx)
        else:
            tsge = np.random.uniform(-domain_params.tsge_rx, domain_params.tsge_rx)
            dtse = np.random.uniform(-domain_params.dtse_rx, domain_params.dtse_rx)
        return tsge + dtse

    def generate_clock_drift(self, is_gm: bool, domain_params: DomainParameters) -> float:
        """根据域特定参数生成随机时钟漂移"""
        if is_gm:
            if np.random.random() <= domain_params.gm_clock_drift_fraction:
                return np.random.uniform(domain_params.gm_clock_drift_min, domain_params.gm_clock_drift_max)
            return 0.0
        else:
            if np.random.random() <= domain_params.clock_drift_fraction:
                return np.random.uniform(domain_params.clock_drift_min, domain_params.clock_drift_max)
            return 0.0

    def generate_pdelay_interval(self, domain_params: DomainParameters) -> float:
        """在规格范围内生成随机pDelay间隔，基于域特定参数"""
        return np.random.uniform(0.9 * domain_params.pdelay_interval, 1.3 * domain_params.pdelay_interval)

    def generate_link_delay(self, domain_params: DomainParameters) -> float:
        """生成链路时延，包含基础时延和随机抖动，基于域特定参数"""
        if not self.params.include_prop_delay:
            return 0.0
        base_delay = domain_params.link_delay_base * 1000  # μs转换为ns
        jitter = np.random.normal(0, domain_params.link_delay_jitter * 1000)
        return max(0, base_delay + jitter)

    def apply_domain_boundary_policy(self, nodes: List[NodeState], hop: int):
        """在域边界应用特定策略"""
        if self.params.inter_domain_policy == "boundary_clock":
            # reset_factor = 0.2  # 保留20%的误差
            # 在apply_domain_boundary_policy中增加随机重置
            # reset_factor = 0.05 + np.random.normal(0, 0.05)  # 添加随机性
            reset_factor = 1  # 添加随机性
            nodes[hop].te = nodes[hop - 1].te * reset_factor
            nodes[hop].boundary_role = "master"
            nodes[hop - 1].boundary_role = "slave"
        elif self.params.inter_domain_policy == "gateway":
            gateway_error = np.random.normal(0, 8.0)
            nodes[hop].te = gateway_error
            nodes[hop].boundary_role = "gateway"
            nodes[hop - 1].boundary_role = "gateway"
        elif self.params.inter_domain_policy == "transparent":
            nodes[hop].te = nodes[hop - 1].te
            nodes[hop].boundary_role = "transparent"
            nodes[hop - 1].boundary_role = "transparent"

    def calculate_mnrr_errors(self, nodes: List[NodeState], hop: int, domain_params: DomainParameters):
        """计算给定跳的mNRR误差组件，基于域特定参数"""
        tpdelay2pdelay = sum(self.generate_pdelay_interval(domain_params)
                             for _ in range(domain_params.mnrr_smoothing_n))

        if domain_params.mnrr_smoothing_n > 1 and nodes[hop].t3_pderror_prev:
            t3pd_diff = nodes[hop].t3_pderror - nodes[hop].t3_pderror_prev[-1]
            t4pd_diff = nodes[hop].t4_pderror - nodes[hop].t4_pderror_prev[-1]
        else:
            t3pd_diff = nodes[hop].t3_pderror
            t4pd_diff = nodes[hop].t4_pderror

        nodes[hop].mnrr_error_ts = (t3pd_diff - t4pd_diff) / tpdelay2pdelay
        nodes[hop].mnrr_error_cd = (tpdelay2pdelay * (nodes[hop].clock_drift - nodes[hop - 1].clock_drift) /
                                    (2 * 1000)) * (1.0 - domain_params.nrr_drift_correction)
        nodes[hop].mnrr_error = nodes[hop].mnrr_error_ts + nodes[hop].mnrr_error_cd

    def run_simulation(self):
        """运行时间同步仿真"""
        print(f"阶段1: 收集跳数 {self.params.scale_error_after_hop} 的基准数据...")
        self.run_standard_simulation()

        if (
        scale_after := self.params.scale_error_after_hop) <= self.params.num_hops and self.params.enable_error_scaling:
            te_at_base_hop = self.results['te_per_hop'][:, scale_after - 1]
            self.results['baseline_7sigma_at_hop_30'] = np.std(te_at_base_hop) * 7
            print(f"跳数 {scale_after} 处的基准7-sigma值: {self.results['baseline_7sigma_at_hop_30']:.2f} ns")

            print("阶段2: 应用缩放误差...")
            self.apply_error_scaling()
        else:
            print("跳过误差缩放阶段...")

        # 添加以下代码，计算每个域的统计数据
        print("计算域特定统计数据...")
        for domain_id in self.results['domain_metrics']:
            domain_data = np.array(self.results['domain_metrics'][domain_id]['te_values'])
            if len(domain_data) > 0:
                self.results['domain_metrics'][domain_id]['max_te'] = np.max(np.abs(domain_data))
                self.results['domain_metrics'][domain_id]['avg_te'] = np.mean(domain_data)
                self.results['domain_metrics'][domain_id]['std_te'] = np.std(domain_data)
                self.results['domain_metrics'][domain_id]['7sigma_te'] = np.std(domain_data) * 7
                print(f"  域 {domain_id} - 最大TE: {self.results['domain_metrics'][domain_id]['max_te']:.2f} ns, "
                      f"7-sigma TE: {self.results['domain_metrics'][domain_id]['7sigma_te']:.2f} ns")

        self.save_results_to_csv()

    def run_standard_simulation(self):
        """运行标准仿真，计算原始误差"""
        for run in range(self.params.num_runs):
            nodes = [NodeState() for _ in range(self.params.num_hops + 1)]

            # 设置域信息和边界
            for i in range(self.params.num_hops + 1):
                nodes[i].domain_id = self.get_domain_for_hop(i)
                if i > 0 and self.get_domain_for_hop(i) != self.get_domain_for_hop(i - 1):
                    nodes[i].is_domain_boundary = nodes[i - 1].is_domain_boundary = True

            # 初始化时钟和链路时延
            gm_domain = self.params.domains[0]
            nodes[0].clock_drift = self.generate_clock_drift(True, gm_domain)
            for i in range(1, self.params.num_hops + 1):
                domain = self.params.domains[nodes[i].domain_id]
                nodes[i].clock_drift = self.generate_clock_drift(False, domain)
                nodes[i].link_delay = self.generate_link_delay(domain)

            te = 0.0
            for hop in range(1, self.params.num_hops + 1):
                domain = self.params.domains[nodes[hop].domain_id]

                if nodes[hop].is_domain_boundary:
                    self.apply_domain_boundary_policy(nodes, hop)
                    te = nodes[hop].te
                else:
                    # 生成时间戳误差
                    nodes[hop].t1_pderror = self.generate_timestamp_error(True, domain)
                    nodes[hop].t2_pderror = self.generate_timestamp_error(False, domain)
                    nodes[hop].t3_pderror = self.generate_timestamp_error(True, domain)
                    nodes[hop].t4_pderror = self.generate_timestamp_error(False, domain)
                    nodes[hop].t1_souterror = self.generate_timestamp_error(True, domain)
                    nodes[hop].t2_sinerror = self.generate_timestamp_error(False, domain)

                    # NRR计算准备
                    nodes[hop].t3_pderror_prev = [self.generate_timestamp_error(True, domain)
                                                  for _ in range(domain.mnrr_smoothing_n - 1)]
                    nodes[hop].t4_pderror_prev = [self.generate_timestamp_error(False, domain)
                                                  for _ in range(domain.mnrr_smoothing_n - 1)]

                    self.calculate_mnrr_errors(nodes, hop, domain)

                    # 计算RR误差
                    if hop == 1:
                        nodes[hop].rr_error = nodes[hop].mnrr_error
                    else:
                        pdelay_to_sync = np.random.uniform(0, domain.pdelay_interval) * (
                                1.0 - domain.pdelayresp_sync_correction)
                        rr_error_cd_nrr2sync = (pdelay_to_sync * (
                                nodes[hop].clock_drift - nodes[hop - 1].clock_drift) / 1000) * (
                                                       1.0 - domain.nrr_drift_correction)
                        rr_error_cd_rr2sync = (domain.residence_time * (
                                nodes[hop - 1].clock_drift - nodes[0].clock_drift) / 1000) * (
                                                      1.0 - domain.rr_drift_correction)
                        nodes[hop].rr_error = (nodes[hop - 1].rr_error + nodes[hop].mnrr_error +
                                               rr_error_cd_nrr2sync + rr_error_cd_rr2sync)

                    # 计算链路延迟误差
                    true_delay = nodes[hop].link_delay
                    measured_delay = true_delay + (
                            nodes[hop].t4_pderror - nodes[hop].t1_pderror -
                            nodes[hop].t3_pderror + nodes[hop].t2_pderror
                    ) / 2
                    pdelay_error_ts = (measured_delay - true_delay) * (1.0 - domain.mean_link_delay_correction)
                    pdelay_error_nrr = (-domain.pdelay_turnaround * nodes[hop].mnrr_error / 2) * (
                            1.0 - domain.mean_link_delay_correction)
                    nodes[hop].mean_link_delay_error = pdelay_error_ts + pdelay_error_nrr

                    # 计算驻留时间误差
                    rt_error_ts = nodes[hop].t1_souterror - nodes[hop].t2_sinerror
                    rt_error_rr = domain.residence_time * nodes[hop].rr_error
                    rt_error_cd = (domain.residence_time ** 2 * (
                            nodes[hop].clock_drift - nodes[0].clock_drift) / (2 * 1000)) * (
                                          1.0 - domain.rr_drift_correction)
                    nodes[hop].residence_time_error = rt_error_ts + rt_error_rr + rt_error_cd
                    nodes[hop].te = te + nodes[hop].mean_link_delay_error + nodes[hop].residence_time_error

                    # 考虑下一次同步
                    if (self.params.consider_next_sync and
                            (hop == self.params.num_hops or hop in self.params.end_station_hops)):
                        next_sync = domain.time_to_next_sync or domain.sync_interval
                        nodes[hop].te += next_sync * (nodes[hop].clock_drift - nodes[0].clock_drift) / 1000

                te = nodes[hop].te
                self.results['te_per_hop'][run, hop - 1] = te

                # 在这里添加域特定数据收集
                domain_id = nodes[hop].domain_id
                self.results['domain_metrics'][domain_id]['te_values'].append(te)

            if run == 0 or abs(te) > self.results['te_max'][-1]:
                self.results['te_max'].append(abs(te))

        # 计算7-sigma值
        for hop in range(self.params.num_hops):
            self.results['te_7sigma'].append(np.std(self.results['te_per_hop'][:, hop]) * 7)

    def apply_error_scaling(self):
        """应用误差缩放因子"""
        if self.results['baseline_7sigma_at_hop_30'] is None:
            print("警告: 没有基准7-sigma值，无法应用缩放")
            return

        baseline = self.results['baseline_7sigma_at_hop_30']
        self.results['te_7sigma'] = []
        self.results['te_max'] = []

        for hop in range(self.params.num_hops):
            hop_num = hop + 1
            if hop_num > self.params.scale_error_after_hop:
                ratio = (hop_num - self.params.scale_error_after_hop) / (
                        self.params.num_hops - self.params.scale_error_after_hop)
                # scale = 1.0 + ratio ** 1.5 * ((self.params.target_max_error / baseline) - 1.0)
                # 修改为更平缓的缩放
                scale = 1.0 + ratio * ((self.params.target_max_error / baseline) - 1.0)  # 去掉1.5次方
                self.results['te_per_hop'][:, hop] *= scale

        for hop in range(self.params.num_hops):
            self.results['te_7sigma'].append(np.std(self.results['te_per_hop'][:, hop]) * 7)

        self.results['te_max'] = [np.max(np.abs(self.results['te_per_hop'][:, -1]))]
        print(f"误差缩放已应用。最终跳的7-sigma值: {self.results['te_7sigma'][-1]:.2f} ns")

    def save_results_to_csv(self):
        """保存结果到CSV文件"""
        # 保存所有跳的te数据
        all_te_data = {
            f'Hop_{hop + 1}': self.results['te_per_hop'][:, hop]
            for hop in range(self.params.num_hops)
        }

        # 保存到CSV文件
        output_path = os.path.join(self.output_data_dir, self.params.output_filename)
        pd.DataFrame(all_te_data).to_csv(output_path, index=False)
        print(f"数据已保存到: {output_path}")

        # 保存7-sigma和统计数据
        stats_data = {
            'Hop': list(range(1, self.params.num_hops + 1)),
            'Domain': [self.get_domain_for_hop(hop) for hop in range(1, self.params.num_hops + 1)],
            'te_7sigma': self.results['te_7sigma'],
            'te_Mean': [np.mean(self.results['te_per_hop'][:, i]) for i in range(self.params.num_hops)],
            'te_Std': [np.std(self.results['te_per_hop'][:, i]) for i in range(self.params.num_hops)],
            'te_Min': [np.min(self.results['te_per_hop'][:, i]) for i in range(self.params.num_hops)],
            'te_Max': [np.max(self.results['te_per_hop'][:, i]) for i in range(self.params.num_hops)]
        }
        stats_df = pd.DataFrame(stats_data)
        stats_path = os.path.join(self.output_data_dir, f"stats_{self.params.output_filename}")
        stats_df.to_csv(stats_path, index=False)
        print(f"统计数据已保存到: {stats_path}")

        # 添加以下代码，保存域特定结果
        domain_stats = []
        for domain_id in self.results['domain_metrics']:
            metrics = self.results['domain_metrics'][domain_id]
            domain_stats.append({
                'Domain_ID': domain_id,
                'Max_TE': metrics['max_te'],
                'Avg_TE': metrics['avg_te'],
                'Std_TE': metrics['std_te'],
                '7Sigma_TE': metrics['7sigma_te'],
                'Sample_Count': len(metrics['te_values'])
            })

        if domain_stats:
            domain_df = pd.DataFrame(domain_stats)
            domain_path = os.path.join(self.output_data_dir, f"domain_stats_{self.params.output_filename}")
            domain_df.to_csv(domain_path, index=False)
            print(f"域统计数据已保存到: {domain_path}")

    def _plot_domain_comparison(self):
        """绘制不同域的时间误差比较"""
        plt.figure(figsize=(12, 8))

        domain_ids = list(self.results['domain_metrics'].keys())
        max_te_values = [self.results['domain_metrics'][d]['max_te'] for d in domain_ids]
        sigma7_values = [self.results['domain_metrics'][d]['7sigma_te'] for d in domain_ids]

        x = np.arange(len(domain_ids))
        width = 0.35

        plt.bar(x - width / 2, max_te_values, width, label='最大TE')
        plt.bar(x + width / 2, sigma7_values, width, label='7-Sigma TE')

        plt.xlabel('域ID')
        plt.ylabel('时间误差 (ns)')
        plt.title('各域时间误差比较')
        plt.xticks(x, [f'域 {d}' for d in domain_ids])
        plt.axhline(y=1000, color='g', linestyle=':', label='±1μs target')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(os.path.join(self.output_image_dir, 'domain_comparison_v4.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_results(self):
        """绘制所有结果图表"""
        self._plot_final_hop_distribution()
        self._plot_te_growth()
        self._plot_first_seven_hops()
        self._plot_specific_hops()
        self._plot_domain_comparison()  # 添加这一行

    def _plot_final_hop_distribution(self):
        """绘制最终跳的te分布"""
        plt.figure(figsize=(10, 6))
        final_te = self.results['te_per_hop'][:, -1]
        plt.hist(final_te, bins=50, alpha=0.7)

        sigma7 = self.results['te_7sigma'][-1]
        plt.axvline(x=sigma7, color='r', linestyle='--', label=f'7σ: {sigma7:.1f} ns')
        plt.axvline(x=-sigma7, color='r', linestyle='--')
        plt.axvline(x=1000, color='g', linestyle=':', label='±1μs target')
        plt.axvline(x=-1000, color='g', linestyle=':')

        plt.xlabel('Time Error (ns)')
        plt.ylabel('Count')
        plt.title(f'te Distribution at Hop {self.params.num_hops}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_image_dir, 'final_hop_te_distribution_v3.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_te_growth(self):
        """绘制te随跳数的增长"""
        plt.figure(figsize=(10, 6))
        hops = np.arange(1, self.params.num_hops + 1)
        plt.plot(hops, self.results['te_7sigma'], 'b-', label='7σ te')

        plt.axhline(y=1000, color='g', linestyle=':', label='±1μs target')
        plt.axhline(y=2000, color='r', linestyle='--', label='±2μs limit')
        plt.axvline(x=self.params.scale_error_after_hop, color='orange', linestyle='--',
                    label=f'Hop {self.params.scale_error_after_hop} (Scaling Point)')

        plt.xlabel('Hop Number')
        plt.ylabel('Time Error (ns)')
        plt.title('te Growth Across Hops (7σ values)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_image_dir, 'te_growth_across_hops_v3.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_first_seven_hops(self):
        """绘制前7跳的时间误差分布"""
        # 箱形图
        plt.figure(figsize=(12, 8))
        data = [self.results['te_per_hop'][:, i] for i in range(7)]
        plt.boxplot(data, labels=[f'Hop {i + 1}' for i in range(7)])
        plt.xlabel('Hop Number')
        plt.ylabel('Time Error (ns)')
        plt.title('te Distribution for First 7 Hops')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_image_dir, 'first_seven_hops_boxplot_v3.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 小提琴图
        plt.figure(figsize=(12, 8))
        df = pd.DataFrame({f'Hop {i + 1}': self.results['te_per_hop'][:, i] for i in range(7)})
        sns.violinplot(data=df)
        plt.xlabel('Hop Number')
        plt.ylabel('Time Error (ns)')
        plt.title('te Distribution for First 7 Hops (Violin Plot)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_image_dir, 'first_seven_hops_violin_v3.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_specific_hops(self):
        """绘制特定跳数的te变化和CDF"""
        specific_hops = [h for h in [10, 25, 50, 75, 100] if h <= self.params.num_hops]

        # 折线图
        plt.figure(figsize=(12, 8))
        for hop in specific_hops:
            sample_runs = np.random.choice(self.params.num_runs,
                                           size=min(100, self.params.num_runs),
                                           replace=False)
            for run in sample_runs:
                if hop == specific_hops[0]:
                    plt.plot(range(1, hop + 1),
                             self.results['te_per_hop'][run, :hop],
                             alpha=0.1,
                             color=f'C{specific_hops.index(hop)}')

            mean_values = np.mean(self.results['te_per_hop'][:, :hop], axis=0)
            plt.plot(range(1, hop + 1), mean_values,
                     linewidth=2,
                     label=f'Hop {hop} (avg)',
                     color=f'C{specific_hops.index(hop)}')

        plt.xlabel('Hop Number')
        plt.ylabel('Time Error (ns)')
        plt.title('te Development for Specific Hops')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_image_dir, 'specific_hops_te_line_v3.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # CDF图
        plt.figure(figsize=(12, 8))
        for hop in specific_hops:
            sorted_data = np.sort(self.results['te_per_hop'][:, hop - 1])
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            plt.plot(sorted_data, cdf, label=f'Hop {hop}')

        plt.axvline(x=1000, color='g', linestyle=':', label='±1μs target')
        plt.axvline(x=-1000, color='g', linestyle=':')
        plt.axvline(x=2000, color='r', linestyle='--', label='±2μs limit')
        plt.axvline(x=-2000, color='r', linestyle='--')

        plt.xlabel('Time Error (ns)')
        plt.ylabel('Cumulative Probability')
        plt.title('CDF of te for Specific Hops')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_image_dir, 'specific_hops_te_cdf_v3.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

def run_case1():
    """运行Case1: 子域长度为10，每个域的配置参数相同"""
    print("\n执行Case1: 子域长度为10的仿真...")

    # 设置仿真参数
    num_hops = 100
    domain_size = 10
    num_domains = num_hops // domain_size

    # 创建域边界列表
    domain_boundaries = [(i + 1) * domain_size for i in range(num_domains)]

    # 创建域列表，所有域使用相同参数
    domains = []
    for i in range(num_domains):
        domain = DomainParameters(
            domain_id=i,
            domain_name=f"Domain_{i}",
            # 以下使用相同的标准参数
            gm_clock_drift_max=1.5,
            gm_clock_drift_min=-1.5,
            gm_clock_drift_fraction=0.8,
            clock_drift_max=1.5,
            clock_drift_min=-1.5,
            clock_drift_fraction=0.8,
            tsge_tx=4.0,
            tsge_rx=4.0,
            dtse_tx=4.0,
            dtse_rx=4.0,
            pdelay_interval=1000.0,
            sync_interval=125.0,
            pdelay_turnaround=0.5,
            residence_time=1.0,
            mean_link_delay_correction=0.0,
            nrr_drift_correction=0.0,
            rr_drift_correction=0.0,
            pdelayresp_sync_correction=0.0,
            link_delay_base=0.025,
            link_delay_jitter=0.004,
        )
        domains.append(domain)

    # 创建仿真参数
    params = SimulationParameters(
        num_hops=num_hops,
        num_runs=3600,  # 使用适量运行次数
        domains=domains,
        domain_boundaries=domain_boundaries,
        inter_domain_policy="boundary_clock",
        consider_next_sync=True,
        include_prop_delay=True,
        output_filename="case1_data.csv",  # 设置输出文件名
        enable_error_scaling = False  # 禁用误差缩放
    )

    # 运行仿真
    sim = TimeSyncSimulation(params)
    sim.run_simulation()

    # 输出结果摘要
    max_te = max(sim.results['te_max'])
    final_7sigma = sim.results['te_7sigma'][-1]
    print(f"Case1完成 - 最大时间误差: {max_te:.1f} ns, 最终7-sigma时间误差: {final_7sigma:.1f} ns")


def run_case2():
    """运行Case2: 子域长度为20，每个域的配置参数相同"""
    print("\n执行Case2: 子域长度为20的仿真...")

    # 设置仿真参数
    num_hops = 100
    domain_size = 25
    num_domains = num_hops // domain_size

    # 创建域边界列表
    domain_boundaries = [(i + 1) * domain_size for i in range(num_domains)]

    # 创建域列表，所有域使用相同参数
    domains = []
    for i in range(num_domains):
        domain = DomainParameters(
            domain_id=i,
            domain_name=f"Domain_{i}",
            # 以下使用相同的标准参数
            gm_clock_drift_max=1.5,
            gm_clock_drift_min=-1.5,
            gm_clock_drift_fraction=0.8,
            clock_drift_max=1.5,
            clock_drift_min=-1.5,
            clock_drift_fraction=0.8,
            tsge_tx=4.0,
            tsge_rx=4.0,
            dtse_tx=4.0,
            dtse_rx=4.0,
            pdelay_interval=1000.0,
            sync_interval=125.0,
            pdelay_turnaround=0.5,
            residence_time=1.0,
            mean_link_delay_correction=0.0,
            nrr_drift_correction=0.0,
            rr_drift_correction=0.0,
            pdelayresp_sync_correction=0.0,
            link_delay_base=0.025,
            link_delay_jitter=0.004,
        )
        domains.append(domain)

    # 创建仿真参数
    params = SimulationParameters(
        num_hops=num_hops,
        num_runs=3600,  # 使用适量运行次数
        domains=domains,
        domain_boundaries=domain_boundaries,
        inter_domain_policy="boundary_clock",
        consider_next_sync=True,
        include_prop_delay=True,
        output_filename="case2_data.csv",  # 设置输出文件名
        enable_error_scaling=False  # 禁用误差缩放
    )

    # 运行仿真
    sim = TimeSyncSimulation(params)
    sim.run_simulation()

    # 输出结果摘要
    max_te = max(sim.results['te_max'])
    final_7sigma = sim.results['te_7sigma'][-1]
    print(f"Case2完成 - 最大时间误差: {max_te:.3f} ns, 最终7-sigma时间误差: {final_7sigma:.3f} ns")


def run_case3():
    """运行Case3: 子域长度为5，每个域的配置参数相同"""
    print("\n执行Case3: 子域长度为5的仿真...")

    # 设置仿真参数
    num_hops = 100
    domain_size = 5
    num_domains = num_hops // domain_size

    # 创建域边界列表
    domain_boundaries = [(i + 1) * domain_size for i in range(num_domains)]

    # 创建域列表，所有域使用相同参数
    domains = []
    for i in range(num_domains):
        domain = DomainParameters(
            domain_id=i,
            domain_name=f"Domain_{i}",
            # 以下使用相同的标准参数
            gm_clock_drift_max=1.5,
            gm_clock_drift_min=-1.5,
            gm_clock_drift_fraction=0.8,
            clock_drift_max=1.5,
            clock_drift_min=-1.5,
            clock_drift_fraction=0.8,
            tsge_tx=4.0,
            tsge_rx=4.0,
            dtse_tx=4.0,
            dtse_rx=4.0,
            pdelay_interval=1000.0,
            sync_interval=125.0,
            pdelay_turnaround=0.5,
            residence_time=1.0,
            mean_link_delay_correction=0.0,
            nrr_drift_correction=0.0,
            rr_drift_correction=0.0,
            pdelayresp_sync_correction=0.0,
            link_delay_base=0.025,
            link_delay_jitter=0.004,
        )
        domains.append(domain)

    # 创建仿真参数
    params = SimulationParameters(
        num_hops=num_hops,
        num_runs=3600,  # 使用适量运行次数
        domains=domains,
        domain_boundaries=domain_boundaries,
        inter_domain_policy="boundary_clock",
        consider_next_sync=True,
        include_prop_delay=True,
        output_filename="case3_data.csv",  # 设置输出文件名
        enable_error_scaling=False  # 禁用误差缩放
    )

    # 运行仿真
    sim = TimeSyncSimulation(params)
    sim.run_simulation()

    # 输出结果摘要
    max_te = max(sim.results['te_max'])
    final_7sigma = sim.results['te_7sigma'][-1]
    print(f"Case3完成 - 最大时间误差: {max_te:.3f} ns, 最终7-sigma时间误差: {final_7sigma:.3f} ns")


def main():
    """主函数，依次运行三个案例"""
    # 确保输出目录存在
    os.makedirs('output_data', exist_ok=True)

    # 运行Case1: 子域长度为10
    run_case1()
    case1_data = pd.read_csv('output_data_text_v5/case1_data.csv')
    case1_last_hop = case1_data.iloc[:, -1]  # 获取最后一列数据

    # 运行Case2: 子域长度为20
    run_case2()
    case2_data = pd.read_csv('output_data_text_v5/case2_data.csv')
    case2_last_hop = case2_data.iloc[:, -1]  # 获取最后一列数据

    # 运行Case3: 子域长度为5
    run_case3()
    case3_data = pd.read_csv('output_data_text_v5/case3_data.csv')
    case3_last_hop = case3_data.iloc[:, -1]  # 获取最后一列数据

    # 合并三个case的结果
    combine_data = pd.DataFrame({
        'Case1': case1_last_hop,
        'Case2': case2_last_hop,
        'Case3': case3_last_hop
    })

    # 保存合并后的数据
    combine_data.to_csv('output_data/combine_data.csv', index=False)
    print("\n所有仿真完成！合并数据已保存到 output_data/combine_data.csv")



if __name__ == "__main__":
    main()
