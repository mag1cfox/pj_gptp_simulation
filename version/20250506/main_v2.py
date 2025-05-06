"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/5/6 20:53
*  @Project :   pj_gptp_simulation
*  @Description :   Description
*  @FileName:   main_v2.py
**************************************
"""
"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/4/29 11:38
*  @Project :   pj_gptp_simulation
*  @Description :   计算时间感知域划分网络的时间误差
*  @FileName:   main_test_20250429_domain_aware.py
**************************************
"""

# Time Synchronization Simulation for IEEE 802.1AS in IEC/IEEE 60802
# Extended with Time-Aware Domain Partitioning Support

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union
import random
import os
import pandas as pd


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
    domains: List[DomainParameters] = field(default_factory=list)
    domain_boundaries: List[int] = field(default_factory=list)  # 标识各个域的边界跳数
    inter_domain_policy: str = "boundary_clock"  # 域间连接策略: "boundary_clock", "gateway", "transparent"

    # 保存结果的文件名
    output_filename: str = "data.csv"


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
    """IEEE 802.1AS 在 IEC/IEEE 60802 中的时间同步仿真，支持时间感知域划分"""

    def __init__(self, params: SimulationParameters):
        self.params = params

        # 确保域参数完整
        self._validate_domain_config()

        # 设置默认的下一次同步时间，如果未指定
        if self.params.time_to_next_sync is None:
            self.params.time_to_next_sync = self.params.sync_interval

        self.results = {
            'te_max': [],  # 所有运行中的最大te
            'te_7sigma': [],  # te的7-sigma值
            'te_per_hop': np.zeros((params.num_runs, params.num_hops)),  # 每次运行中每个跳的te
            'domain_metrics': {}  # 存储各个域的性能指标
        }

        # 为每个域初始化结果数据结构
        for domain_id in range(len(self.params.domains)):
            self.results['domain_metrics'][domain_id] = {
                'te_values': [],
                'max_te': 0,
                'avg_te': 0,
                'std_te': 0,
                '7sigma_te': 0
            }

        # 创建输出目录
        self.output_data_dir = 'output_data'
        os.makedirs(self.output_data_dir, exist_ok=True)

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
        # 理论上不会执行到这里，因为前面的验证已确保所有跳都有对应的域
        return len(self.params.domains) - 1

    def get_domain_params(self, domain_id: int) -> DomainParameters:
        """获取特定域的参数"""
        return self.params.domains[domain_id]

    def generate_timestamp_error(self, is_tx: bool, domain_params: DomainParameters) -> float:
        """使用高斯分布生成随机时间戳误差，基于域特定参数"""
        if is_tx:
            tsge = np.random.normal(0, domain_params.tsge_tx)
            dtse = np.random.normal(0, domain_params.dtse_tx)
        else:
            tsge = np.random.normal(0, domain_params.tsge_rx)
            dtse = np.random.normal(0, domain_params.dtse_rx)
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

        # 转换为ns，添加随机抖动
        base_delay = domain_params.link_delay_base * 1000  # μs转换为ns
        jitter = np.random.normal(0, domain_params.link_delay_jitter * 1000)
        return max(0, base_delay + jitter)  # 确保时延不为负

    def apply_domain_boundary_policy(self, nodes: List[NodeState], hop: int):
        """在域边界应用特定策略"""
        if self.params.inter_domain_policy == "boundary_clock":
            # 边界时钟策略 - 重置时间误差但保留部分累积效应
            reset_factor = 0.2  # 保留20%的误差
            nodes[hop].te = nodes[hop - 1].te * reset_factor
            nodes[hop].boundary_role = "master"
            nodes[hop - 1].boundary_role = "slave"

        elif self.params.inter_domain_policy == "gateway":
            # 网关策略 - 完全重置时间误差，可能引入新误差
            gateway_error = np.random.normal(0, 8.0)  # 引入新的初始误差
            nodes[hop].te = gateway_error
            nodes[hop].boundary_role = "gateway"
            nodes[hop - 1].boundary_role = "gateway"

        elif self.params.inter_domain_policy == "transparent":
            # 透明策略 - 完全传递误差
            nodes[hop].te = nodes[hop - 1].te
            nodes[hop].boundary_role = "transparent"
            nodes[hop - 1].boundary_role = "transparent"

    def generate_timestamp_errors(self, nodes: List[NodeState], hop: int, domain_params: DomainParameters):
        """为指定跳生成所有时间戳误差，基于域特定参数"""
        # 生成pDelay相关时间戳误差
        nodes[hop].t1_pderror = self.generate_timestamp_error(is_tx=True, domain_params=domain_params)
        nodes[hop].t2_pderror = self.generate_timestamp_error(is_tx=False, domain_params=domain_params)
        nodes[hop].t3_pderror = self.generate_timestamp_error(is_tx=True, domain_params=domain_params)
        nodes[hop].t4_pderror = self.generate_timestamp_error(is_tx=False, domain_params=domain_params)

        # 生成同步相关时间戳误差
        nodes[hop].t1_souterror = self.generate_timestamp_error(is_tx=True, domain_params=domain_params)
        nodes[hop].t2_sinerror = self.generate_timestamp_error(is_tx=False, domain_params=domain_params)

        # 为NRR计算生成先前的时间戳
        nodes[hop].t3_pderror_prev = []
        nodes[hop].t4_pderror_prev = []
        for n in range(1, domain_params.mnrr_smoothing_n):
            nodes[hop].t3_pderror_prev.append(self.generate_timestamp_error(is_tx=True, domain_params=domain_params))
            nodes[hop].t4_pderror_prev.append(self.generate_timestamp_error(is_tx=False, domain_params=domain_params))

    def calculate_mnrr_errors(self, nodes: List[NodeState], hop: int, domain_params: DomainParameters):
        """计算给定跳的mNRR误差组件，基于域特定参数"""
        # 基于mNRR平滑计算有效pDelay间隔
        tpdelay2pdelay = 0
        for n in range(domain_params.mnrr_smoothing_n):
            tpdelay2pdelay += self.generate_pdelay_interval(domain_params)

        # 计算由时间戳引起的mNRR误差
        if domain_params.mnrr_smoothing_n > 1 and len(nodes[hop].t3_pderror_prev) >= domain_params.mnrr_smoothing_n - 1:
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
                2 * 1000)) * (1.0 - domain_params.nrr_drift_correction)

        # 总mNRR误差
        nodes[hop].mnrr_error = nodes[hop].mnrr_error_ts + nodes[hop].mnrr_error_cd

    def calculate_errors(self, nodes: List[NodeState], hop: int, domain_params: DomainParameters):
        """计算给定跳的所有误差组件，基于域特定参数"""
        # 计算NRR误差
        self.calculate_mnrr_errors(nodes, hop, domain_params)

        # 计算RR误差
        if hop == 1:
            nodes[hop].rr_error = nodes[hop].mnrr_error
            nodes[hop].rr_error_sum = nodes[hop].rr_error
        else:
            # 添加由NRR测量到同步之间的时钟漂移引起的RR误差
            pdelay_to_sync = np.random.uniform(0, domain_params.pdelay_interval) * (
                    1.0 - domain_params.pdelayresp_sync_correction)
            rr_error_cd_nrr2sync = (pdelay_to_sync * (
                    nodes[hop].clock_drift - nodes[hop - 1].clock_drift) / 1000) * (
                                           1.0 - domain_params.nrr_drift_correction)

            # 添加由上游RR计算到同步之间的时钟漂移引起的RR误差
            rr_error_cd_rr2sync = (domain_params.residence_time * (
                    nodes[hop - 1].clock_drift - nodes[0].clock_drift) / 1000) * (
                                          1.0 - domain_params.rr_drift_correction)

            # 累积RR误差
            nodes[hop].rr_error = nodes[hop - 1].rr_error + nodes[
                hop].mnrr_error + rr_error_cd_nrr2sync + rr_error_cd_rr2sync
            nodes[hop].rr_error_sum = nodes[hop].rr_error

        # 真实的链路传播时延(ns)
        true_link_delay = nodes[hop].link_delay

        # pDelay报文实际传输时的延迟计算
        req_propagation = true_link_delay  # 请求报文的传播时延
        resp_propagation = true_link_delay  # 响应报文的传播时延

        # pDelay测量结果计算（IEEE 802.1AS协议中的链路延迟计算方式）
        timestamp_errors = (
                    nodes[hop].t4_pderror - nodes[hop].t1_pderror - nodes[hop].t3_pderror + nodes[hop].t2_pderror)
        measured_link_delay = (req_propagation + resp_propagation) / 2 + timestamp_errors / 2

        # 链路延迟测量误差
        pdelay_error_ts = measured_link_delay - true_link_delay
        pdelay_error_ts *= (1.0 - domain_params.mean_link_delay_correction)

        pdelay_error_nrr = -domain_params.pdelay_turnaround * nodes[hop].mnrr_error / 2
        pdelay_error_nrr *= (1.0 - domain_params.mean_link_delay_correction)

        nodes[hop].mean_link_delay_error = pdelay_error_ts + pdelay_error_nrr

        # 计算住留时间误差
        rt_error_ts_direct = nodes[hop].t1_souterror - nodes[hop].t2_sinerror
        rt_error_rr = domain_params.residence_time * nodes[hop].rr_error
        rt_error_cd_direct = (domain_params.residence_time ** 2 * (
                nodes[hop].clock_drift - nodes[0].clock_drift) / (2 * 1000)) * (
                                     1.0 - domain_params.rr_drift_correction)

        nodes[hop].residence_time_error = rt_error_ts_direct + rt_error_rr + rt_error_cd_direct

    def run_simulation(self):
        """运行时间同步仿真"""
        for run in range(self.params.num_runs):
            # 为新的运行重置
            nodes = [NodeState() for _ in range(self.params.num_hops + 1)]  # +1 为GM

            # 设置节点的域信息
            for i in range(self.params.num_hops + 1):
                domain_id = self.get_domain_for_hop(i)
                nodes[i].domain_id = domain_id

                # 标记域边界节点
                if i > 0 and self.get_domain_for_hop(i) != self.get_domain_for_hop(i - 1):
                    nodes[i].is_domain_boundary = True
                    nodes[i - 1].is_domain_boundary = True

            # 为所有节点生成时钟漂移
            gm_domain_params = self.get_domain_params(0)  # GM始终在第一个域
            nodes[0].clock_drift = self.generate_clock_drift(is_gm=True, domain_params=gm_domain_params)  # GM

            for i in range(1, self.params.num_hops + 1):
                domain_params = self.get_domain_params(nodes[i].domain_id)
                nodes[i].clock_drift = self.generate_clock_drift(is_gm=False, domain_params=domain_params)
                nodes[i].link_delay = self.generate_link_delay(domain_params)

            # 计算所有跳的误差
            te = 0.0
            for hop in range(1, self.params.num_hops + 1):
                domain_params = self.get_domain_params(nodes[hop].domain_id)

                # 在域边界应用特定策略
                if nodes[hop].is_domain_boundary:
                    self.apply_domain_boundary_policy(nodes, hop)
                    te = nodes[hop].te  # 更新累积误差
                else:
                    # 生成时间戳误差
                    self.generate_timestamp_errors(nodes, hop, domain_params)

                    # 计算NRR和其他误差
                    self.calculate_errors(nodes, hop, domain_params)

                    # 计算总时间误差
                    nodes[hop].te = te + nodes[hop].mean_link_delay_error + nodes[hop].residence_time_error

                    # 如果需要考虑下一次同步消息的影响并且是指定跳或最后一跳
                    if self.params.consider_next_sync and (
                            hop == self.params.num_hops or hop in self.params.end_station_hops):
                        # 使用域特定的下一次同步时间
                        next_sync_time = domain_params.time_to_next_sync if domain_params.time_to_next_sync is not None else domain_params.sync_interval

                        # 计算到下一次同步到达之前积累的额外时钟漂移误差
                        additional_drift_error = (
                                next_sync_time * (nodes[hop].clock_drift - nodes[0].clock_drift) / 1000)
                        nodes[hop].te += additional_drift_error

                # 更新累积te
                te = nodes[hop].te

                # 存储结果
                self.results['te_per_hop'][run, hop - 1] = te

                # 收集域特定数据
                domain_id = nodes[hop].domain_id
                self.results['domain_metrics'][domain_id]['te_values'].append(te)

            # 计算完所有跳后，存储此次运行的最大te
            if run == 0 or abs(te) > self.results['te_max'][-1]:
                self.results['te_max'].append(abs(te))

        # 计算7-sigma te（比最大值更具统计代表性）
        for hop in range(self.params.num_hops):
            te_at_hop = self.results['te_per_hop'][:, hop]
            self.results['te_7sigma'].append(np.std(te_at_hop) * 7)

        # 计算每个域的统计数据
        for domain_id in self.results['domain_metrics']:
            domain_data = np.array(self.results['domain_metrics'][domain_id]['te_values'])
            if len(domain_data) > 0:
                self.results['domain_metrics'][domain_id]['max_te'] = np.max(np.abs(domain_data))
                self.results['domain_metrics'][domain_id]['avg_te'] = np.mean(domain_data)
                self.results['domain_metrics'][domain_id]['std_te'] = np.std(domain_data)
                self.results['domain_metrics'][domain_id]['7sigma_te'] = np.std(domain_data) * 7

        # 保存数据到CSV文件
        self.save_results_to_csv()

    def save_results_to_csv(self):
        """将所有节点的时间误差结果保存到CSV文件中"""
        # 创建一个包含所有跳的te数据的DataFrame
        all_te_data = {}
        for hop in range(1, self.params.num_hops + 1):
            hop_data = self.results['te_per_hop'][:, hop - 1]
            all_te_data[f'Hop_{hop}'] = hop_data

        df = pd.DataFrame(all_te_data)

        # 保存到CSV文件
        df.to_csv(os.path.join(self.output_data_dir, 'te_all_hops.csv'), index=False)

        # 保存到CSV文件
        output_path = os.path.join(self.output_data_dir, self.params.output_filename)
        df.to_csv(output_path, index=False)
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
        output_filename="case1_data.csv"  # 设置输出文件名
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
        output_filename="case2_data.csv"  # 设置输出文件名
    )

    # 运行仿真
    sim = TimeSyncSimulation(params)
    sim.run_simulation()

    # 输出结果摘要
    max_te = max(sim.results['te_max'])
    final_7sigma = sim.results['te_7sigma'][-1]
    print(f"Case2完成 - 最大时间误差: {max_te:.1f} ns, 最终7-sigma时间误差: {final_7sigma:.1f} ns")


def main():
    """主函数，依次运行两个案例"""
    # 确保输出目录存在
    os.makedirs('output_data', exist_ok=True)

    # 运行Case1: 子域长度为10
    run_case1()

    # 运行Case2: 子域长度为20
    run_case2()

    print("\n所有仿真完成！")


if __name__ == "__main__":
    main()