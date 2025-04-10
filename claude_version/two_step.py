"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/4/9 10:11
*  @Project :   pj_gptp_simulation
*  @Description :   IEEE 802.1AS 时间同步仿真 based on Gutierrez et al. paper
*  @FileName:   ieee8021as_simulation.py
**************************************
"""

"""
增强版 IEEE 802.1AS 时间同步仿真
- 运行时间: 600 秒
- 监控节点: 10, 25, 50, 75, 100
- 绘制时间偏差曲线图
- 完整消息集: Sync, Follow_Up, Announce, Signaling
- 支持两步法同步模式
- 导出数据到CSV文件
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import random
import enum
import time
import sys
import csv
import os
import pandas as pd
from collections import defaultdict

# 是否启用调试输出
DEBUG_ENABLED = True


def debug_print(*args, **kwargs):
    """打印调试信息，可以通过DEBUG_ENABLED全局变量控制"""
    if DEBUG_ENABLED:
        print(*args, **kwargs)
        sys.stdout.flush()  # 确保输出立即显示


class MessageType(enum.Enum):
    """IEEE 802.1AS 消息类型枚举"""
    SYNC = 0
    FOLLOW_UP = 1
    PDELAY_REQ = 2
    PDELAY_RESP = 3
    PDELAY_RESP_FOLLOW_UP = 4
    ANNOUNCE = 5
    SIGNALING = 6


class ClockIdentity:
    """时钟标识类"""

    def __init__(self, node_id):
        """
        初始化时钟标识

        参数:
            node_id: 节点标识符
        """
        self.node_id = node_id
        # 实际802.1AS中，这是从MAC地址派生的8字节值
        self.clock_identity = f"clock-{node_id:04d}"

    def __str__(self):
        return self.clock_identity


class ClockQuality:
    """时钟质量属性类"""

    def __init__(self, clock_class=248, clock_accuracy=0xFE, offset_scaled_log_variance=0xFFFF):
        """
        初始化时钟质量

        参数:
            clock_class: 时钟类别 (默认: 248)
            clock_accuracy: 时钟精度 (默认: 0xFE)
            offset_scaled_log_variance: 偏移缩放对数方差 (默认: 0xFFFF)
        """
        self.clock_class = clock_class
        self.clock_accuracy = clock_accuracy
        self.offset_scaled_log_variance = offset_scaled_log_variance

    def is_better_than(self, other):
        """
        与另一个时钟质量比较，确定哪个更好

        参数:
            other: 另一个ClockQuality实例

        返回:
            如果此时钟质量优于另一个，则返回True
        """
        if self.clock_class < other.clock_class:
            return True
        elif self.clock_class > other.clock_class:
            return False

        if self.clock_accuracy < other.clock_accuracy:
            return True
        elif self.clock_accuracy > other.clock_accuracy:
            return False

        return self.offset_scaled_log_variance < other.offset_scaled_log_variance


class Clock:
    """带漂移和粒度的时钟模型"""

    def __init__(self, node_id=0, initial_time=None, initial_drift_rate=None, max_drift_ppm=10, granularity_ns=8):
        """
        初始化带漂移的时钟

        参数:
            node_id: 节点ID（用于调试）
            initial_time: 初始时间值（默认为随机值）
            initial_drift_rate: 初始漂移率（ppm），默认为随机值
            max_drift_ppm: 最大漂移率（ppm）
            granularity_ns: 时钟粒度（纳秒）
        """
        self.node_id = node_id  # 保存节点ID，便于调试

        # 非GM时钟都有初始偏差，越远的节点初始偏差越大
        if initial_time is None:
            # 基于节点ID添加显著的初始偏移
            # 节点0(GM)为0，其他节点初始误差递增(µs为单位)
            self.local_time = (node_id * 10) * 1e-6  # 节点ID * 10µs
        else:
            self.local_time = initial_time

        if initial_drift_rate is None:
            # 在-max_drift_ppm和max_drift_ppm ppm之间随机漂移
            self.drift_rate = random.uniform(-max_drift_ppm, max_drift_ppm) * 1e-6
        else:
            self.drift_rate = initial_drift_rate * 1e-6

        # 漂移率变化率（随时间而变）
        self.drift_change_rate = random.uniform(-0.5, 0.5) * 1e-9  # 小的随机漂移变化
        self.granularity = granularity_ns * 1e-9  # 时钟粒度（秒）

        debug_print(f"初始化节点{node_id}的时钟: 初始时间={self.local_time}秒, 漂移率={self.drift_rate * 1e6} ppm")

    def update(self, elapsed_perfect_time):
        """
        基于完美时间更新时钟

        参数:
            elapsed_perfect_time: 完美时间尺度中的经过时间

        返回:
            当前本地时间
        """
        # 更新漂移率（可能随时间变化）
        self.drift_rate += self.drift_change_rate * elapsed_perfect_time

        # 使用漂移因子更新本地时间
        drift_factor = 1.0 + self.drift_rate
        self.local_time += elapsed_perfect_time * drift_factor

        # 应用粒度
        ticks = int(self.local_time / self.granularity)
        self.local_time = ticks * self.granularity

        return self.local_time

    def get_time(self):
        """获取当前本地时间"""
        return self.local_time

    def set_time(self, new_time):
        """设置本地时间"""
        self.local_time = new_time

    def get_drift_rate(self):
        """获取当前漂移率（ppm）"""
        return self.drift_rate * 1e6


class PortState(enum.Enum):
    """IEEE 802.1AS中的端口状态枚举"""
    INITIALIZING = 0
    FAULTY = 1
    DISABLED = 2
    LISTENING = 3
    PRE_MASTER = 4
    MASTER = 5
    PASSIVE = 6
    UNCALIBRATED = 7
    SLAVE = 8


class PTPMessage:
    """IEEE 802.1AS中使用的PTP消息基类"""

    def __init__(self, message_type, source_port_identity, sequence_id):
        """
        初始化PTP消息

        参数:
            message_type: 消息类型（MessageType枚举）
            source_port_identity: 源端口标识
            sequence_id: 序列标识符
        """
        self.message_type = message_type
        self.source_port_identity = source_port_identity
        self.sequence_id = sequence_id
        self.correction_field = 0.0


class SyncMessage(PTPMessage):
    """IEEE 802.1AS中的Sync消息"""

    def __init__(self, source_port_identity, sequence_id, timestamp=None):
        """
        初始化Sync消息

        参数:
            source_port_identity: 源端口标识
            sequence_id: 序列标识符
            timestamp: 源时间戳
        """
        super().__init__(MessageType.SYNC, source_port_identity, sequence_id)
        self.origin_timestamp = timestamp
        self.two_step_flag = True


class FollowUpMessage(PTPMessage):
    """IEEE 802.1AS中的Follow Up消息"""

    def __init__(self, source_port_identity, sequence_id, precise_origin_timestamp):
        """
        初始化Follow Up消息

        参数:
            source_port_identity: 源端口标识
            sequence_id: 序列标识符
            precise_origin_timestamp: 精确的源时间戳
        """
        super().__init__(MessageType.FOLLOW_UP, source_port_identity, sequence_id)
        self.precise_origin_timestamp = precise_origin_timestamp


class PdelayReqMessage(PTPMessage):
    """IEEE 802.1AS中的Pdelay请求消息"""

    def __init__(self, source_port_identity, sequence_id):
        """
        初始化Pdelay请求消息

        参数:
            source_port_identity: 源端口标识
            sequence_id: 序列标识符
        """
        super().__init__(MessageType.PDELAY_REQ, source_port_identity, sequence_id)
        self.origin_timestamp = None  # 发送时设置


class PdelayRespMessage(PTPMessage):
    """IEEE 802.1AS中的Pdelay响应消息"""

    def __init__(self, source_port_identity, sequence_id, request_receipt_timestamp, requesting_port_identity):
        """
        初始化Pdelay响应消息

        参数:
            source_port_identity: 源端口标识
            sequence_id: 序列标识符
            request_receipt_timestamp: 接收Pdelay请求的时间戳
            requesting_port_identity: 发送请求的端口标识
        """
        super().__init__(MessageType.PDELAY_RESP, source_port_identity, sequence_id)
        self.request_receipt_timestamp = request_receipt_timestamp
        self.requesting_port_identity = requesting_port_identity


class PdelayRespFollowUpMessage(PTPMessage):
    """IEEE 802.1AS中的Pdelay响应跟随消息"""

    def __init__(self, source_port_identity, sequence_id, response_origin_timestamp, requesting_port_identity):
        """
        初始化Pdelay响应跟随消息

        参数:
            source_port_identity: 源端口标识
            sequence_id: 序列标识符
            response_origin_timestamp: 发送Pdelay响应的精确时间戳
            requesting_port_identity: 发送请求的端口标识
        """
        super().__init__(MessageType.PDELAY_RESP_FOLLOW_UP, source_port_identity, sequence_id)
        self.response_origin_timestamp = response_origin_timestamp
        self.requesting_port_identity = requesting_port_identity


class AnnounceMessage(PTPMessage):
    """IEEE 802.1AS中的Announce消息"""

    def __init__(self, source_port_identity, sequence_id, gm_priority1, gm_clock_quality,
                 gm_priority2, gm_identity, steps_removed):
        """
        初始化Announce消息

        参数:
            source_port_identity: 源端口标识
            sequence_id: 序列标识符
            gm_priority1: 主时钟的优先级1
            gm_clock_quality: 主时钟的时钟质量
            gm_priority2: 主时钟的优先级2
            gm_identity: 主时钟的标识
            steps_removed: 距离主时钟的跳数
        """
        super().__init__(MessageType.ANNOUNCE, source_port_identity, sequence_id)
        self.gm_priority1 = gm_priority1
        self.gm_clock_quality = gm_clock_quality
        self.gm_priority2 = gm_priority2
        self.gm_identity = gm_identity
        self.steps_removed = steps_removed


class SignalingMessage(PTPMessage):
    """IEEE 802.1AS中的Signaling消息"""

    def __init__(self, source_port_identity, sequence_id, target_port_identity, tlv_type, tlv_value):
        """
        初始化Signaling消息

        参数:
            source_port_identity: 源端口标识
            sequence_id: 序列标识符
            target_port_identity: 目标端口标识
            tlv_type: TLV（类型-长度-值）的类型
            tlv_value: TLV的值
        """
        super().__init__(MessageType.SIGNALING, source_port_identity, sequence_id)
        self.target_port_identity = target_port_identity
        self.tlv_type = tlv_type
        self.tlv_value = tlv_value


class Port:
    """IEEE 802.1AS时间感知系统中的端口"""

    def __init__(self, node, port_number, phy_jitter_ns=8):
        """
        初始化端口

        参数:
            node: 该端口所属的时间感知系统
            port_number: 端口号
            phy_jitter_ns: 最大物理抖动（纳秒）
        """
        self.node = node
        self.port_number = port_number
        self.port_identity = f"{node.clock_identity}:{port_number}"
        self.state = PortState.INITIALIZING
        self.max_phy_jitter = phy_jitter_ns * 1e-9  # 最大物理抖动（秒）

        # 连接的对端端口
        self.peer_port = None

        # 序列计数器
        self.sync_sequence_id = 0
        self.announce_sequence_id = 0
        self.pdelay_sequence_id = 0
        self.signaling_sequence_id = 0

        # 路径延迟测量
        self.pdelay_interval = 1.0  # Pdelay间隔（1秒）
        self.next_pdelay_time = 0.0
        self.t1 = 0.0  # Pdelay_Req发送时间戳
        self.t2 = 0.0  # Pdelay_Req接收时间戳
        self.t3 = 0.0  # Pdelay_Resp发送时间戳
        self.t4 = 0.0  # Pdelay_Resp接收时间戳
        self.current_pdelay_sequence_id = None  # 当前路径延迟测量序列ID
        self.neighbor_rate_ratio = 1.0  # 邻居频率与本地频率比率
        self.mean_path_delay = 0.0  # 平均路径延迟（秒）

        # 同步相关
        self.sync_receipt_timeout = 3.0  # 同步接收超时（秒）
        self.last_sync_receipt_time = 0.0  # 最后接收同步的时间
        self.last_sync_sequence_id = None  # 最后接收的同步序列ID
        self.pending_follow_up = False  # 是否等待Follow_Up
        self.sync_receive_time = 0.0  # 同步接收时间（用于两步法）

        # Announce相关
        self.announce_interval = 1.0  # Announce间隔（1秒）
        self.next_announce_time = 0.0
        self.announce_receipt_timeout = 3.0  # Announce接收超时
        self.last_announce_receipt_time = 0.0  # 最后接收Announce的时间

        # 两步同步数据
        self.sync_send_times = {}  # 将sequence_id映射到Follow_Up的发送时间

    def connect(self, peer_port):
        """
        将此端口连接到另一个端口

        参数:
            peer_port: 要连接的端口
        """
        self.peer_port = peer_port
        peer_port.peer_port = self

    def send_message(self, message, perfect_time):
        """
        向对端端口发送消息

        参数:
            message: 要发送的消息
            perfect_time: 发送消息的完美时间

        返回:
            (到达时间, 消息)的元组，如果没有对端则为None
        """
        if not self.peer_port:
            return None

        # 计算此链路的传播延迟（固定 + 抖动）
        link_delay = 50e-9  # 基础50ns传播延迟

        # 应用发送物理抖动
        jitter_out = random.uniform(0, self.max_phy_jitter)

        # 消息到达对端的完美时间
        arrival_time = perfect_time + link_delay + jitter_out

        # 返回调度器的事件
        return (arrival_time, self.peer_port, message)

    def receive_message(self, message, perfect_time):
        """
        处理接收到的消息

        参数:
            message: 接收到的消息
            perfect_time: 接收消息的完美时间

        返回:
            此消息接收触发的新事件列表
        """
        # 应用接收物理抖动
        jitter = random.uniform(0, self.max_phy_jitter)
        reception_time = self.node.clock.get_time() + jitter

        events = []

        # 调试输出，确认消息处理
        message_type_str = message.message_type.name if hasattr(message.message_type, 'name') else str(message.message_type)
        debug_print(
            f"节点{self.node.node_id}端口{self.port_number}收到{message_type_str}消息，完美时间={perfect_time:.6f}")

        if message.message_type == MessageType.SYNC:
            new_events = self.process_sync(message, reception_time, perfect_time)
            if new_events:
                events.extend(new_events)

        elif message.message_type == MessageType.FOLLOW_UP:
            new_events = self.process_follow_up(message, reception_time, perfect_time)
            if new_events:
                events.extend(new_events)

        elif message.message_type == MessageType.PDELAY_REQ:
            new_events = self.process_pdelay_req(message, reception_time, perfect_time)
            if new_events:
                events.extend(new_events)

        elif message.message_type == MessageType.PDELAY_RESP:
            new_events = self.process_pdelay_resp(message, reception_time, perfect_time)
            if new_events:
                events.extend(new_events)

        elif message.message_type == MessageType.PDELAY_RESP_FOLLOW_UP:
            new_events = self.process_pdelay_resp_follow_up(message, reception_time, perfect_time)
            if new_events:
                events.extend(new_events)

        elif message.message_type == MessageType.ANNOUNCE:
            new_events = self.process_announce(message, reception_time, perfect_time)
            if new_events:
                events.extend(new_events)

        elif message.message_type == MessageType.SIGNALING:
            new_events = self.process_signaling(message, reception_time, perfect_time)
            if new_events:
                events.extend(new_events)

        return events

    def process_sync(self, message, reception_time, perfect_time):
        """
        处理接收到的Sync消息

        参数:
            message: Sync消息
            reception_time: 接收消息的本地时间
            perfect_time: 接收消息的完美时间

        返回:
            此消息接收触发的新事件列表
        """
        events = []

        # 更新同步接收时间
        self.last_sync_receipt_time = perfect_time
        self.last_sync_sequence_id = message.sequence_id

        debug_print(f"节点{self.node.node_id}端口{self.port_number}处理Sync消息")

        # 如果这是从其主端口接收的从端口
        if self.state == PortState.SLAVE:
            debug_print(f"节点{self.node.node_id}等待Follow_Up消息, seq={message.sequence_id}")
            self.pending_follow_up = True
            # 存储同步接收时间，供后续Follow_Up使用
            self.sync_receive_time = reception_time

        return events

    def process_follow_up(self, message, reception_time, perfect_time):
        """
        处理接收到的Follow_Up消息

        参数:
            message: Follow_Up消息
            reception_time: 接收消息的本地时间
            perfect_time: 接收消息的完美时间

        返回:
            此消息接收触发的新事件列表
        """
        events = []

        # 只处理我们预期的Follow_Up
        if not self.pending_follow_up or message.sequence_id != self.last_sync_sequence_id:
            debug_print(f"节点{self.node.node_id}端口{self.port_number}忽略意外的Follow_Up seq={message.sequence_id}")
            return events

        debug_print(f"节点{self.node.node_id}端口{self.port_number}处理Follow_Up seq={message.sequence_id}")

        # 清除挂起标志
        self.pending_follow_up = False

        # 如果这是从其主端口接收的从端口
        if self.state == PortState.SLAVE:
            # 从Follow_Up获取精确时间戳
            precise_origin_timestamp = message.precise_origin_timestamp

            # 基于消息数据计算GM时间
            gm_time = precise_origin_timestamp + message.correction_field + self.mean_path_delay
            local_time = self.node.clock.get_time()

            # 计算校正前的时间偏差
            time_deviation = local_time - gm_time

            debug_print(f"节点{self.node.node_id}两步同步偏差: {time_deviation * 1e6:.3f} µs")

            # 记录偏差（无论值大小，始终记录）
            self.node.time_deviations.append((perfect_time, time_deviation))

            # 校正本地时钟
            self.node.clock.set_time(gm_time)
            self.node.sync_receptions += 1

            # 如果启用，则将同步转发到下游端口
            if self.node.sync_locked:
                new_events = self.node.forward_follow_up(message, perfect_time, reception_time)
                if new_events:
                    events.extend(new_events)

        return events

    def process_pdelay_req(self, message, reception_time, perfect_time):
        """
        处理接收到的Pdelay_Req消息

        参数:
            message: Pdelay_Req消息
            reception_time: 接收消息的本地时间
            perfect_time: 接收消息的完美时间

        返回:
            此消息接收触发的新事件列表
        """
        events = []

        # 记录接收时间戳(t2)用于响应
        t2 = reception_time

        # 创建并发送Pdelay_Resp
        resp_message = PdelayRespMessage(
            self.port_identity,
            message.sequence_id,
            t2,
            message.source_port_identity
        )

        # 队列发送响应
        send_time = perfect_time
        resp_event = self.send_message(resp_message, send_time)
        if resp_event:
            events.append(resp_event)

        # 记录响应发送时间用于Follow_Up
        t3 = self.node.clock.get_time()

        # 创建并发送Pdelay_Resp_Follow_Up
        follow_up_message = PdelayRespFollowUpMessage(
            self.port_identity,
            message.sequence_id,
            t3,
            message.source_port_identity
        )

        # 在发送follow-up前添加小延迟
        follow_up_send_time = perfect_time + 100e-9  # Pdelay_Resp后100 ns
        follow_up_event = self.send_message(follow_up_message, follow_up_send_time)
        if follow_up_event:
            events.append(follow_up_event)

        return events

    def process_pdelay_resp(self, message, reception_time, perfect_time):
        """
        处理接收到的Pdelay_Resp消息

        参数:
            message: Pdelay_Resp消息
            reception_time: 接收消息的本地时间
            perfect_time: 接收消息的完美时间

        返回:
            此消息接收触发的新事件列表
        """
        events = []

        # 只处理对我们请求的响应
        if message.requesting_port_identity != self.port_identity:
            return events

        # 检查这是否与我们当前的路径延迟测量匹配
        if message.sequence_id != self.current_pdelay_sequence_id:
            return events

        # 记录t4（响应接收时间）
        self.t4 = reception_time

        # 存储来自消息的t2
        self.t2 = message.request_receipt_timestamp

        return events

    def process_pdelay_resp_follow_up(self, message, reception_time, perfect_time):
        """
        处理接收到的Pdelay_Resp_Follow_Up消息

        参数:
            message: Pdelay_Resp_Follow_Up消息
            reception_time: 接收消息的本地时间
            perfect_time: 接收消息的完美时间

        返回:
            此消息接收触发的新事件列表
        """
        events = []

        # 只处理对我们请求的响应
        if message.requesting_port_identity != self.port_identity:
            return events

        # 检查这是否与我们当前的路径延迟测量匹配
        if message.sequence_id != self.current_pdelay_sequence_id:
            return events

        # 存储来自消息的t3
        self.t3 = message.response_origin_timestamp

        # 计算路径延迟
        self.calculate_path_delay(perfect_time)

        return events

    def calculate_path_delay(self, perfect_time):
        """
        基于收集的时间戳计算平均路径延迟

        参数:
            perfect_time: 执行计算的完美时间
        """
        # 计算邻居速率比（简化）
        # 在实际系统中，这将随时间测量
        if self.peer_port:
            self.neighbor_rate_ratio = (1 + self.peer_port.node.clock.get_drift_rate() * 1e-6) / \
                                       (1 + self.node.clock.get_drift_rate() * 1e-6)

        # 添加邻居速率比误差（根据标准，最高0.1 ppm）
        nr_error = random.uniform(-0.1, 0.1) * 1e-6
        self.neighbor_rate_ratio += nr_error

        # 使用IEEE 802.1AS中的公式(4)计算传播延迟
        path_delay = 0.5 * ((self.t4 - self.t1) - self.neighbor_rate_ratio * (self.t3 - self.t2))

        # 使用低通滤波更新平均路径延迟
        alpha = 0.1  # 滤波系数
        self.mean_path_delay = (1 - alpha) * self.mean_path_delay + alpha * path_delay

        debug_print(f"节点{self.node.node_id}端口{self.port_number}计算路径延迟: {self.mean_path_delay * 1e9:.2f} ns")

        # 重置当前序列ID
        self.current_pdelay_sequence_id = None

    def initiate_pdelay_req(self, perfect_time):
        """
        发起路径延迟请求测量

        参数:
            perfect_time: 发起测量的完美时间

        返回:
            调度器的事件或None
        """
        # 只在有对端时测量
        if not self.peer_port:
            self.next_pdelay_time = perfect_time + self.pdelay_interval
            return None

        # 创建Pdelay_Req消息
        self.pdelay_sequence_id += 1
        req_message = PdelayReqMessage(
            self.port_identity,
            self.pdelay_sequence_id
        )

        # 记录t1（请求发送时间）
        self.t1 = self.node.clock.get_time()
        self.current_pdelay_sequence_id = self.pdelay_sequence_id

        debug_print(f"节点{self.node.node_id}端口{self.port_number}发起Pdelay请求, seq={self.pdelay_sequence_id}")

        # 安排下一次pdelay测量
        self.next_pdelay_time = perfect_time + self.pdelay_interval

        # 发送请求
        return self.send_message(req_message, perfect_time)

    def process_announce(self, message, reception_time, perfect_time):
        """
        处理接收到的Announce消息

        参数:
            message: Announce消息
            reception_time: 接收消息的本地时间
            perfect_time: 接收消息的完美时间

        返回:
            此消息接收触发的新事件列表
        """
        # 更新announce接收时间
        self.last_announce_receipt_time = perfect_time

        # 由于我们忽略BMCA，返回空事件列表
        return []

    def process_signaling(self, message, reception_time, perfect_time):
        """
        处理接收到的Signaling消息

        参数:
            message: Signaling消息
            reception_time: 接收消息的本地时间
            perfect_time: 接收消息的完美时间

        返回:
            此消息接收触发的新事件列表
        """
        # 处理信令消息中的TLV
        # （在完整实现中，这将处理各种TLV）
        return []

    def initiate_announce(self, perfect_time):
        """
        发起Announce消息

        参数:
            perfect_time: 发起Announce的完美时间

        返回:
            调度器的事件或None
        """
        # 仅当此端口处于MASTER状态时发送Announce
        if self.state != PortState.MASTER:
            self.next_announce_time = perfect_time + self.announce_interval
            return None

        # 创建具有当前时间属性数据集的Announce消息
        self.announce_sequence_id += 1
        announce_message = AnnounceMessage(
            self.port_identity,
            self.announce_sequence_id,
            self.node.priority1,
            self.node.clock_quality,
            self.node.priority2,
            self.node.current_gm_identity,
            self.node.steps_removed
        )

        debug_print(f"节点{self.node.node_id}端口{self.port_number}发起Announce消息, seq={self.announce_sequence_id}")

        # 安排下一个announce
        self.next_announce_time = perfect_time + self.announce_interval

        # 发送announce
        return self.send_message(announce_message, perfect_time)


class TimeAwareSystem:
    """IEEE 802.1AS网络中的时间感知系统实现"""

    def __init__(self, node_id, is_grandmaster=False, priority1=248, priority2=248, phy_jitter_ns=8):
        """
        初始化时间感知系统

        参数:
            node_id: 节点标识符
            is_grandmaster: 如果此节点是主时钟则为True
            priority1: BMCA的Priority1值
            priority2: BMCA的Priority2值
            phy_jitter_ns: 最大物理抖动（纳秒）
        """
        self.node_id = node_id
        self.is_grandmaster = is_grandmaster

        # 使用节点ID初始化时钟（用于初始化）
        if is_grandmaster:
            # GM时钟应该是准确的
            self.clock = Clock(node_id=node_id, initial_time=0.0, initial_drift_rate=0.0)
        else:
            # 非GM时钟有随机初始偏差和漂移
            self.clock = Clock(node_id=node_id)

        self.clock_identity = ClockIdentity(node_id)
        self.max_phy_jitter = phy_jitter_ns * 1e-9  # 最大物理抖动（秒）

        # BMCA相关属性
        self.priority1 = 128 if is_grandmaster else priority1
        self.priority2 = 128 if is_grandmaster else priority2
        self.clock_quality = ClockQuality(
            clock_class=6 if is_grandmaster else 248,
            clock_accuracy=0x20 if is_grandmaster else 0xFE,
            offset_scaled_log_variance=0x4100 if is_grandmaster else 0xFFFF
        )

        # 当前时间属性数据集
        self.current_gm_identity = str(self.clock_identity) if is_grandmaster else None
        self.steps_removed = 0 if is_grandmaster else 255

        # 每个节点创建两个端口，除了首尾节点可能只有一个端口
        if is_grandmaster or node_id == 0:  # GM只需要一个下游端口
            self.ports = [Port(self, 1)]
        else:
            # 所有非GM节点具有上游和下游端口
            # 端口1：上游接收同步 (SLAVE)
            # 端口2：下游传播同步 (MASTER)
            self.ports = [Port(self, 1), Port(self, 2)]

        # 同步配置
        self.sync_locked = True  # syncLocked标志（True为更好的精度）
        self.sync_interval = 31.25e-3  # syncInterval（31.25毫秒）- 根据论文
        self.two_step_pdelay = True  # 使用两步pdelay模式

        # 选定的端口角色
        self.slave_port = None  # 从上游接收时间的端口

        # 用于统计和分析
        self.time_deviations = []  # 记录与GM的时间偏差
        self.sync_receptions = 0  # 接收的同步消息计数

        debug_print(f"初始化时间感知系统: 节点{node_id}, GM={is_grandmaster}")

    def record_time_deviation(self, perfect_time, deviation):
        """
        记录用于分析的时间偏差

        参数:
            perfect_time: 测量偏差的完美时间
            deviation: 与主时钟的时间偏差
        """
        self.time_deviations.append((perfect_time, deviation))

    def initiate_sync(self, perfect_time):
        """
        发起同步消息（仅适用于主时钟或处于MASTER状态的端口）

        参数:
            perfect_time: 发起同步的完美时间

        返回:
            要处理的同步事件列表
        """
        sync_events = []

        # 对于主时钟或具有主端口的节点
        for port in self.ports:
            if port.state == PortState.MASTER:
                debug_print(f"节点{self.node_id}端口{port.port_number}发起同步")

                # 创建Sync消息
                port.sync_sequence_id += 1
                sync_message = SyncMessage(
                    port.port_identity,
                    port.sync_sequence_id,
                    None
                )

                # 存储发送时间，用于Follow_Up
                port.sync_send_times[port.sync_sequence_id] = self.clock.get_time()
                debug_print(f"存储Follow_Up的发送时间: seq={port.sync_sequence_id}")

                # 发送同步
                sync_event = port.send_message(sync_message, perfect_time)
                if sync_event:
                    debug_print(f"生成同步事件到节点{sync_event[1].node.node_id}的端口{sync_event[1].port_number}")
                    sync_events.append(sync_event)

                # 创建并发送Follow_Up
                follow_up_message = FollowUpMessage(
                    port.port_identity,
                    port.sync_sequence_id,
                    port.sync_send_times[port.sync_sequence_id]
                )

                # 在发送follow-up前添加小延迟
                follow_up_send_time = perfect_time + 100e-9  # 同步后100 ns
                follow_up_event = port.send_message(follow_up_message, follow_up_send_time)
                if follow_up_event:
                    debug_print(
                        f"生成follow-up事件到节点{follow_up_event[1].node.node_id}的端口{follow_up_event[1].port_number}")
                    sync_events.append(follow_up_event)

                # 清理存储的发送时间
                del port.sync_send_times[port.sync_sequence_id]

        return sync_events

    def forward_sync(self, sync_message, perfect_time, reception_time):
        """
        将同步消息转发到下游端口

        参数:
            sync_message: 要转发的Sync消息
            perfect_time: 转发消息的完美时间
            reception_time: 接收消息的本地时间

        返回:
            要处理的同步事件列表
        """
        sync_events = []

        # 计算驻留时间
        residence_time = min(random.uniform(0, 1e-3), 1e-3)  # 最多1毫秒驻留时间

        # 转发到所有MASTER端口
        for port in self.ports:
            if port.state == PortState.MASTER:
                # 创建一个新的Sync消息
                port.sync_sequence_id += 1
                debug_print(f"节点{self.node.node_id}通过端口{port.port_number}转发Sync消息, seq={port.sync_sequence_id}")

                # 创建两步Sync
                forwarded_sync = SyncMessage(
                    port.port_identity,
                    port.sync_sequence_id,
                    None
                )
                forwarded_sync.correction_field = sync_message.correction_field

                # 存储发送时间，用于Follow_Up
                port.sync_send_times[port.sync_sequence_id] = self.clock.get_time()

                # 发送同步
                sync_event = port.send_message(forwarded_sync, perfect_time + residence_time)
                if sync_event:
                    debug_print(f"生成转发Sync事件到节点{sync_event[1].node.node_id}的端口{sync_event[1].port_number}")
                    sync_events.append(sync_event)

        return sync_events

    def forward_follow_up(self, follow_up_message, perfect_time, reception_time):
        """
        将Follow_Up消息转发到下游端口

        参数:
            follow_up_message: 要转发的Follow_Up消息
            perfect_time: 转发消息的完美时间
            reception_time: 接收消息的本地时间

        返回:
            要处理的事件列表
        """
        events = []

        # 计算驻留时间
        residence_time = min(random.uniform(0, 1e-4), 1e-4)  # 最多100微秒驻留时间

        # 转发到所有MASTER端口
        for port in self.ports:
            if port.state == PortState.MASTER and port.sync_sequence_id in port.sync_send_times:
                debug_print(f"节点{self.node.node_id}通过端口{port.port_number}转发Follow_Up消息")
                # 创建一个新的Follow_Up消息
                follow_up = FollowUpMessage(
                    port.port_identity,
                    port.sync_sequence_id,
                    port.sync_send_times[port.sync_sequence_id]
                )

                # 更新校正字段
                follow_up.correction_field = follow_up_message.correction_field + \
                                             self.slave_port.mean_path_delay + \
                                             (residence_time * self.get_rate_ratio())

                # 发送Follow_Up
                event = port.send_message(follow_up, perfect_time + residence_time)
                if event:
                    debug_print(f"生成转发Follow_Up事件到节点{event[1].node.node_id}的端口{event[1].port_number}")
                    events.append(event)

                # 清理存储的发送时间
                del port.sync_send_times[port.sync_sequence_id]

        return events

    def get_rate_ratio(self):
        """获取从GM到此节点的累积速率比"""
        if self.is_grandmaster or self.slave_port is None:
            return 1.0

        # 在实际实现中，这将基于观察到的频率比来计算
        rate_ratio = 1.0
        if self.slave_port:
            rate_ratio = self.slave_port.neighbor_rate_ratio

        return rate_ratio


class IEEE8021ASSimulation:
    """具有完整消息集的IEEE 802.1AS网络仿真"""

    def __init__(self, num_nodes=100, simulation_time=600.0):
        """
        初始化仿真

        参数:
            num_nodes: 网络中的时间感知系统数量
            simulation_time: 总仿真时间（秒）
        """
        self.num_nodes = num_nodes + 1  # +1是因为节点从0开始计数，这样才能有num_nodes个节点
        self.simulation_time = simulation_time
        self.perfect_time = 0.0
        self.nodes = []
        self.events = []  # (time, event_type, params)

        # 创建节点
        for i in range(self.num_nodes):
            is_gm = (i == 0)
            node = TimeAwareSystem(
                i,
                is_grandmaster=is_gm,
                priority1=128 if is_gm else 248,
                priority2=128 if is_gm else 248
            )
            self.nodes.append(node)

        # 按线性拓扑连接节点
        self._setup_linear_topology()

    def _setup_linear_topology(self):
        """设置相邻节点之间连接的线性拓扑"""
        # GM (节点0) 和第一个节点
        self.nodes[0].ports[0].connect(self.nodes[1].ports[0])
        self.nodes[0].ports[0].state = PortState.MASTER
        self.nodes[1].ports[0].state = PortState.SLAVE
        self.nodes[1].slave_port = self.nodes[1].ports[0]

        # 连接中间节点
        for i in range(1, self.num_nodes - 1):
            # 确保节点有两个端口
            if len(self.nodes[i].ports) >= 2:
                # 将节点i的下游端口(1)连接到节点i+1的上游端口(0)
                self.nodes[i].ports[1].connect(self.nodes[i+1].ports[0])
                self.nodes[i].ports[1].state = PortState.MASTER
                self.nodes[i+1].ports[0].state = PortState.SLAVE
                self.nodes[i+1].slave_port = self.nodes[i+1].ports[0]
            else:
                debug_print(f"警告: 节点{i}没有足够的端口来连接下游节点")

    def run(self):
        """运行仿真"""
        # 初始化事件
        self._schedule_initial_events()

        # 添加进度报告
        last_progress_time = 0
        progress_interval = 10  # 每10秒报告一次进度

        # 处理事件
        debug_print(f"启动仿真，初始事件数量：{len(self.events)}")
        while self.events and self.perfect_time < self.simulation_time:
            # 获取下一个事件
            event_time, event_type, params = self.events.pop(0)

            # 更新完美时间
            dt = event_time - self.perfect_time
            if dt > 0:
                self._update_clocks(dt)
                self.perfect_time = event_time

                # 报告进度
                if int(self.perfect_time) > last_progress_time + progress_interval:
                    last_progress_time = int(self.perfect_time)
                    debug_print(
                        f"仿真进度: {self.perfect_time:.1f}秒 / {self.simulation_time:.1f}秒 ({(self.perfect_time / self.simulation_time) * 100:.1f}%)")
                    # 报告当前事件队列大小
                    debug_print(f"  事件队列大小: {len(self.events)}个事件")

            # 处理事件
            if event_type == "message":
                self._process_message_event(*params)
            elif event_type == "initiate_sync":
                self._process_initiate_sync(params)
            elif event_type == "initiate_announce":
                self._process_initiate_announce(params)
            elif event_type == "initiate_pdelay":
                self._process_initiate_pdelay(params)

        debug_print(f"仿真在时间{self.perfect_time:.1f}秒完成")
        # 收集和分析结果
        return self._analyze_results()

    def _schedule_initial_events(self):
        """安排初始事件"""
        # 安排来自主时钟和所有主端口的初始同步
        for i in range(self.num_nodes):
            for port in self.nodes[i].ports:
                if port.state == PortState.MASTER:
                    # 初始化同步
                    self.events.append((0.0, "initiate_sync", (i, port.port_number)))

                    # 初始化Announce
                    announce_time = random.uniform(0, 0.1)  # 略微随机化announce时间
                    self.events.append((announce_time, "initiate_announce", (i, port.port_number)))

        # 安排初始传播延迟测量
        for i in range(self.num_nodes):
            for port in self.nodes[i].ports:
                # 错开初始pdelay测量
                pdelay_time = random.uniform(0, 1.0)
                self.events.append((pdelay_time, "initiate_pdelay", (i, port.port_number)))

    def _update_clocks(self, dt):
        """
        基于经过的完美时间更新所有时钟

        参数:
            dt: 经过的完美时间（秒）
        """
        for node in self.nodes:
            node.clock.update(dt)

    def _process_message_event(self, port, message):
        """处理消息接收事件"""
        # 获取接收节点和端口
        receive_port = port
        node = receive_port.node

        message_type_str = message.message_type.name if hasattr(message.message_type, 'name') else str(message.message_type)
        debug_print(f"处理消息: 节点{node.node_id}端口{receive_port.port_number}接收{message_type_str}消息")

        # 处理消息
        new_events = receive_port.receive_message(message, self.perfect_time)

        # 安排任何新事件
        if new_events:
            debug_print(f"  -> 生成{len(new_events)}个新事件")
            for event in new_events:
                arrival_time, target_port, new_message = event
                new_message_type = new_message.message_type.name if hasattr(new_message.message_type, 'name') else str(new_message.message_type)
                debug_print(f"  -> 发送{new_message_type}消息到节点{target_port.node.node_id}端口{target_port.port_number}，到达时间={arrival_time:.6f}")
                self._insert_event((arrival_time, "message", (target_port, new_message)))
        else:
            debug_print(f"  -> 未生成新事件")

    def _process_initiate_sync(self, params):
        """处理同步发起事件"""
        node_id, port_number = params
        node = self.nodes[node_id]

        debug_print(f"发起同步: 节点{node_id}端口{port_number}")

        # 发起同步
        sync_events = node.initiate_sync(self.perfect_time)

        # 安排任何新事件
        if sync_events:
            debug_print(f"  -> 生成{len(sync_events)}个同步事件")
            for event in sync_events:
                arrival_time, target_port, message = event
                message_type = message.message_type.name if hasattr(message.message_type, 'name') else str(message.message_type)
                debug_print(f"  -> 发送{message_type}消息到节点{target_port.node.node_id}端口{target_port.port_number}，到达时间={arrival_time:.6f}")
                self._insert_event((arrival_time, "message", (target_port, message)))
        else:
            debug_print(f"  -> 未生成同步事件")

        # 为所有主端口安排下一次同步发起
        for port in node.ports:
            if port.state == PortState.MASTER:
                next_sync_time = self.perfect_time + node.sync_interval
                self._insert_event((next_sync_time, "initiate_sync", (node_id, port.port_number)))

    def _process_initiate_announce(self, params):
        """处理Announce发起事件"""
        node_id, port_number = params
        node = self.nodes[node_id]
        port = node.ports[port_number - 1]  # 端口号从1开始

        # 发起announce
        announce_event = port.initiate_announce(self.perfect_time)

        # 安排任何新事件
        if announce_event:
            arrival_time, target_port, message = announce_event
            self._insert_event((arrival_time, "message", (target_port, message)))

        # 安排下一个announce
        next_announce_time = port.next_announce_time
        self._insert_event((next_announce_time, "initiate_announce", (node_id, port_number)))

    def _process_initiate_pdelay(self, params):
        """处理pdelay发起事件"""
        node_id, port_number = params
        node = self.nodes[node_id]
        port = node.ports[port_number - 1]  # 端口号从1开始

        # 发起pdelay
        pdelay_event = port.initiate_pdelay_req(self.perfect_time)

        # 安排任何新事件
        if pdelay_event:
            arrival_time, target_port, message = pdelay_event
            self._insert_event((arrival_time, "message", (target_port, message)))

        # 安排下一个pdelay
        next_pdelay_time = port.next_pdelay_time
        self._insert_event((next_pdelay_time, "initiate_pdelay", (node_id, port_number)))

    def _insert_event(self, event):
        """按正确的时间顺序将事件插入事件队列"""
        time, event_type, params = event
        index = 0
        while index < len(self.events) and self.events[index][0] < time:
            index += 1
        self.events.insert(index, event)

    def _analyze_results(self):
        """分析仿真结果"""
        results = {
            "node_deviations": {},
            "node_deviations_time_series": {},  # 添加时间序列数据
            "propagation_delays": [],
            "sync_receptions": []
        }

        # 打印所有节点的同步接收计数和偏差记录数
        debug_print("\n节点数据摘要:")
        for i, node in enumerate(self.nodes):
            debug_print(f"节点{i}: 同步接收={node.sync_receptions}, 时间偏差记录={len(node.time_deviations)}")

        # 收集每个节点的时间偏差
        monitored_nodes = [10, 25, 50, 75, 100]
        debug_print("\n收集监控节点的数据:")

        for i, node in enumerate(self.nodes):
            if i > 0 and i in monitored_nodes:  # 仅收集监控节点的数据
                deviations = [dev for _, dev in node.time_deviations]
                results["node_deviations"][i] = deviations
                # 存储时间序列数据
                results["node_deviations_time_series"][i] = node.time_deviations
                results["sync_receptions"].append(node.sync_receptions)

                debug_print(f"监控节点{i}: 收集了{len(deviations)}条偏差记录")
                if deviations:
                    debug_print(f"  偏差样本(µs): {[d * 1e6 for d in deviations[:5]]}")

        # 收集传播延迟
        for i in monitored_nodes:
            if i < self.num_nodes:
                for port in self.nodes[i].ports:
                    if port.state == PortState.SLAVE:
                        results["propagation_delays"].append(port.mean_path_delay)
                        debug_print(f"节点{i}路径延迟: {port.mean_path_delay * 1e9:.2f} ns")
                        break

        return results


def analyze_sync_precision(results, threshold_us=1.0):
    """
    从仿真结果分析同步精度

    参数:
        results: 仿真结果
        threshold_us: 精度阈值（微秒）

    返回:
        具有分析结果的字典
    """
    analysis = {
        "max_deviations": {},
        "sync_probabilities": {},
        "overall_precision": 0.0
    }

    # 计算每个节点的最大偏差
    for node_id, deviations in results["node_deviations"].items():
        if deviations:
            max_dev = max(abs(dev) for dev in deviations)
            analysis["max_deviations"][node_id] = max_dev

    # 计算不同阈值的同步概率
    threshold_s = threshold_us * 1e-6
    for node_id, deviations in results["node_deviations"].items():
        if deviations:
            in_sync_count = sum(1 for dev in deviations if abs(dev) < threshold_s)
            probability = in_sync_count / len(deviations) if deviations else 0
            analysis["sync_probabilities"][node_id] = probability

    # 计算整体精度（所有节点的最大偏差）
    if analysis["max_deviations"]:
        analysis["overall_precision"] = max(analysis["max_deviations"].values())

    return analysis


def plot_time_deviations(results):
    """
    绘制指定节点随时间的时间偏差

    参数:
        results: 包含时间序列数据的仿真结果
    """
    plt.figure(figsize=(12, 8))

    # 选择要绘制的节点（10, 25, 50, 75, 100）
    nodes_to_plot = [10, 25, 50, 75, 100]

    # 为每个节点绘制时间偏差
    for node_id in nodes_to_plot:
        if node_id in results["node_deviations_time_series"]:
            time_series = results["node_deviations_time_series"][node_id]
            times = [t for t, _ in time_series]
            deviations = [d * 1e6 for _, d in time_series]  # 转换为微秒
            plt.plot(times[10:], deviations, label=f'节点 {node_id}')

    plt.xlabel('仿真时间 (秒)')
    plt.ylabel('时间偏差 (µs)')
    plt.title('各节点相对于主时钟的时间偏差')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('time_deviations_over_time.png')
    plt.show()


def save_results_to_csv(results, filename="time_deviations.csv"):
    """
    将仿真结果保存到CSV文件中，每一列代表一个节点的时间误差数据

    参数:
        results: 仿真结果
        filename: CSV文件名
    """
    # 创建一个字典，为每个节点存储时间和偏差
    data_dict = defaultdict(list)

    # 记录所有可能的时间点
    all_times = set()

    # 收集所有节点的数据
    for node_id, time_series in results["node_deviations_time_series"].items():
        for perfect_time, deviation in time_series:
            # 将偏差转换为微秒
            deviation_us = deviation * 1e6
            data_dict[f"node_{node_id}_time"].append(perfect_time)
            data_dict[f"node_{node_id}_deviation"].append(deviation_us)
            all_times.add(perfect_time)

    # 将数据转换为pandas DataFrame以便于处理
    df = pd.DataFrame()

    # 选择要导出的监控节点（10, 25, 50, 75, 100）
    monitored_nodes=[]
    for i in range(1,100):
        monitored_nodes.append(i)
    # monitored_nodes = [1,10, 25, 50, 75, 100]

    # 准备数据，按照监控节点创建DataFrame
    for node_id in monitored_nodes:
        if f"node_{node_id}_time" in data_dict:
            # 创建该节点的时间和偏差数据的DataFrame
            node_df = pd.DataFrame({
                'time': data_dict[f"node_{node_id}_time"],
                f'node_{node_id}': data_dict[f"node_{node_id}_deviation"]
            })

            # 如果是第一个节点，直接赋值给df
            if df.empty:
                df = node_df
            else:
                # 否则，基于时间列合并
                df = pd.merge(df, node_df, on='time', how='outer')

    # 确保时间列是排序的
    if not df.empty:
        df = df.sort_values('time')

        # 保存到CSV文件
        df.to_csv(filename, index=False)
        print(f"已将时间偏差数据保存到: {filename}")


def create_aligned_time_series_csv(results, filename="aligned_time_deviations.csv"):
    """
    创建时间对齐的时间序列数据并保存到CSV，每一列代表一个节点的时间误差数据

    参数:
        results: 仿真结果
        filename: CSV文件名
    """
    # 选择要导出的监控节点（10, 25, 50, 75, 100）
    monitored_nodes = [1,10, 25, 50, 75, 100]

    # 为每个节点创建一个字典，将时间映射到偏差
    node_data = {}
    for node_id in monitored_nodes:
        if node_id in results["node_deviations_time_series"]:
            node_data[node_id] = {time: dev * 1e6 for time, dev in results["node_deviations_time_series"][node_id]}

    # 创建一个所有时间点的集合
    all_times = set()
    for data in node_data.values():
        all_times.update(data.keys())

    # 按时间排序
    all_times = sorted(all_times)

    # 创建数据表
    data_table = {'time': all_times}
    for node_id in monitored_nodes:
        if node_id in node_data:
            # 为每个节点添加一列偏差数据
            data_table[f'node_{node_id}'] = [node_data[node_id].get(t, float('nan')) for t in all_times]

    # 创建DataFrame并保存到CSV
    df = pd.DataFrame(data_table)
    df.to_csv(filename, index=False)
    print(f"已将对齐的时间偏差数据保存到: {filename}")


if __name__ == "__main__":
    # 设置是否启用详细调试输出
    DEBUG_ENABLED = True

    print("运行IEEE 802.1AS仿真（完整消息集和两步同步）...")

    # 配置信息
    print("配置:")
    print(f"- 仿真时间: 600秒")
    print(f"- 节点数量: 101")
    print(f"- 监控节点: 10, 25, 50, 75, 100")

    # 运行两步法仿真
    print("\n开始两步法仿真...")
    sim = IEEE8021ASSimulation(num_nodes=100, simulation_time=70.0)
    results = sim.run()

    # 检查数据
    data_count = sum(len(deviations) for deviations in results["node_deviations"].values())
    print(f"两步法仿真收集了{data_count}个数据点")

    # 分析结果
    analysis = analyze_sync_precision(results)

    # 打印两步法摘要
    print("\n两步法同步结果:")
    print(f"使用{sim.num_nodes}个节点完成{sim.simulation_time}秒仿真")
    print(f"整体同步精度: {analysis['overall_precision'] * 1e6:.3f} µs")

    # 打印特定节点的详细结果
    selected_nodes = [10, 25, 50, 75, 100]
    print("\n选定节点的详细结果:")
    for node in selected_nodes:
        if node in analysis["max_deviations"]:
            max_dev = analysis["max_deviations"][node]
            sync_prob = analysis["sync_probabilities"][node]
            print(f"节点 {node}: 最大偏差 = {max_dev * 1e6:.3f} µs, "
                  f"同步概率 (1 µs) = {sync_prob:.2%}")

    # 保存结果到CSV文件
    create_aligned_time_series_csv(results, "gptp_time_deviations.csv")

    # 只有当数据点足够时才绘图
    if data_count > 0:
        print("\n绘制时间偏差图...")
        plot_time_deviations(results)
    else:
        print("\n警告: 绘图所需数据点不足!")