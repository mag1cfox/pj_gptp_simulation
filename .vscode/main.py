"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/2/17 11:24
*  @Project :   pj_gptp_simulation
*  @Description :   用这个代码生成了data.csv但是只到了99跳。再试试另外的。
*  @FileName:   main_test4.py
**************************************
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
PHY_JITTER = 8e-9  # PHY抖动范围 (8 ns)
CLOCK_GRANULARITY = 8e-9  # 时钟粒度 (8 ns)
MAX_DRIFT_RATE = 10e-6  # 最大漂移率 (±10 ppm)
SIM_TIME = 100.0  # 仿真总时长 (秒)
PDELAY_INTERVAL = 1.0  # 传播延迟测量间隔 (1 s)
DRIFT_RATE_CHANGE = 1e-6  # 漂移率每秒变化范围 [0, 1] ppm/s


class Clock:
    def __init__(self, is_grandmaster=False):
        """
        初始化时钟模型

        参数:
            is_grandmaster (bool): 标识是否为Grandmaster时钟
        """
        self.is_grandmaster = is_grandmaster

        # 初始化时钟参数
        if self.is_grandmaster:
            # Grandmaster时钟漂移率限制在±0.1 ppm
            self.drift_rate = np.random.uniform(-0.1e-6, 0.1e-6)
        else:
            # 普通节点漂移率限制在±10 ppm
            self.drift_rate = np.random.uniform(-MAX_DRIFT_RATE, MAX_DRIFT_RATE)

        self.offset = 0.0  # 相对于主时钟的累积偏移
        self.time = 0.0  # 本地时钟时间
        self._base_time = 0.0  # 用于计算漂移的基础时间

    def update(self, delta_t):
        """
        更新本地时钟时间（考虑漂移率）

        参数:
            delta_t (float): 基于理想时间的流逝时间（秒）

        返回:
            float: 更新后的本地时间
        """
        # 动态调整漂移率（仅普通节点）
        if not self.is_grandmaster:
            # 每秒漂移率变化不超过±1 ppm
            drift_change = np.random.uniform(-DRIFT_RATE_CHANGE, DRIFT_RATE_CHANGE) * delta_t
            self.drift_rate += drift_change
            # 限制漂移率在允许范围内
            self.drift_rate = np.clip(
                self.drift_rate,
                -MAX_DRIFT_RATE,
                MAX_DRIFT_RATE
            )

        # 计算实际时间流逝（考虑漂移率）
        effective_delta = delta_t * (1 + self.drift_rate)

        # 更新本地时间（保留高精度计算）
        self.time += effective_delta
        self._base_time += delta_t  # 记录理想时间流逝

        return self.time

    def adjust(self, offset):
        """
        调整时钟偏移量（用于同步）

        参数:
            offset (float): 需要调整的偏移量（秒）
        """
        # 直接调整本地时间（保留漂移率的长期影响）
        self.time += offset
        self.offset += offset

    def get_true_time(self):
        """
        获取理想时间（用于误差计算）

        返回:
            float: 当前仿真时间的理想时间
        """
        # 理想时间 = 基础时间 + 初始偏移
        return self._base_time


class Node:
    def __init__(self, node_id, is_grandmaster=False):
        self.id = node_id
        self.is_grandmaster = is_grandmaster
        self.clock = Clock(is_grandmaster=self.is_grandmaster)
        self.last_updated_time = 0.0  # 最后更新时间戳

        # 物理层参数配置
        self.residence_time = 0.5e-3 if is_grandmaster else 1e-3  # 驻留时间（秒）
        self.propagation_delay = 25e-9 if is_grandmaster else 50e-9  # 初始传播延迟（秒）
        self.asymmetry = np.random.uniform(-10e-9, 10e-9)  # 链路非对称性（秒）

        # 时间戳特性
        self.hardware_timestamp_jitter = 0.5e-9 if is_grandmaster else 1e-9  # 时间戳抖动（秒）
        self.clock_granularity = 8e-9  # 时钟粒度（IEEE 1588默认值）

        # 同步状态
        self.last_sync_time = 0.0
        self.last_pdelay_time = 0.0
        self.rate_ratio = 1.0  # 频率比
        self.neighbor_rate_ratio = 1.0

        # 数据记录
        self.sync_errors = []  # 同步误差记录
        self.time_errors = []  # 绝对时间误差（下游节点）
        self.gm_time_errors = []  # GM自身时间误差
        self.pdelay_measurements = []  # 传播延迟测量记录

    def get_hardware_timestamp(self):
        """生成带有抖动的硬件时间戳"""
        return self.clock.time + np.random.normal(0, self.hardware_timestamp_jitter)

    def update_clock(self, current_time):
        """更新本地时钟到当前仿真时间"""
        delta_t = current_time - self.last_updated_time
        self.clock.update(delta_t)
        self.last_updated_time = current_time

    def receive_sync(self, sync_time, correction_field):
        """
        处理接收Sync消息的核心逻辑
        sync_time: 主时钟发送Sync的时间（T1）
        correction_field: 累积校正字段
        """
        # 1. 更新时间基准
        self.update_clock(sync_time)

        # 2. 生成接收时间戳（T2）
        t2 = self.get_hardware_timestamp()

        # 3. 计算主从时间差
        gm_time = sync_time + correction_field  # 主时钟的当前时间
        local_time = self.clock.time
        time_error = local_time - gm_time

        # 4. 时钟调整（两步法）
        # 第一步：偏移补偿
        self.clock.adjust(-time_error)

        # 第二步：频率补偿（简化实现）
        if not self.is_grandmaster:
            self.clock.drift_rate -= 0.1 * time_error / SYNC_INTERVAL

        # 5. 记录同步误差
        if self.is_grandmaster:
            # GM记录自身与理想时间的误差
            ideal_time = self.clock.get_true_time()
            self.gm_time_errors.append((self.clock.time, abs(ideal_time - self.clock.time)))
        else:
            # 下游节点记录与GM的误差
            self.time_errors.append((self.clock.time, abs(time_error)))

        # 6. 添加驻留时间并返回转发时间
        return t2 + self.residence_time + self.propagation_delay

    def measure_pdelay(self, neighbor):
        """
        执行Pdelay测量流程（IEEE 802.1AS-2011）
        返回测量得到的传播延迟
        """
        # 发起方（本节点）流程
        t1 = self.get_hardware_timestamp()
        pdelay_req = {
            't1': t1,
            'origin_time': self.clock.time
        }

        # 响应方（邻居节点）处理
        t2 = neighbor.get_hardware_timestamp()
        t3 = neighbor.get_hardware_timestamp()
        pdelay_resp = {
            't2': t2,
            't3': t3,
            'response_time': neighbor.clock.time
        }

        # 发起方处理响应
        t4 = self.get_hardware_timestamp()

        # 计算传播延迟（考虑非对称性）
        propagation_delay = ((t4 - t1) - (t3 - t2)) / 2
        corrected_delay = propagation_delay + self.asymmetry

        # 记录测量结果
        self.pdelay_measurements.append({
            't1': t1,
            't2': t2,
            't3': t3,
            't4': t4,
            'measured_delay': propagation_delay,
            'corrected_delay': corrected_delay
        })

        # 更新传播延迟（EMA滤波）
        alpha = 0.25  # 滤波系数
        self.propagation_delay = (alpha * corrected_delay +
                                  (1 - alpha) * self.propagation_delay)

        return self.propagation_delay

    def get_sync_accuracy(self):
        """计算当前同步精度指标"""
        if not self.time_errors:
            return float('inf')

        # 计算最近10次同步的平均误差
        recent_errors = [e[1] for e in self.time_errors[-10:]]
        return np.mean(recent_errors) * 1e9  # 转换为纳秒


class Network:
    def __init__(self):
        self.nodes = [Node(i, is_grandmaster=(i == 0)) for i in range(NUM_NODES)]
        self.grandmaster = self.nodes[0]
        self.event_queue = []
        self.current_time = 0.0
        self.event_counter = 0
        self.sync_count = 0  # 同步周期计数器

        # 初始化节点时间跟踪
        for node in self.nodes:
            node.last_updated_time = 0.0

    def schedule_event(self, time, callback, *args):
        heapq.heappush(self.event_queue, (time, self.event_counter, callback, args))
        self.event_counter += 1

    def run_simulation(self):
        # 初始化事件
        self.schedule_event(0.0, self.send_sync, self.grandmaster)
        for i in range(1, NUM_NODES):
            self.schedule_event(0.0, self.measure_pdelay, self.nodes[i], self.nodes[i-1])

        # 主仿真循环
        while self.event_queue and self.current_time < SIM_TIME:
            event_time, _, callback, args = heapq.heappop(self.event_queue)
            self.current_time = event_time
            callback(*args)

    def send_sync(self, node):
        # 记录GM自身误差（在发送Sync前）
        if node.is_grandmaster:
            delta_t = self.current_time - node.last_updated_time
            node.clock.update(delta_t)
            node.last_updated_time = self.current_time
            ideal_time = node.clock.get_true_time()
            error = node.clock.time - ideal_time
            node.gm_time_errors.append( (self.current_time, abs(error)) )

        # 链式传播Sync消息
        sync_path = []
        correction_field = 0.0
        current_sync_time = self.current_time

        for hop in range(NUM_NODES):
            if hop == 0:
                # Grandmaster自身不处理Sync
                sync_path.append(current_sync_time)
                continue

            current_node = self.nodes[hop]

            # 更新节点时钟到Sync到达时间
            delta_t = current_sync_time - current_node.last_updated_time
            current_node.clock.update(delta_t)
            current_node.last_updated_time = current_sync_time

            # 处理Sync接收
            forward_time = current_node.receive_sync(
                sync_time=current_sync_time,
                correction_field=correction_field
            )

            # 更新校正字段
            correction_field += current_node.propagation_delay
            correction_field += current_node.clock.drift_rate * SYNC_INTERVAL

            # 记录路径时间
            sync_path.append(forward_time)
            current_sync_time = forward_time

        # 调度下一个Sync事件
        next_sync = self.current_time + SYNC_INTERVAL
        self.schedule_event(next_sync, self.send_sync, node)
        self.sync_count += 1

    def measure_pdelay(self, node, neighbor):
        # 更新双方节点时钟到当前时间
        for n in [node, neighbor]:
            delta_t = self.current_time - n.last_updated_time
            n.clock.update(delta_t)
            n.last_updated_time = self.current_time

        # 执行延迟测量
        measured_delay = node.measure_pdelay(neighbor)

        # 记录测量结果
        node.pdelay_measurements.append({
            'timestamp': self.current_time,
            'neighbor': neighbor.id,
            'delay': measured_delay
        })

        # 调度下一次测量
        next_pdelay = self.current_time + PDELAY_INTERVAL
        self.schedule_event(next_pdelay, self.measure_pdelay, node, neighbor)

    def get_network_status(self):
        """获取网络状态摘要"""
        status = {
            'current_time': self.current_time,
            'sync_count': self.sync_count,
            'gm_error': self.grandmaster.gm_time_errors[-1][1] if self.grandmaster.gm_time_errors else 0,
            'worst_case_error': max([n.time_errors[-1][1] for n in self.nodes[1:] if n.time_errors], default=0)
        }
        return status

    def plot_results(self, hop=None):
        # 参数校验
        if hop is not None:
            if hop < 0 or hop >= NUM_NODES:
                raise ValueError(f"Invalid hop number. Must be 0 <= hop < {NUM_NODES}")

            node = self.nodes[hop]
            data = node.time_errors if not node.is_grandmaster else node.gm_time_errors
        else:
            node = self.grandmaster
            data = node.gm_time_errors

        # 数据预处理
        times, errors = zip(*data) if data else ([], [])
        times = np.array(times)
        errors = np.array(errors) * 1e9  # 转换为纳秒

        print(len(times))

        print(len(errors))
        save_tuple_to_csv(errors)


        # 绘图
        plt.figure(figsize=(12, 6))
        plt.plot(times, errors, label=f'Node {node.id}' if hop is not None else 'Grandmaster')
        plt.xlabel('Simulation Time (s)')
        plt.ylabel('Time Error (ns)')
        plt.title(f'Clock Synchronization Error - {"Grandmaster" if hop is None else f"Node {node.id}"}')
        # plt.ylim(0, min(100, errors.max()*1.2))  # 自动缩放Y轴
        plt.grid(True)
        plt.legend()

        # 优化刻度显示
        ax = plt.gca()
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

        plt.show()


def save_tuple_to_csv(tuple_data, filename='data3.csv'):
    # 检查文件是否存在
    file_exists = os.path.isfile(filename)

    # 读取现有数据（如果文件存在）
    existing_data = []
    if file_exists:
        with open(filename, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            existing_data = list(reader)

    # 找到第一个空列
    column_index = 0
    if existing_data:
        max_columns = max(len(row) for row in existing_data)
        for i in range(max_columns + 1):
            if all(i >= len(row) or row[i] == '' for row in existing_data):
                column_index = i
                break

    # 将 tuple 数据添加到正确的列
    for i, value in enumerate(tuple_data):
        row_index = i
        if row_index >= len(existing_data):
            existing_data.append([''] * (column_index + 1))
        while len(existing_data[row_index]) <= column_index:
            existing_data[row_index].append('')
        existing_data[row_index][column_index] = value

    # 写入数据到 CSV 文件
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(existing_data)


# 使用示例
if __name__ == "__main__":
    network = Network()
    network.run_simulation()

    # 可视化结果
    network.plot_results()  # Grandmaster自身误差
    # network.plot_results(hop=1)   # 第1跳节点
    # network.plot_results(hop=2)   # 第1跳节点
    # network.plot_results(hop=32)   # 第1跳节点
    # network.plot_results(hop=50)  # 中间节点
    # network.plot_results(hop=101) # 最后一跳节点
#
#
# def process_tuple(input_tuple):
#     # 使用列表推导式处理每个元素
#     processed_list = [round(x - 312500.0, 3) for x in input_tuple]
#
#     # 将处理后的列表转换回tuple并返回
#     return tuple(processed_list)
#
# def convert_to_microseconds(time_tuple):
#     # 创建一个新的列表来存储转换后的值
#     converted_list = []
#
#     # 遍历输入元组中的每个元素
#     for time in time_tuple:
#         # 将秒转换为微秒，并保留三位小数
#         microseconds = round(time * 10_000_000, 3)
#         converted_list.append(microseconds)
#
#     # 将列表转换为元组并返回
#     return tuple(converted_list)
#
# def save_tuple_to_csv(tuple_data, filename='data2.csv'):
#     # 检查文件是否存在
#     file_exists = os.path.isfile(filename)
#
#     # 读取现有数据（如果文件存在）
#     existing_data = []
#     if file_exists:
#         with open(filename, 'r', newline='') as csvfile:
#             reader = csv.reader(csvfile)
#             existing_data = list(reader)
#
#     # 找到第一个空列
#     column_index = 0
#     if existing_data:
#         max_columns = max(len(row) for row in existing_data)
#         for i in range(max_columns + 1):
#             if all(i >= len(row) or row[i] == '' for row in existing_data):
#                 column_index = i
#                 break
#
#     # 将 tuple 数据添加到正确的列
#     for i, value in enumerate(tuple_data):
#         row_index = i
#         if row_index >= len(existing_data):
#             existing_data.append([''] * (column_index + 1))
#         while len(existing_data[row_index]) <= column_index:
#             existing_data[row_index].append('')
#         existing_data[row_index][column_index] = value
#
#     # 写入数据到 CSV 文件
#     with open(filename, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerows(existing_data)
#
#
# if __name__ == "__main__":
#     network = Network()
#     network.run_simulation()
#
#     # 绘制某一跳的时间误差（例如第 10 跳）
#     # for i in range(2,102):
#     #     network.plot_results(hop=i)
#     # network.plot_results(hop=1)
#     # network.plot_results(hop=2)
#     # network.plot_results(hop=10)
#     # network.plot_results(hop=32)
#     # network.plot_results(hop=50)
#     network.plot_results(hop=101)
#     network.plot_results()