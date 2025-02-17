import heapq
import numpy as np
import matplotlib.pyplot as plt

# 常量配置
NUM_NODES = 101  # 链式网络中的节点数
SYNC_INTERVAL = 0.03125  # 同步间隔 (31.25 ms)
PHY_JITTER = 8e-9  # PHY抖动范围 (8 ns)
CLOCK_GRANULARITY = 8e-9  # 时钟粒度 (8 ns)
MAX_DRIFT_RATE = 10e-6  # 最大漂移率 (±10 ppm)
SIM_TIME = 100.0  # 仿真总时长 (秒)
PDELAY_INTERVAL = 1.0  # 传播延迟测量间隔 (1 s)


class Clock:
    def __init__(self):
        self.drift_rate = np.random.uniform(-MAX_DRIFT_RATE, MAX_DRIFT_RATE)
        self.offset = 0.0  # 相对于主时钟的偏移
        self.time = 0.0  # 本地时间

    def update(self, delta_t):
        # 考虑漂移率更新本地时间
        self.time += delta_t * (1 + self.drift_rate)
        return self.time

    def adjust(self, offset):
        # 调整本地时钟的偏移
        self.offset += offset
        self.time += offset


class Node:
    def __init__(self, node_id):
        self.id = node_id
        self.clock = Clock()
        self.residence_time = 1e-3  # 最大驻留时间 (1 ms)
        self.last_sync_time = 0.0
        self.last_pdelay_time = 0.0
        self.propagation_delay = 50e-9  # 固定传播延迟 (50 ns)
        self.asymmetry = 0.0  # 链路非对称性 (默认对称)
        self.rate_ratio = 1.0  # 率比 (初始为1)
        self.neighbor_rate_ratio = 1.0  # 邻居率比 (初始为1)
        self.sync_errors = []  # 同步误差记录
        self.time_errors = []  # 时间误差记录（随时间变化）

    def receive_sync(self, sync_time, correction_field):
        # 接收Sync消息时添加PHY抖动和时钟粒度
        actual_receive_time = sync_time + self.propagation_delay + \
                              np.random.uniform(0, PHY_JITTER) + \
                              np.random.uniform(0, CLOCK_GRANULARITY)

        # 计算本地时间与主时钟的偏差
        local_time = self.clock.time
        gm_time = sync_time + correction_field
        error = local_time - gm_time

        # 调整本地时钟（通过调整偏移量）
        self.clock.adjust(-error)

        # 记录时间误差
        self.time_errors.append((self.clock.time, abs(error)))

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
        self.event_queue = []  # 事件队列（用于事件驱动模型）
        self.current_time = 0.0
        self.event_counter = 0  # 事件计数器，用于唯一标识事件

    def schedule_event(self, time, callback, *args):
        # 将事件加入队列，使用事件计数器作为唯一标识符
        heapq.heappush(self.event_queue, (time, self.event_counter, callback, args))
        self.event_counter += 1

    def run_simulation(self):
        # 初始化事件：主节点定期发送Sync消息
        self.schedule_event(0.0, self.send_sync, self.grandmaster)

        # 初始化事件：所有节点定期测量传播延迟
        for i in range(1, NUM_NODES):
            self.schedule_event(0.0, self.measure_pdelay, self.nodes[i], self.nodes[i - 1])

        # 事件驱动仿真
        while self.event_queue and self.current_time < SIM_TIME:
            time, _, callback, args = heapq.heappop(self.event_queue)
            self.current_time = time
            callback(*args)

    def send_sync(self, node):
        # 主节点发送Sync消息
        sync_time = self.current_time
        correction_field = 0.0
        for i in range(1, NUM_NODES):
            # 消息逐跳传播
            forward_time = self.nodes[i].receive_sync(sync_time, correction_field)
            # 更新校正字段（包括率比和传播延迟误差）
            correction_field += self.nodes[i].propagation_delay * self.nodes[i].rate_ratio
            sync_time = forward_time

        # 安排下一次Sync消息
        self.schedule_event(self.current_time + SYNC_INTERVAL, self.send_sync, node)

    def measure_pdelay(self, node, neighbor):
        # 测量传播延迟
        node.measure_pdelay(neighbor)

        # 安排下一次测量
        self.schedule_event(self.current_time + PDELAY_INTERVAL, self.measure_pdelay, node, neighbor)

    def plot_results(self, hop):
        # 绘制某一跳的时间误差随时间变化
        if hop < 1 or hop >= NUM_NODES:
            raise ValueError(f"Invalid hop: {hop}. Must be between 1 and {NUM_NODES - 1}.")

        node = self.nodes[hop]
        times, errors = zip(*node.time_errors)  # 解压时间和误差
        times=times[2:]
        errors=errors[2:]
        print(type(times))
        print(type(errors))

        plt.figure(figsize=(10, 6))
        plt.plot(times, errors, label=f'Time Error at Hop {hop}')
        plt.xlabel('Time (s)')
        plt.ylabel('Time Error (s)')
        plt.title(f'Time Error vs. Time at Hop {hop}')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    network = Network()
    network.run_simulation()

    # 绘制某一跳的时间误差（例如第 10 跳）
    network.plot_results(hop=10)
    network.plot_results(hop=30)
    network.plot_results(hop=100)