import heapq
import numpy as np
import matplotlib.pyplot as plt

# 常量配置
NUM_NODES = 100          # 链式网络中的节点数
SYNC_INTERVAL = 0.03125  # 同步间隔 (31.25 ms)
PHY_JITTER = 8e-9        # PHY抖动范围 (8 ns)
CLOCK_GRANULARITY = 8e-9 # 时钟粒度 (8 ns)
MAX_DRIFT_RATE = 10e-6   # 最大漂移率 (±10 ppm)
SIM_TIME = 100.0         # 仿真总时长 (秒)
PDELAY_INTERVAL = 1.0    # 传播延迟测量间隔 (1 s)
DRIFT_RATE_CHANGE = 1e-6 # 漂移率每秒变化范围 [0, 1] ppm/s

class Clock:
    def __init__(self, is_grandmaster=False):
        self.is_grandmaster = is_grandmaster
        # Grandmaster时钟漂移率设为极小的非零值（例如±0.1 ppm）
        if self.is_grandmaster:
            self.drift_rate = np.random.uniform(-0.1e-6, 0.1e-6)
        else:
            self.drift_rate = np.random.uniform(-MAX_DRIFT_RATE, MAX_DRIFT_RATE)
        self.offset = 0.0  # 相对于主时钟的偏移
        self.time = 0.0    # 本地时间
    
    def update(self, delta_t):
        # 动态调整漂移率（普通节点）
        if not self.is_grandmaster:
            self.drift_rate += np.random.uniform(-DRIFT_RATE_CHANGE, DRIFT_RATE_CHANGE) * delta_t
            self.drift_rate = np.clip(self.drift_rate, -MAX_DRIFT_RATE, MAX_DRIFT_RATE)
        
        # 更新本地时间（考虑漂移率）
        self.time += delta_t * (1 + self.drift_rate)
        return self.time
    
    def adjust(self, offset):
        # 调整本地时钟的偏移量（保留漂移率的影响）
        self.offset += offset
        self.time += offset

class Node:
    def __init__(self, node_id, is_grandmaster=False):
        self.id = node_id
        self.is_grandmaster = is_grandmaster
        self.clock = Clock(is_grandmaster=self.is_grandmaster)
        
        # Grandmaster物理层参数优化
        self.residence_time = 0.5e-3 if is_grandmaster else 1e-3  # 驻留时间减半
        self.propagation_delay = 25e-9 if is_grandmaster else 50e-9  # 传播延迟减半
        
        self.last_sync_time = 0.0
        self.last_pdelay_time = 0.0
        self.asymmetry = 0.0
        self.rate_ratio = 1.0
        self.neighbor_rate_ratio = 1.0
        self.sync_errors = []
        self.time_errors = []  # 时间误差记录（下游节点）
        self.gm_time_errors = []  # GM自身时间误差记录
    
    def receive_sync(self, sync_time, correction_field):
        # Grandmaster的PHY抖动更小
        phy_jitter = 2e-9 if self.is_grandmaster else PHY_JITTER
        
        # 接收Sync消息时添加抖动和时钟粒度
        actual_receive_time = sync_time + self.propagation_delay + \
                             np.random.uniform(0, phy_jitter) + \
                             np.random.uniform(0, CLOCK_GRANULARITY)
        
        # 计算本地时间与主时钟的偏差
        local_time = self.clock.time
        gm_time = sync_time + correction_field
        error = local_time - gm_time
        
        # 调整本地时钟（保留漂移率的影响）
        self.clock.adjust(-error)
        
        # 记录时间误差
        if self.is_grandmaster:
            # GM自身的时间误差（相对于理想时间）
            self.gm_time_errors.append((self.clock.time, abs(error)))
        else:
            # 下游节点的时间误差（相对于GM）
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
        self.nodes = [Node(i, is_grandmaster=(i==0)) for i in range(NUM_NODES)]
        self.grandmaster = self.nodes[0]
        self.event_queue = []
        self.current_time = 0.0
        self.event_counter = 0
    
    def schedule_event(self, time, callback, *args):
        heapq.heappush(self.event_queue, (time, self.event_counter, callback, args))
        self.event_counter += 1
    
    def run_simulation(self):
        self.schedule_event(0.0, self.send_sync, self.grandmaster)
        for i in range(1, NUM_NODES):
            self.schedule_event(0.0, self.measure_pdelay, self.nodes[i], self.nodes[i - 1])
        
        while self.event_queue and self.current_time < SIM_TIME:
            time, _, callback, args = heapq.heappop(self.event_queue)
            self.current_time = time
            callback(*args)
    
    def send_sync(self, node):
        sync_time = self.current_time
        correction_field = 0.0
        for i in range(1, NUM_NODES):
            forward_time = self.nodes[i].receive_sync(sync_time, correction_field)
            # 更新校正字段：传播延迟 + 率比误差 + 漂移率影响
            correction_field += self.nodes[i].propagation_delay * self.nodes[i].rate_ratio
            correction_field += self.nodes[i].clock.drift_rate * SYNC_INTERVAL
            sync_time = forward_time
        
        self.schedule_event(self.current_time + SYNC_INTERVAL, self.send_sync, node)
    
    def measure_pdelay(self, node, neighbor):
        node.measure_pdelay(neighbor)
        self.schedule_event(self.current_time + PDELAY_INTERVAL, self.measure_pdelay, node, neighbor)
    
    def plot_results(self, hop=None):
        if hop is not None:
            # 绘制某一跳的时间误差
            if hop < 1 or hop >= NUM_NODES:
                raise ValueError(f"Invalid hop: {hop}. Must be between 1 and {NUM_NODES - 1}.")
            node = self.nodes[hop]
            times, errors = zip(*node.time_errors)
            plt.figure(figsize=(12, 6))
            plt.plot(times, np.array(errors)*1e6)  # 转换为微秒
            plt.xlabel('Time (s)')
            plt.ylabel('Time Error (μs)')
            plt.title(f'Time Error at Hop {hop}')
            plt.grid(True)
            plt.show()
        else:
            # 绘制GM自身的时间误差
            gm_times, gm_errors = zip(*self.grandmaster.gm_time_errors)
            plt.figure(figsize=(12, 6))
            plt.plot(gm_times, np.array(gm_errors)*1e6)  # 转换为微秒
            plt.xlabel('Time (s)')
            plt.ylabel('Time Error (μs)')
            plt.title('GM Time Error')
            plt.grid(True)
            plt.show()

if __name__ == "__main__":
    network = Network()
    network.run_simulation()
    
    # 绘制GM自身的时间误差
    network.plot_results()
    
    # 绘制某一跳的时间误差
    network.plot_results(1)   # 第一跳（索引1）
    network.plot_results(10)  # 第十跳
    network.plot_results(99)  # 最后一跳