import numpy as np
import matplotlib.pyplot as plt

# 常量配置
NUM_NODES = 100          # 链式网络中的节点数
SYNC_INTERVAL = 0.03125  # 同步间隔 (31.25 ms)
PHY_JITTER = 8e-9        # PHY抖动范围 (8 ns)
CLOCK_GRANULARITY = 8e-9 # 时钟粒度 (8 ns)
MAX_DRIFT_RATE = 10e-6   # 最大漂移率 (±10 ppm)
SIM_TIME = 100.0         # 仿真总时长 (秒)

class Clock:
    def __init__(self):
        self.drift_rate = np.random.uniform(-MAX_DRIFT_RATE, MAX_DRIFT_RATE)
        self.offset = 0.0  # 相对于主时钟的偏移
        self.time = 0.0    # 本地时间
    
    def update(self, delta_t):
        # 考虑漂移率更新本地时间
        self.time += delta_t * (1 + self.drift_rate)
        return self.time

class Node:
    def __init__(self, node_id):
        self.id = node_id
        self.clock = Clock()
        self.residence_time = 1e-3  # 最大驻留时间 (1 ms)
        self.last_sync_time = 0.0
        self.propagation_delay = 50e-9  # 固定传播延迟 (50 ns)
    
    def receive_sync(self, sync_time, correction_field):
        # 接收Sync消息时添加PHY抖动和时钟粒度
        actual_receive_time = sync_time + self.propagation_delay + \
                             np.random.uniform(0, PHY_JITTER) + \
                             np.random.uniform(0, CLOCK_GRANULARITY)
        
        # 计算本地时间与主时钟的偏差
        local_time = self.clock.time
        gm_time = sync_time + correction_field
        self.clock.offset = local_time - gm_time
        
        # 更新本地时钟（简单线性调整）
        self.clock.time = gm_time
        
        # 添加驻留时间并转发Sync消息
        forward_time = actual_receive_time + self.residence_time
        return forward_time

class Network:
    def __init__(self):
        self.nodes = [Node(i) for i in range(NUM_NODES)]
        self.grandmaster = self.nodes[0]
        self.sync_errors = [[] for _ in range(NUM_NODES)]
    
    def run_simulation(self):
        current_time = 0.0
        while current_time < SIM_TIME:
            if current_time - self.grandmaster.last_sync_time >= SYNC_INTERVAL:
                # 主节点发送Sync消息
                sync_time = current_time
                correction_field = 0.0
                for i in range(1, NUM_NODES):
                    # 消息逐跳传播
                    forward_time = self.nodes[i].receive_sync(sync_time, correction_field)
                    # 记录当前节点的同步误差
                    error = self.nodes[i].clock.offset
                    self.sync_errors[i].append(abs(error))
                    # 更新校正字段（简化为累积传播延迟）
                    correction_field += self.nodes[i].propagation_delay
                    sync_time = forward_time
                self.grandmaster.last_sync_time = current_time
            
            # 时间步进（1 ms精度）
            current_time += 1e-3
            for node in self.nodes:
                node.clock.update(1e-3)

    def plot_results(self):
        # 绘制不同跳数的同步误差
        avg_errors = [np.mean(errors) for errors in self.sync_errors]
        max_errors = [np.max(errors) for errors in self.sync_errors]
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(NUM_NODES), avg_errors, label='Average Error')
        plt.plot(range(NUM_NODES), max_errors, label='Max Error')
        plt.xlabel('Hop Count')
        plt.ylabel('Synchronization Error (s)')
        plt.title('IEEE 802.1AS Synchronization Error vs. Hop Count')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    network = Network()
    network.run_simulation()
    network.plot_results()