"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/4/7 0:24
*  @Project :   pj_gptp_simulation
*  @Description :   main_20250406claude 升级
*  @FileName:   main_20250407claude_v1.py
**************************************
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class Clock:
    def __init__(self, node_id):
        # 初始化时钟参数（单位：秒）
        self.drift_rate_initial = np.random.uniform(-10, 10) * 1e-6  # 初始漂移率 (±10ppm)
        self.drift_rate_change = np.random.uniform(0, 1) * 1e-6  # 变化率上限1ppm/s
        self.current_drift_rate = self.drift_rate_initial
        self.granularity = 8e-9  # 8ns计时粒度
        self.physical_time = 0.0  # 实际物理时间
        self.offset = 0.0  # 对主时钟的偏移量
        self.rate_ratio = 1.0  # 相对主时钟的频率比
        self.last_update_time = 0.0  # 记录上一次更新时间

        # 温度影响模型
        self.temp_sensitivity = np.random.uniform(0.1, 0.5) * 1e-6  # 温度敏感度 (ppm/°C)
        self.ambient_temp = 25.0  # 初始环境温度(°C)

    def simulate_temp_change(self, duration):
        """模拟温度变化对时钟的影响"""
        # 随机温度波动 ±0.5°C
        self.ambient_temp += np.random.uniform(-0.5, 0.5) * min(duration, 1.0)
        temp_effect = (self.ambient_temp - 25.0) * self.temp_sensitivity
        return temp_effect

    def advance(self, duration):
        # 漂移率随时间线性变化（而非随机波动）
        drift_change = self.drift_rate_change * duration
        self.current_drift_rate += drift_change

        # 添加温度相关漂移（可保留一定随机性）
        temp_effect = self.simulate_temp_change(duration)
        self.current_drift_rate += temp_effect

        # 正常时钟前进逻辑
        self.physical_time += duration * (1 + self.current_drift_rate)

    def get_raw_time(self):
        """获取原始时间戳（带粒度噪声）"""
        return self.physical_time + np.random.uniform(0, self.granularity)

    def get_time(self):
        """获取修正后的时间（应用同步修正）"""
        # 将本地时钟修正为主时钟时间
        return self.get_raw_time() * self.rate_ratio + self.offset

    def correct(self, offset, rate_ratio):
        """修正时钟参数"""
        self.offset = offset
        self.rate_ratio = rate_ratio


class NetworkNode:
    def __init__(self, node_id, is_grandmaster=False):
        self.id = node_id
        self.is_grandmaster = is_grandmaster
        self.clock = Clock(node_id)

        # 网络延迟参数
        self.prop_delay = 50e-9  # 50ns固定传播延迟
        self.phy_jitter = np.random.uniform(0, 8e-9)  # PHY抖动0-8ns
        self.residence_time = min(np.random.exponential(500e-6), 1e-3)  # 驻留时间
        self.asymmetry_delay = np.random.uniform(-5e-9, 5e-9)  # 随机路径不对称性±5ns

        # 网络拓扑
        self.upstream = None
        self.downstream = None

        # 频率比测量误差±0.1ppm
        self.freq_error = np.random.uniform(-0.1e-6, 0.1e-6)

        # 同步状态跟踪
        self.sync_count = 0
        self.last_ratio_update = 0
        self.last_upstream_time = 0

        # 网络事件跟踪
        self.last_event_time = 0
        self.event_cooldown = 30.0  # 事件冷却时间（秒）

    def simulate_network_event(self):
        """模拟随机网络事件(如负载突变)"""
        # 检查是否已经过了冷却时间
        if self.clock.physical_time - self.last_event_time < self.event_cooldown:
            return 0

        if np.random.random() < 0.01:  # 1%概率发生突发事件
            self.last_event_time = self.clock.physical_time
            # 突发延迟增加5-50μs
            return np.random.uniform(5e-6, 50e-6)
        return 0

    def measure_propagation_delay(self):
        """测量链路延迟（PDelay过程），包含不对称性补偿"""
        # 基础延迟（对称部分）
        symmetric_delay = self.prop_delay + 0.5 * (self.phy_jitter + self.upstream.phy_jitter)

        # 添加随机波动、不对称性补偿与网络事件
        # event_delay = self.simulate_network_event()
        measured_delay = (symmetric_delay * (1 + np.random.uniform(-0.05, 0.05)) +
                          # self.asymmetry_delay + event_delay)
                          self.asymmetry_delay)

        # 确保延迟不为负
        return max(measured_delay, 1e-9)

    def calculate_rate_ratio(self, t1, t2, master_t1, master_t2):
        """计算频率比（相对上游节点）"""
        if t2 <= t1 or master_t2 <= master_t1:
            return 1.0  # 防止无效测量

        # 真实频率比（使用当前动态漂移率）
        real_ratio = ((1 + self.upstream.clock.current_drift_rate) /
                      (1 + self.clock.current_drift_rate))

        # 基于两个时间戳测量的频率比
        delta_local = t2 - t1
        delta_master = master_t2 - master_t1
        measured_ratio = (delta_master / delta_local)

        # 添加测量误差但限制在合理范围
        noisy_ratio = measured_ratio * (1 + self.freq_error)

        # 限制在合理范围内 (±1ppm)
        max_error = 1e-6
        if abs(noisy_ratio - real_ratio) > max_error:
            noisy_ratio = real_ratio + np.sign(noisy_ratio - real_ratio) * max_error

        return noisy_ratio

    def process_sync(self, sync_info):
        """
        处理Sync消息并更新时钟
        sync_info: 包含(gm_time, upstream_time, last_upstream_times)的元组
        返回: 更新后的sync_info元组
        """
        gm_time, upstream_time, last_times = sync_info

        # 主时钟节点直接返回自身时间，不需要更新时钟参数
        if self.is_grandmaster:
            local_time = self.clock.get_raw_time()
            return (gm_time, local_time, (0, 0))

        # 接收同步消息的时间戳（本地时钟）
        recv_ts = self.clock.get_raw_time() + self.phy_jitter

        # 测量链路延迟
        prop_delay = self.measure_propagation_delay()

        # 计算频率比（每10个周期更新一次，提高稳定性）
        self.sync_count += 1
        curr_time = self.clock.get_raw_time()

        # 更新频率比
        if self.sync_count % 10 == 0 or self.sync_count <= 3:
            if self.last_ratio_update > 0 and last_times[0] > 0:
                ratio = self.calculate_rate_ratio(
                    self.last_ratio_update, curr_time,
                    last_times[0], upstream_time
                )
                # 启用频率比更新
                if self.sync_count > 10:
                    ratio = 0.2 * ratio + 0.8 * self.clock.rate_ratio
                self.clock.rate_ratio = ratio

            self.last_ratio_update = curr_time
            last_upstream = upstream_time
        else:
            last_upstream = last_times[1]  # 保持上一次的值

        # 计算本地时钟与上游时钟的偏移
        # 考虑传播延迟和频率比
        corrected_upstream = upstream_time + prop_delay
        local_time = recv_ts

        # 时钟偏移计算（考虑频率比）
        offset = corrected_upstream - (local_time * self.clock.rate_ratio)

        # 更新时钟修正参数
        self.clock.offset = offset

        # 返回更新后的信息
        return (gm_time, self.clock.get_time(), (self.last_ratio_update, last_upstream))


class GPTP_Simulator:
    def __init__(self, hops=100, interval=31.25e-3, duration=60):
    # def __init__(self, hops=100, interval=125e-3, duration=60):
    # def __init__(self, hops=100, interval=1, duration=60):
        self.hops = hops
        self.interval = interval  # 同步间隔（默认31.25ms）
        self.duration = duration  # 仿真持续时间（秒）
        self.nodes = []
        self.errors = defaultdict(list)
        self.time_records = defaultdict(list)
        self.drift_records = defaultdict(list)  # 记录漂移率变化
        self.setup_network()

    def setup_network(self):
        """创建链式网络拓扑"""
        # 创建主时钟节点
        self.nodes.append(NetworkNode(0, is_grandmaster=True))

        # 创建级联的从时钟节点
        for i in range(1, self.hops + 1):
            node = NetworkNode(i)
            node.upstream = self.nodes[i - 1]
            self.nodes[i - 1].downstream = node
            self.nodes.append(node)

    def run(self):
        """运行仿真"""
        steps = int(self.duration / self.interval)

        for step in range(steps):
            # 主时钟当前时间
            gm_time = self.nodes[0].clock.get_time()

            # 初始同步信息: (主时钟时间, 上游时间, 上一次的时间元组)
            sync_info = (gm_time, gm_time, (0, 0))

            # 逐跳传播同步消息
            for hop in range(len(self.nodes)):
                # 每个节点处理同步消息并更新
                sync_info = self.nodes[hop].process_sync(sync_info)

                if hop > 0:  # 跳过主时钟
                    # 计算并记录时钟误差（相对主时钟）
                    corrected_time = self.nodes[hop].clock.get_time()
                    master_time = self.nodes[0].clock.get_time()
                    error = corrected_time - master_time
                    self.errors[hop].append(abs(error * 1e6))  # 转为μs

                    # 记录原始误差（用于分析）
                    if step > steps // 2:  # 只记录稳定后的数据
                        self.time_records[hop].append(error * 1e6)

                # 记录漂移率
                self.drift_records[hop].append(self.nodes[hop].clock.current_drift_rate * 1e6)  # 转为ppm

            # 推进所有节点的物理时钟
            for node in self.nodes:
                node.clock.advance(self.interval)

    def analyze(self):
        """分析并输出结果统计"""
        stats = {}
        for hop in sorted([1, 10, 25, 50, 75, 100]):
            if hop >= len(self.nodes):
                continue

            # 只分析后半段（稳定）
            start_idx = len(self.errors[hop]) // 2
            err = np.array(self.errors[hop][start_idx:])

            # 获取原始误差值（有正负）
            raw_errors = np.array(self.time_records[hop])

            # 分析误差的动态性
            if len(raw_errors) > 1:
                error_changes = np.diff(raw_errors)
            else:
                error_changes = np.array([0])

            stats[hop] = {
                'avg': np.mean(err),
                'max': np.max(err),
                'min': np.min(raw_errors),  # 最小误差（可能为负值）
                'abs_min': np.min(np.abs(raw_errors)),  # 绝对值最小误差
                'abs_max': np.max(np.abs(raw_errors)),  # 绝对值最大误差
                '99%': np.percentile(err, 99),
                'std': np.std(err),
                'error_rate_of_change': np.mean(np.abs(error_changes)),  # 误差变化率
                'stability_metric': np.std(error_changes)  # 稳定性指标
            }

            # 漂移率分析
            drift_data = np.array(self.drift_records[hop][start_idx:])
            stats[hop]['drift_avg'] = np.mean(drift_data)
            stats[hop]['drift_std'] = np.std(drift_data)

        return stats

    def visualize_results(self, results):
        """可视化仿真结果"""
        plt.figure(figsize=(15, 12))

        # 图1: 平均误差 vs 跳数
        plt.subplot(3, 1, 1)
        hops = sorted(list(results.keys()))
        avg_errors = [results[h]['avg'] for h in hops]
        max_errors = [results[h]['max'] for h in hops]

        plt.plot(hops, avg_errors, 'b-', linewidth=2, marker='o', label='Average Error')
        plt.plot(hops, max_errors, 'r--', linewidth=1.5, marker='s', label='Maximum Error')
        plt.axhline(y=1, color='g', linestyle='--', label='1μs Threshold')
        plt.axhline(y=2, color='orange', linestyle='--', label='2μs Threshold')
        plt.xlabel('Hops')
        plt.ylabel('Synchronization Error (μs)')
        plt.title('IEEE 802.1AS gPTP Synchronization Precision vs. Hop Count')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 图2: 所有跳数的误差分布
        plt.subplot(3, 1, 2)
        plot_hops = range(1, self.hops + 1, max(1, self.hops // 10))
        data_to_plot = [self.time_records[h] for h in plot_hops if h in self.time_records]
        if data_to_plot:
            plt.boxplot(data_to_plot, positions=list(plot_hops), widths=3)
            plt.xlabel('Hop Count')
            plt.ylabel('Error Distribution (μs)')
            plt.title('Error Distribution across Different Hop Counts')
            plt.grid(True, alpha=0.3)

        # 图3: 时钟漂移率变化
        plt.subplot(3, 1, 3)
        time_axis = np.arange(len(self.drift_records[1])) * self.interval
        for hop in [1, self.hops // 2, self.hops] if self.hops > 2 else [1]:
            if hop in self.drift_records:
                plt.plot(time_axis, self.drift_records[hop],
                         label=f'Node {hop} Drift Rate')

        plt.xlabel('Simulation Time (s)')
        plt.ylabel('Drift Rate (ppm)')
        plt.title('Clock Drift Rate Variation Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig('gptp_simulation_results.png', dpi=300)
        plt.show()


# 主程序调用
if __name__ == "__main__":
    np.random.seed(640)  # 固定随机种子以便结果可重现

    print("运行IEEE 802.1AS gPTP时间同步仿真...")

    # 可以试验不同的同步间隔
    intervals = {
        '31.25ms': 31.25e-3,
        '125ms': 125e-3,
        '1s': 1.0
    }

    # 选择要使用的间隔
    # selected_interval = '31.25ms'
    # selected_interval = '125ms'
    selected_interval = '1s'

    sim = GPTP_Simulator(hops=100, interval=intervals[selected_interval], duration=600)
    sim.run()
    results = sim.analyze()

    print(f"\n最终验证结果 (单位: μs) - 同步间隔: {selected_interval}")
    print("跳数\t\t平均误差\t\t最大误差\t\t最小误差\t\t|误差|最小\t\t|误差|最大\t\t99%分位数\t\t标准差\t\t误差变化率\t\t稳定性指标")
    print("-" * 120)
    for hop in sorted([1, 10, 25, 50, 75, 100]):
        if hop in results:
            print(f"{hop}\t\t{results[hop]['avg']:.3f}\t\t{results[hop]['max']:.3f}\t\t" +
                  f"{results[hop]['min']:.3f}\t\t{results[hop]['abs_min']:.3f}\t\t" +
                  f"{results[hop]['abs_max']:.3f}\t\t{results[hop]['99%']:.3f}\t\t" +
                  f"{results[hop]['std']:.3f}\t\t{results[hop]['error_rate_of_change']:.6f}\t\t" +
                  f"{results[hop]['stability_metric']:.6f}")

    print("\n时钟漂移率统计 (单位: ppm)")
    print("跳数\t平均漂移率\t漂移率标准差")
    print("-" * 40)
    for hop in sorted([1, 10, 25, 50, 75, 100]):
        if hop in results:
            print(f"{hop}\t\t{results[hop]['drift_avg']:.3f}\t\t{results[hop]['drift_std']:.3f}")

    # 可视化结果
    sim.visualize_results(results)

    print("\n仿真完成，结果已保存为图像文件。")