"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/2/16 21:20
*  @Project :   pj_gptp_simulation
*  @Description :   Description
*  @FileName:   test_claude.py
**************************************
"""

import time
import random

class Clock:
    def __init__(self, id, is_grandmaster=False):
        self.id = id
        self.is_grandmaster = is_grandmaster
        self.time = 0
        self.offset = random.uniform(-0.1, 0.1)  # 模拟初始时间偏移

    def get_time(self):
        return self.time + self.offset

    def adjust_time(self, offset):
        self.offset -= offset

class GPTPSimulation:
    def __init__(self, num_clocks):
        self.clocks = [Clock(0, True)] + [Clock(i) for i in range(1, num_clocks)]
        self.network_delay = 0.001  # 1ms 网络延迟

    def simulate_sync(self, steps):
        for step in range(steps):
            print(f"\nStep {step + 1}")
            grandmaster_time = self.clocks[0].get_time()

            for slave in self.clocks[1:]:
                # 模拟 Sync 消息
                t1 = grandmaster_time
                t2 = slave.get_time() + self.network_delay

                # 模拟 Follow_Up 消息
                t3 = grandmaster_time + self.network_delay

                # 模拟 Delay_Req 消息
                t4 = slave.get_time()

                # 计算时间偏移和路径延迟
                offset = ((t2 - t1) - (t4 - t3)) / 2
                path_delay = ((t2 - t1) + (t4 - t3)) / 2

                # 调整从时钟
                slave.adjust_time(offset)

                print(f"Clock {slave.id}: Offset = {offset:.9f}, Path Delay = {path_delay:.9f}")

            # 模拟时间流逝
            for clock in self.clocks:
                clock.time += 1

    def print_clock_status(self):
        for clock in self.clocks:
            print(f"Clock {clock.id}: {'Grandmaster' if clock.is_grandmaster else 'Slave'}, "
                  f"Time = {clock.get_time():.9f}")

# 运行仿真
sim = GPTPSimulation(5)  # 1个主时钟和4个从时钟
print("Initial status:")
sim.print_clock_status()

sim.simulate_sync(10)  # 运行10步同步

print("\nFinal status:")
sim.print_clock_status()
