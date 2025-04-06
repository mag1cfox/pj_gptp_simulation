"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/4/6 21:17
*  @Project :   pj_gptp_simulation
*  @Description :   Description
*  @FileName:   main_20250406_deepseek.py
**************************************
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


class Clock:
    def __init__(self, drift_rate=0, granularity=8e-9):
        self.drift_rate = drift_rate  # in ppm (parts per million)
        self.granularity = granularity  # in seconds
        self.time = 0
        self.correction = 0
        self.rate_ratio = 1.0
        self.neighbor_rate_ratio = 1.0

    def advance(self, duration):
        # Advance time including drift rate effects
        self.time += duration * (1 + self.drift_rate * 1e-6)

    def get_time(self):
        # Return time with granularity effects
        return self.time + np.random.uniform(0, self.granularity)

    def apply_correction(self, gm_offset, rate_ratio):
        self.correction = gm_offset
        self.rate_ratio = rate_ratio


class NetworkNode:
    def __init__(self, node_id, is_grandmaster=False):
        self.id = node_id
        self.is_grandmaster = is_grandmaster
        self.clock = Clock(drift_rate=np.random.uniform(-10, 10))
        self.prop_delay = 50e-9  # 50 ns initial
        self.phy_jitter = np.random.uniform(0, 8e-9)
        self.residence_time = np.random.uniform(0, 1e-3)  # up to 1 ms
        self.upstream = None
        self.downstream = None
        self.last_sync = 0

    def measure_prop_delay(self):
        # Simulate Pdelay measurement with four timestamps
        t1 = self.clock.get_time()
        t2 = self.upstream.clock.get_time() + self.phy_jitter
        t3 = self.upstream.clock.get_time() + self.upstream.phy_jitter
        t4 = self.clock.get_time() + self.phy_jitter

        # Calculate propagation delay with added effects
        measured_delay = 0.5 * ((t4 - t1) - self.neighbor_rate_ratio * (t3 - t2))
        return max(measured_delay, 0)  # can't be negative

    def receive_sync(self, sync_time, origin_timestamp, correction_field, rate_ratio):
        # Add PHY jitter to received timestamp
        sync_time += self.phy_jitter

        # Calculate neighbor rate ratio
        if not self.is_grandmaster:
            self.neighbor_rate_ratio = (1 + self.upstream.clock.drift_rate * 1e-6) / (1 + self.clock.drift_rate * 1e-6)

        # Update clock correction
        gm_time = origin_timestamp + correction_field + self.measure_prop_delay()
        local_time = sync_time
        offset = gm_time - local_time
        self.clock.apply_correction(offset, rate_ratio * self.neighbor_rate_ratio)

        # Return updated Sync message for next hop
        new_correction = correction_field + self.measure_prop_delay() + (self.residence_time * self.clock.rate_ratio)
        return (
            self.clock.get_time() + self.phy_jitter,  # transmission time
            origin_timestamp,
            new_correction,
            self.clock.rate_ratio
        )


class NetworkSimulation:
    def __init__(self, num_hops=100, sync_interval=31.25e-3, sim_time=100):
        self.num_hops = num_hops
        self.sync_interval = sync_interval
        self.sim_time = sim_time
        self.nodes = []
        self.errors = [[] for _ in range(num_hops)]
        self.setup_network()

    def setup_network(self):
        # Create grandmaster
        gm = NetworkNode(0, is_grandmaster=True)
        self.nodes.append(gm)

        # Create slave nodes
        for i in range(1, self.num_hops + 1):
            node = NetworkNode(i)
            node.upstream = self.nodes[i - 1]
            self.nodes[i - 1].downstream = node
            self.nodes.append(node)

    def run(self):
        steps = int(self.sim_time / self.sync_interval)

        for step in range(steps):
            # Grandmaster initiates Sync message
            origin_time = self.nodes[0].clock.get_time()
            sync_msg = (origin_time, origin_time, 0, 1.0)

            # Propagate Sync through each hop
            for i in range(1, len(self.nodes)):
                sync_msg = self.nodes[i].receive_sync(*sync_msg)

                # Store error vs grandmaster
                node_time = self.nodes[i].clock.get_time() + self.nodes[i].clock.correction
                error = abs(node_time - self.nodes[0].clock.get_time())
                self.errors[i - 1].append(error)

            # Advance all clocks between sync intervals
            for node in self.nodes:
                node.clock.advance(self.sync_interval)

    def plot_results(self):
        plt.figure(figsize=(12, 6))

        # Plot maximum error per hop
        max_errors = [max(errors) * 1e6 for errors in self.errors]
        plt.plot(range(1, self.num_hops + 1), max_errors, label='Max error')

        # Plot theoretical worst case
        theoretical = [0.625 + 0.0625 * hop for hop in range(self.num_hops)]
        plt.plot(range(1, self.num_hops + 1), theoretical, '--', label='Theoretical worst-case')

        plt.xlabel('Hop number')
        plt.ylabel('Synchronization error (μs)')
        plt.title(f'gPTP Synchronization Error over {self.num_hops} Hops')
        plt.grid(True)
        plt.legend()
        plt.show()

        # Plot error distribution for 100th hop
        plt.figure(figsize=(12, 6))
        plt.hist(np.array(self.errors[-1]) * 1e6, bins=50)
        plt.xlabel('Synchronization error (μs)')
        plt.ylabel('Count')
        plt.title('Error Distribution for Node 100')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    sim = NetworkSimulation(num_hops=100, sim_time=100)
    sim.run()
    sim.plot_results()
