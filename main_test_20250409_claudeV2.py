"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/4/9 10:11
*  @Project :   pj_gptp_simulation
*  @Description :   Description
*  @FileName:   main_test_20250409_claudeV2.py
**************************************
"""

"""
Modified IEEE 802.1AS Time Synchronization Simulation
- Runtime: 600 seconds
- Monitoring nodes: 10, 25, 50, 75, 100
- Plotting time deviations as a line graph
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import random


class Clock:
    """Model of a clock with drift and granularity."""

    def __init__(self, initial_drift_rate=None, max_drift_ppm=10, granularity_ns=8):
        """
        Initialize a clock with drift.

        Args:
            initial_drift_rate: Initial drift rate in ppm, or None for random
            max_drift_ppm: Maximum drift rate in ppm
            granularity_ns: Clock granularity in nanoseconds
        """
        if initial_drift_rate is None:
            # Random drift between -max_drift_ppm and max_drift_ppm ppm
            self.drift_rate = random.uniform(-max_drift_ppm, max_drift_ppm) * 1e-6
        else:
            self.drift_rate = initial_drift_rate * 1e-6

        self.drift_change_rate = 0  # Rate of change of drift rate
        self.max_drift_change_ppm_per_s = 1  # Maximum 1 ppm/s change in drift rate
        self.granularity = granularity_ns * 1e-9  # Clock granularity in seconds
        self.local_time = 0.0

    def update(self, elapsed_perfect_time):
        """
        Update the clock based on elapsed perfect time.

        Args:
            elapsed_perfect_time: Elapsed time in perfect time scale

        Returns:
            Current local time
        """
        # Update drift rate (may change over time)
        self.drift_rate += self.drift_change_rate * elapsed_perfect_time

        # Update local time with drift
        drift_factor = 1.0 + self.drift_rate
        self.local_time += elapsed_perfect_time * drift_factor

        # Apply granularity
        ticks = int(self.local_time / self.granularity)
        self.local_time = ticks * self.granularity

        return self.local_time

    def get_time(self):
        """Get current local time."""
        return self.local_time

    def set_time(self, new_time):
        """Set the local time."""
        self.local_time = new_time

    def get_drift_rate(self):
        """Get the current drift rate in ppm."""
        return self.drift_rate * 1e6


class TimeAwareSystem:
    """Implementation of a time-aware system in the IEEE 802.1AS network."""

    def __init__(self, node_id, is_grandmaster=False, phy_jitter_ns=8):
        """
        Initialize a time-aware system.

        Args:
            node_id: Identifier of the node
            is_grandmaster: True if this node is the grandmaster
            phy_jitter_ns: Maximum PHY jitter in nanoseconds
        """
        self.node_id = node_id
        self.is_grandmaster = is_grandmaster
        self.clock = Clock()
        self.max_phy_jitter = phy_jitter_ns * 1e-9  # Maximum PHY jitter in seconds

        # Neighbors
        self.upstream_neighbor = None  # Toward grandmaster
        self.downstream_neighbors = []  # Away from grandmaster

        # Sync-related state
        self.sync_locked = True  # syncLocked flag (True for better precision)
        # self.sync_interval = 31.25e-3  # syncInterval (31.25 ms)
        # self.sync_interval = 125e-3  # syncInterval (125 ms)
        self.sync_interval = 1.0  # syncInterval (1 s)
        self.next_sync_time = 0.0

        # Time synchronization information
        self.precise_origin_timestamp = 0.0  # O
        self.correction_field = 0.0  # C
        self.rate_ratio = 1.0  # r (ratio of GM frequency to local frequency)

        # Propagation delay state
        self.propagation_delay = 0.0  # D
        self.pdelay_interval = 1.0  # PdelayInterval (1 s)
        self.next_pdelay_time = 0.0
        self.neighbor_rate_ratio = 1.0  # nr (ratio of neighbor frequency to local frequency)

        # For statistics and analysis
        self.time_deviations = []  # Record of time deviations from GM
        self.sync_receptions = 0  # Count of sync messages received

    def receive_sync(self, perfect_time, origin_timestamp, correction_field, rate_ratio, sender_id):
        """
        Process a received Sync message.

        Args:
            perfect_time: Perfect time when message is received
            origin_timestamp: Precise origin timestamp (O)
            correction_field: Correction field (C)
            rate_ratio: Rate ratio (r)
            sender_id: ID of the sender node
        """
        # Apply PHY jitter to reception time
        jitter = random.uniform(0, self.max_phy_jitter)
        reception_time = self.clock.get_time() + jitter

        # Record time before correction for statistics
        if not self.is_grandmaster:
            # Calculate time deviation before correction
            gm_time = origin_timestamp + correction_field + self.propagation_delay
            local_time = self.clock.get_time()
            time_deviation = local_time - gm_time
            self.time_deviations.append((perfect_time, time_deviation))

        # If not grandmaster, correct the local clock
        if not self.is_grandmaster:
            # Calculate time at grandmaster
            gm_time = origin_timestamp + correction_field + self.propagation_delay

            # Correct local clock
            self.clock.set_time(gm_time)

            # Update rate ratio
            self.rate_ratio = rate_ratio * self.neighbor_rate_ratio

            self.sync_receptions += 1

            # Forward sync if we have downstream neighbors
            if self.sync_locked and self.downstream_neighbors:
                # Calculate residence time
                residence_time = min(random.uniform(0, 1e-3), 1e-3)  # Up to 1 ms residence time

                # Update correction field
                new_correction = correction_field + self.propagation_delay + (residence_time * self.rate_ratio)

                # Forward to downstream neighbors
                for neighbor in self.downstream_neighbors:
                    return self.forward_sync(perfect_time + residence_time, origin_timestamp,
                                      new_correction, self.rate_ratio)

            # If not in sync_locked mode, schedule next sync based on interval
            if not self.sync_locked:
                self.next_sync_time = perfect_time + self.sync_interval

    def forward_sync(self, perfect_time, origin_timestamp, correction_field, rate_ratio):
        """
        Forward a Sync message to downstream neighbors.

        Args:
            perfect_time: Perfect time when message is forwarded
            origin_timestamp: Precise origin timestamp (O)
            correction_field: Updated correction field (C)
            rate_ratio: Updated rate ratio (r)
        """
        for neighbor in self.downstream_neighbors:
            # Calculate propagation delay for this link (fixed + jitter)
            link_delay = 50e-9  # Base 50 ns propagation delay

            # Apply PHY jitter for sending
            jitter_out = random.uniform(0, self.max_phy_jitter)

            # Perfect time when message will arrive at neighbor
            arrival_time = perfect_time + link_delay + jitter_out

            # Schedule sync reception at the neighbor
            return (arrival_time, neighbor.node_id, origin_timestamp, correction_field, rate_ratio, self.node_id)

    def initiate_sync(self, perfect_time):
        """
        Initiate a Sync message (only for grandmaster).

        Args:
            perfect_time: Perfect time when sync is initiated

        Returns:
            List of sync events to be processed
        """
        if not self.is_grandmaster:
            return None

        # Set precise origin timestamp to current grandmaster time
        origin_timestamp = self.clock.get_time()
        correction_field = 0.0
        rate_ratio = 1.0

        # Schedule next sync
        self.next_sync_time = perfect_time + self.sync_interval

        # Forward to all downstream neighbors
        sync_events = []
        for neighbor in self.downstream_neighbors:
            event = self.forward_sync(perfect_time, origin_timestamp, correction_field, rate_ratio)
            if event:
                sync_events.append(event)

        return sync_events

    def measure_propagation_delay(self, perfect_time):
        """
        Perform propagation delay measurement.

        Args:
            perfect_time: Perfect time when measurement is initiated
        """
        if self.is_grandmaster or not self.upstream_neighbor:
            return

        # Timestamps for Pdelay_Req and Pdelay_Resp
        t1 = self.clock.get_time()

        # Propagation delay (one-way)
        link_delay = 50e-9  # 50 ns base propagation delay

        # Apply PHY jitter for sending Pdelay_Req
        jitter_out = random.uniform(0, self.max_phy_jitter)

        # Time when Pdelay_Req arrives at neighbor
        req_arrival = perfect_time + link_delay + jitter_out

        # Neighbor receives and timestamps Pdelay_Req
        jitter_in = random.uniform(0, self.max_phy_jitter)
        t2 = self.upstream_neighbor.clock.get_time() + jitter_in

        # Residence time at neighbor
        residence_time = random.uniform(0, 1e-6)  # Small residence time

        # Neighbor sends Pdelay_Resp
        t3 = self.upstream_neighbor.clock.get_time() + residence_time

        # Apply PHY jitter for sending Pdelay_Resp
        jitter_out2 = random.uniform(0, self.max_phy_jitter)

        # Time when Pdelay_Resp arrives back
        resp_arrival = req_arrival + residence_time + link_delay + jitter_out2

        # Receive and timestamp Pdelay_Resp
        jitter_in2 = random.uniform(0, self.max_phy_jitter)
        t4 = self.clock.get_time() + (resp_arrival - perfect_time) + jitter_in2

        # Calculate neighbor rate ratio (simplified)
        # In a real system, this would be measured over time
        self.neighbor_rate_ratio = (1 + self.upstream_neighbor.clock.get_drift_rate() * 1e-6) / \
                                   (1 + self.clock.get_drift_rate() * 1e-6)

        # Add error to neighbor rate ratio (up to 0.1 ppm as per standard)
        nr_error = random.uniform(-0.1, 0.1) * 1e-6
        self.neighbor_rate_ratio += nr_error

        # Calculate propagation delay using equation (4) from the paper
        self.propagation_delay = 0.5 * ((t4 - t1) - self.neighbor_rate_ratio * (t3 - t2))

        # Schedule next propagation delay measurement
        self.next_pdelay_time = perfect_time + self.pdelay_interval


class IEEE8021ASSimulation:
    """Simulation of IEEE 802.1AS network."""

    def __init__(self, num_nodes=100, simulation_time=600.0):
        """
        Initialize the simulation.

        Args:
            num_nodes: Number of time-aware systems in the network
            simulation_time: Total simulation time in seconds
        """
        self.num_nodes = num_nodes
        self.simulation_time = simulation_time
        self.perfect_time = 0.0
        self.nodes = []
        self.events = []  # (time, event_type, params)

        # Create nodes
        for i in range(num_nodes):
            is_gm = (i == 0)
            node = TimeAwareSystem(i, is_grandmaster=is_gm)
            self.nodes.append(node)

        # Connect nodes in a linear topology
        for i in range(num_nodes - 1):
            self.nodes[i + 1].upstream_neighbor = self.nodes[i]
            self.nodes[i].downstream_neighbors.append(self.nodes[i + 1])

    def run(self):
        """Run the simulation."""
        # Initialize events
        self._schedule_initial_events()

        # Process events
        while self.events and self.perfect_time < self.simulation_time:
            # Get next event
            event_time, event_type, params = self.events.pop(0)

            # Update perfect time
            dt = event_time - self.perfect_time
            if dt > 0:
                self._update_clocks(dt)
                self.perfect_time = event_time

            # Process event
            if event_type == "sync":
                self._process_sync_event(*params)
            elif event_type == "initiate_sync":
                self._process_initiate_sync(params)
            elif event_type == "pdelay":
                self._process_pdelay(params)

        # Collect and analyze results
        return self._analyze_results()

    def _schedule_initial_events(self):
        """Schedule initial events."""
        # Schedule initial sync from grandmaster
        self.events.append((0.0, "initiate_sync", 0))

        # Schedule initial propagation delay measurements
        for i in range(1, self.num_nodes):
            self.events.append((random.uniform(0, 1.0), "pdelay", i))

    def _update_clocks(self, dt):
        """
        Update all clocks based on elapsed perfect time.

        Args:
            dt: Elapsed perfect time in seconds
        """
        for node in self.nodes:
            node.clock.update(dt)

    def _process_sync_event(self, receiver_id, origin_timestamp, correction_field, rate_ratio, sender_id):
        """Process a sync reception event."""
        node = self.nodes[receiver_id]
        sync_event = node.receive_sync(self.perfect_time, origin_timestamp,
                                       correction_field, rate_ratio, sender_id)

        if sync_event:
            arrival_time, receiver, origin, correction, rate, sender = sync_event
            self._insert_event((arrival_time, "sync",
                                (receiver, origin, correction, rate, sender)))

    def _process_initiate_sync(self, node_id):
        """Process a sync initiation event."""
        node = self.nodes[node_id]
        sync_events = node.initiate_sync(self.perfect_time)

        if sync_events:
            for event in sync_events:
                arrival_time, receiver, origin, correction, rate, sender = event
                self._insert_event((arrival_time, "sync",
                                    (receiver, origin, correction, rate, sender)))

        # Schedule next sync initiation
        next_sync_time = node.next_sync_time
        self._insert_event((next_sync_time, "initiate_sync", node_id))

    def _process_pdelay(self, node_id):
        """Process a propagation delay measurement event."""
        node = self.nodes[node_id]
        node.measure_propagation_delay(self.perfect_time)

        # Schedule next pdelay measurement
        next_pdelay_time = node.next_pdelay_time
        self._insert_event((next_pdelay_time, "pdelay", node_id))

    def _insert_event(self, event):
        """Insert an event into the event queue in correct time order."""
        time, event_type, params = event
        index = 0
        while index < len(self.events) and self.events[index][0] < time:
            index += 1
        self.events.insert(index, event)

    def _analyze_results(self):
        """Analyze simulation results."""
        results = {
            "node_deviations": {},
            "node_deviations_time_series": {},  # Add time series data
            "propagation_delays": [],
            "sync_receptions": []
        }

        # Collect time deviations for each node
        monitored_nodes = [10, 25, 50, 75, 100]
        for i, node in enumerate(self.nodes):
            if i > 0 and i in monitored_nodes:  # Only collect data for monitored nodes
                deviations = [dev for _, dev in node.time_deviations]
                results["node_deviations"][i] = deviations
                # Store time series data
                results["node_deviations_time_series"][i] = node.time_deviations
                results["sync_receptions"].append(node.sync_receptions)

        # Collect propagation delays
        for i in monitored_nodes:
            if i < self.num_nodes:
                results["propagation_delays"].append(self.nodes[i].propagation_delay)

        return results


def analyze_sync_precision(results, threshold_us=1.0):
    """
    Analyze synchronization precision from simulation results.

    Args:
        results: Simulation results
        threshold_us: Precision threshold in microseconds

    Returns:
        Dict with analysis results
    """
    analysis = {
        "max_deviations": {},
        "sync_probabilities": {},
        "overall_precision": 0.0
    }

    # Calculate maximum deviation for each node
    for node_id, deviations in results["node_deviations"].items():
        if deviations:
            max_dev = max(abs(dev) for dev in deviations)
            analysis["max_deviations"][node_id] = max_dev

    # Calculate synchronization probability for different thresholds
    threshold_s = threshold_us * 1e-6
    for node_id, deviations in results["node_deviations"].items():
        if deviations:
            in_sync_count = sum(1 for dev in deviations if abs(dev) < threshold_s)
            probability = in_sync_count / len(deviations) if deviations else 0
            analysis["sync_probabilities"][node_id] = probability

    # Calculate overall precision (maximum deviation across all nodes)
    if analysis["max_deviations"]:
        analysis["overall_precision"] = max(analysis["max_deviations"].values())

    return analysis


def plot_time_deviations(results):
    """
    Plot time deviations for specified nodes over time.

    Args:
        results: Simulation results containing time series data
    """
    plt.figure(figsize=(12, 8))

    # Select nodes to plot (10, 25, 50, 75, 100)
    nodes_to_plot = [10, 25, 50, 75, 100]

    # Plot time deviations for each node
    for node_id in nodes_to_plot:
        if node_id in results["node_deviations_time_series"]:
            time_series = results["node_deviations_time_series"][node_id]
            times = [t for t, _ in time_series]
            deviations = [d * 1e6 for _, d in time_series]  # Convert to microseconds
            plt.plot(times, deviations, label=f'Node {node_id}')

    plt.xlabel('Simulation Time (s)')
    plt.ylabel('Time Deviation (µs)')
    plt.title('Time Deviations from Grandmaster Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('time_deviations_over_time.png')
    plt.close()


if __name__ == "__main__":
    # Run simulation with 101 nodes (0-100) for 600 seconds
    sim = IEEE8021ASSimulation(num_nodes=101, simulation_time=600.0)
    results = sim.run()

    # Analyze results
    analysis = analyze_sync_precision(results)

    # Print summary
    print(f"Simulation completed with {sim.num_nodes} nodes for {sim.simulation_time} seconds")
    print(f"Overall synchronization precision: {analysis['overall_precision'] * 1e6:.3f} µs")

    # Plot time deviations over time
    plot_time_deviations(results)

    # Print detailed results for specific nodes
    selected_nodes = [10, 25, 50, 75, 100]
    print("\nDetailed results for selected nodes:")
    for node in selected_nodes:
        if node in analysis["max_deviations"]:
            max_dev = analysis["max_deviations"][node]
            sync_prob = analysis["sync_probabilities"][node]
            print(f"Node {node}: Max deviation = {max_dev * 1e6:.3f} µs, "
                  f"Sync probability (1 µs) = {sync_prob:.2%}")