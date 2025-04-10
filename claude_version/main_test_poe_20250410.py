"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/4/10 18:15
*  @Project :   pj_gptp_simulation
*  @Description :  code based on main_test_20250409_claudeV2.py
*  @FileName:   main_test_poe_20250410.py
**************************************
"""

"""
Enhanced IEEE 802.1AS Time Synchronization Simulation
- Runtime: 600 seconds
- Monitoring nodes: 10, 25, 50, 75, 100
- Plotting time deviations as a line graph
- Complete message set: Sync, Follow_Up, Announce, Signaling
- Support for One-Step and Two-Step synchronization modes
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import random
import enum
import time


class MessageType(enum.Enum):
    """Enumeration of IEEE 802.1AS message types."""
    SYNC = 0
    FOLLOW_UP = 1
    PDELAY_REQ = 2
    PDELAY_RESP = 3
    PDELAY_RESP_FOLLOW_UP = 4
    ANNOUNCE = 5
    SIGNALING = 6


class ClockIdentity:
    """Class representing a clock identity."""

    def __init__(self, node_id):
        """
        Initialize a clock identity.

        Args:
            node_id: The node identifier
        """
        self.node_id = node_id
        # In real 802.1AS, this would be an 8-byte value derived from MAC address
        self.clock_identity = f"clock-{node_id:04d}"

    def __str__(self):
        return self.clock_identity


class ClockQuality:
    """Class representing clock quality attributes."""

    def __init__(self, clock_class=248, clock_accuracy=0xFE, offset_scaled_log_variance=0xFFFF):
        """
        Initialize clock quality.

        Args:
            clock_class: Clock class (default: 248)
            clock_accuracy: Clock accuracy (default: 0xFE)
            offset_scaled_log_variance: Offset scaled log variance (default: 0xFFFF)
        """
        self.clock_class = clock_class
        self.clock_accuracy = clock_accuracy
        self.offset_scaled_log_variance = offset_scaled_log_variance

    def is_better_than(self, other):
        """
        Compare with another clock quality to determine which is better.

        Args:
            other: Another ClockQuality instance

        Returns:
            True if this clock quality is better than the other
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


class PortState(enum.Enum):
    """Enumeration of port states in IEEE 802.1AS."""
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
    """Base class for PTP messages used in IEEE 802.1AS."""

    def __init__(self, message_type, source_port_identity, sequence_id):
        """
        Initialize a PTP message.

        Args:
            message_type: Type of message (MessageType enum)
            source_port_identity: Source port identity
            sequence_id: Sequence identifier
        """
        self.message_type = message_type
        self.source_port_identity = source_port_identity
        self.sequence_id = sequence_id
        self.correction_field = 0.0


class SyncMessage(PTPMessage):
    """Sync message in IEEE 802.1AS."""

    def __init__(self, source_port_identity, sequence_id, timestamp=None, two_step_flag=False):
        """
        Initialize a Sync message.

        Args:
            source_port_identity: Source port identity
            sequence_id: Sequence identifier
            timestamp: Origin timestamp for one-step mode, None for two-step
            two_step_flag: Whether this is part of a two-step synchronization
        """
        super().__init__(MessageType.SYNC, source_port_identity, sequence_id)
        self.origin_timestamp = timestamp
        self.two_step_flag = two_step_flag


class FollowUpMessage(PTPMessage):
    """Follow Up message in IEEE 802.1AS."""

    def __init__(self, source_port_identity, sequence_id, precise_origin_timestamp):
        """
        Initialize a Follow Up message.

        Args:
            source_port_identity: Source port identity
            sequence_id: Sequence identifier
            precise_origin_timestamp: Precise origin timestamp
        """
        super().__init__(MessageType.FOLLOW_UP, source_port_identity, sequence_id)
        self.precise_origin_timestamp = precise_origin_timestamp


class PdelayReqMessage(PTPMessage):
    """Pdelay Request message in IEEE 802.1AS."""

    def __init__(self, source_port_identity, sequence_id):
        """
        Initialize a Pdelay Request message.

        Args:
            source_port_identity: Source port identity
            sequence_id: Sequence identifier
        """
        super().__init__(MessageType.PDELAY_REQ, source_port_identity, sequence_id)
        self.origin_timestamp = None  # Set when sent


class PdelayRespMessage(PTPMessage):
    """Pdelay Response message in IEEE 802.1AS."""

    def __init__(self, source_port_identity, sequence_id, request_receipt_timestamp, requesting_port_identity):
        """
        Initialize a Pdelay Response message.

        Args:
            source_port_identity: Source port identity
            sequence_id: Sequence identifier
            request_receipt_timestamp: Timestamp when Pdelay Request was received
            requesting_port_identity: Identity of port that sent the request
        """
        super().__init__(MessageType.PDELAY_RESP, source_port_identity, sequence_id)
        self.request_receipt_timestamp = request_receipt_timestamp
        self.requesting_port_identity = requesting_port_identity


class PdelayRespFollowUpMessage(PTPMessage):
    """Pdelay Response Follow Up message in IEEE 802.1AS."""

    def __init__(self, source_port_identity, sequence_id, response_origin_timestamp, requesting_port_identity):
        """
        Initialize a Pdelay Response Follow Up message.

        Args:
            source_port_identity: Source port identity
            sequence_id: Sequence identifier
            response_origin_timestamp: Precise timestamp when Pdelay Response was sent
            requesting_port_identity: Identity of port that sent the request
        """
        super().__init__(MessageType.PDELAY_RESP_FOLLOW_UP, source_port_identity, sequence_id)
        self.response_origin_timestamp = response_origin_timestamp
        self.requesting_port_identity = requesting_port_identity


class AnnounceMessage(PTPMessage):
    """Announce message in IEEE 802.1AS."""

    def __init__(self, source_port_identity, sequence_id, gm_priority1, gm_clock_quality,
                 gm_priority2, gm_identity, steps_removed):
        """
        Initialize an Announce message.

        Args:
            source_port_identity: Source port identity
            sequence_id: Sequence identifier
            gm_priority1: Priority1 of the grandmaster
            gm_clock_quality: Clock quality of the grandmaster
            gm_priority2: Priority2 of the grandmaster
            gm_identity: Identity of the grandmaster
            steps_removed: Steps removed from the grandmaster
        """
        super().__init__(MessageType.ANNOUNCE, source_port_identity, sequence_id)
        self.gm_priority1 = gm_priority1
        self.gm_clock_quality = gm_clock_quality
        self.gm_priority2 = gm_priority2
        self.gm_identity = gm_identity
        self.steps_removed = steps_removed


class SignalingMessage(PTPMessage):
    """Signaling message in IEEE 802.1AS."""

    def __init__(self, source_port_identity, sequence_id, target_port_identity, tlv_type, tlv_value):
        """
        Initialize a Signaling message.

        Args:
            source_port_identity: Source port identity
            sequence_id: Sequence identifier
            target_port_identity: Target port identity
            tlv_type: Type of TLV (Type-Length-Value)
            tlv_value: Value of the TLV
        """
        super().__init__(MessageType.SIGNALING, source_port_identity, sequence_id)
        self.target_port_identity = target_port_identity
        self.tlv_type = tlv_type
        self.tlv_value = tlv_value


class Port:
    """Port in an IEEE 802.1AS time-aware system."""

    def __init__(self, node, port_number, phy_jitter_ns=8):
        """
        Initialize a port.

        Args:
            node: The time-aware system this port belongs to
            port_number: Port number
            phy_jitter_ns: Maximum PHY jitter in nanoseconds
        """
        self.node = node
        self.port_number = port_number
        self.port_identity = f"{node.clock_identity}:{port_number}"
        self.state = PortState.INITIALIZING
        self.max_phy_jitter = phy_jitter_ns * 1e-9  # Maximum PHY jitter in seconds

        # Connected port
        self.peer_port = None

        # Sequence counters
        self.sync_sequence_id = 0
        self.announce_sequence_id = 0
        self.pdelay_sequence_id = 0
        self.signaling_sequence_id = 0

        # Path delay measurement
        self.pdelay_interval = 1.0  # PdelayInterval (1 s)
        self.next_pdelay_time = 0.0
        self.t1 = 0.0  # Pdelay_Req transmission timestamp
        self.t2 = 0.0  # Pdelay_Req receipt timestamp
        self.t3 = 0.0  # Pdelay_Resp transmission timestamp
        self.t4 = 0.0  # Pdelay_Resp receipt timestamp
        self.current_pdelay_sequence_id = None  # Current path delay measurement sequence ID
        self.neighbor_rate_ratio = 1.0  # Ratio of neighbor frequency to local frequency
        self.mean_path_delay = 0.0  # Mean path delay in seconds

        # Sync related
        self.sync_receipt_timeout = 3.0  # Sync receipt timeout in seconds
        self.last_sync_receipt_time = 0.0  # Last time a Sync was received
        self.last_sync_sequence_id = None  # Last received Sync sequence ID
        self.pending_follow_up = False  # Whether waiting for a Follow_Up

        # Announce related
        self.announce_interval = 1.0  # AnnounceInterval (1 s)
        self.next_announce_time = 0.0
        self.announce_receipt_timeout = 3.0  # Announce receipt timeout
        self.last_announce_receipt_time = 0.0  # Last time an Announce was received

        # Two-step sync data
        self.sync_send_times = {}  # Maps sequence_id to send time for Follow_Up

    def connect(self, peer_port):
        """
        Connect this port to another port.

        Args:
            peer_port: Port to connect to
        """
        self.peer_port = peer_port
        peer_port.peer_port = self

    def send_message(self, message, perfect_time):
        """
        Send a message to the peer port.

        Args:
            message: The message to send
            perfect_time: Perfect time when sending the message

        Returns:
            Tuple of (arrival_time, message) or None if no peer
        """
        if not self.peer_port:
            return None

        # Calculate propagation delay for this link (fixed + jitter)
        link_delay = 50e-9  # Base 50 ns propagation delay

        # Apply PHY jitter for sending
        jitter_out = random.uniform(0, self.max_phy_jitter)

        # Perfect time when message will arrive at peer
        arrival_time = perfect_time + link_delay + jitter_out

        # Return event for scheduler
        return (arrival_time, self.peer_port, message)

    # 在receive_message方法中添加以下代码来确保消息处理被正确记录
    def receive_message(self, message, perfect_time):
        """
        Process a received message.

        Args:
            message: The received message
            perfect_time: Perfect time when the message is received

        Returns:
            List of new events triggered by this message reception
        """
        # Apply PHY jitter to reception time
        jitter = random.uniform(0, self.max_phy_jitter)
        reception_time = self.node.clock.get_time() + jitter

        events = []

        # 调试输出，确认消息处理
        # print(f"Node {self.node.node_id} Port {self.port_number} received {message.message_type} at {perfect_time:.6f}")

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
        Process a received Sync message.

        Args:
            message: The Sync message
            reception_time: Local time when message was received
            perfect_time: Perfect time when message was received

        Returns:
            List of new events triggered by this message reception
        """
        events = []

        # Update sync receipt time
        self.last_sync_receipt_time = perfect_time
        self.last_sync_sequence_id = message.sequence_id

        # If this is a slave port receiving from its master
        if self.state == PortState.SLAVE:
            # If one-step mode (timestamp included in Sync)
            if not message.two_step_flag and message.origin_timestamp is not None:
                # Record time before correction for statistics
                gm_time = message.origin_timestamp + message.correction_field + self.mean_path_delay
                local_time = self.node.clock.get_time()
                time_deviation = local_time - gm_time

                # Record the deviation
                self.node.record_time_deviation(perfect_time, time_deviation)

                # Correct local clock
                self.node.clock.set_time(gm_time)
                self.node.sync_receptions += 1

                # Forward sync to downstream ports if enabled
                if self.node.sync_locked:
                    new_events = self.node.forward_sync(message, perfect_time, reception_time)
                    if new_events:
                        events.extend(new_events)

            # If two-step mode, mark as waiting for Follow_Up
            elif message.two_step_flag:
                self.pending_follow_up = True
                # Store sync receive time for later use with Follow_Up
                self.sync_receive_time = reception_time

        return events

    def process_follow_up(self, message, reception_time, perfect_time):
        """
        Process a received Follow_Up message.

        Args:
            message: The Follow_Up message
            reception_time: Local time when message was received
            perfect_time: Perfect time when message was received

        Returns:
            List of new events triggered by this message reception
        """
        events = []

        # Only process if we're expecting a Follow_Up
        if not self.pending_follow_up or message.sequence_id != self.last_sync_sequence_id:
            return events

        # Clear pending flag
        self.pending_follow_up = False

        # If this is a slave port receiving from its master
        if self.state == PortState.SLAVE:
            # Get precise timestamp from Follow_Up
            precise_origin_timestamp = message.precise_origin_timestamp

            # Record time before correction for statistics
            gm_time = precise_origin_timestamp + message.correction_field + self.mean_path_delay
            local_time = self.node.clock.get_time()
            time_deviation = local_time - gm_time

            # Record the deviation
            self.node.record_time_deviation(perfect_time, time_deviation)

            # Correct local clock
            self.node.clock.set_time(gm_time)
            self.node.sync_receptions += 1

            # Forward sync to downstream ports if enabled
            if self.node.sync_locked:
                new_events = self.node.forward_follow_up(message, perfect_time, reception_time)
                if new_events:
                    events.extend(new_events)

        return events

    def process_pdelay_req(self, message, reception_time, perfect_time):
        """
        Process a received Pdelay_Req message.

        Args:
            message: The Pdelay_Req message
            reception_time: Local time when message was received
            perfect_time: Perfect time when message was received

        Returns:
            List of new events triggered by this message reception
        """
        events = []

        # Record receipt timestamp (t2) for responding
        t2 = reception_time

        # Create and send Pdelay_Resp
        resp_message = PdelayRespMessage(
            self.port_identity,
            message.sequence_id,
            t2,
            message.source_port_identity
        )

        # Queue sending the response
        send_time = perfect_time
        resp_event = self.send_message(resp_message, send_time)
        if resp_event:
            events.append(resp_event)

        # If using two-step for Pdelay
        if self.node.two_step_pdelay:
            # Record response send time for Follow_Up
            t3 = self.node.clock.get_time()

            # Create and send Pdelay_Resp_Follow_Up
            follow_up_message = PdelayRespFollowUpMessage(
                self.port_identity,
                message.sequence_id,
                t3,
                message.source_port_identity
            )

            # Add small delay before sending follow-up
            follow_up_send_time = perfect_time + 100e-9  # 100 ns after Pdelay_Resp
            follow_up_event = self.send_message(follow_up_message, follow_up_send_time)
            if follow_up_event:
                events.append(follow_up_event)

        return events

    def process_pdelay_resp(self, message, reception_time, perfect_time):
        """
        Process a received Pdelay_Resp message.

        Args:
            message: The Pdelay_Resp message
            reception_time: Local time when message was received
            perfect_time: Perfect time when message was received

        Returns:
            List of new events triggered by this message reception
        """
        events = []

        # Only process if this is a response to our request
        if message.requesting_port_identity != self.port_identity:
            return events

        # Check if this matches our current path delay measurement
        if message.sequence_id != self.current_pdelay_sequence_id:
            return events

        # Record t4 (response receipt time)
        self.t4 = reception_time

        # Store t2 from the message
        self.t2 = message.request_receipt_timestamp

        # If one-step Pdelay mode, calculate path delay now
        if not self.node.two_step_pdelay:
            self.calculate_path_delay(perfect_time)

        return events

    def process_pdelay_resp_follow_up(self, message, reception_time, perfect_time):
        """
        Process a received Pdelay_Resp_Follow_Up message.

        Args:
            message: The Pdelay_Resp_Follow_Up message
            reception_time: Local time when message was received
            perfect_time: Perfect time when message was received

        Returns:
            List of new events triggered by this message reception
        """
        events = []

        # Only process if this is a response to our request
        if message.requesting_port_identity != self.port_identity:
            return events

        # Check if this matches our current path delay measurement
        if message.sequence_id != self.current_pdelay_sequence_id:
            return events

        # Store t3 from the message
        self.t3 = message.response_origin_timestamp

        # Calculate path delay
        self.calculate_path_delay(perfect_time)

        return events

    def calculate_path_delay(self, perfect_time):
        """
        Calculate mean path delay based on collected timestamps.

        Args:
            perfect_time: Perfect time when calculation is performed
        """
        # Calculate neighbor rate ratio (simplified)
        # In a real system, this would be measured over time
        if self.peer_port:
            self.neighbor_rate_ratio = (1 + self.peer_port.node.clock.get_drift_rate() * 1e-6) / \
                                       (1 + self.node.clock.get_drift_rate() * 1e-6)

        # Add error to neighbor rate ratio (up to 0.1 ppm as per standard)
        nr_error = random.uniform(-0.1, 0.1) * 1e-6
        self.neighbor_rate_ratio += nr_error

        # Calculate propagation delay using equation (4) from IEEE 802.1AS
        path_delay = 0.5 * ((self.t4 - self.t1) - self.neighbor_rate_ratio * (self.t3 - self.t2))

        # Update mean path delay with low-pass filtering
        alpha = 0.1  # Filtering coefficient
        self.mean_path_delay = (1 - alpha) * self.mean_path_delay + alpha * path_delay

        # Reset current sequence ID
        self.current_pdelay_sequence_id = None

    def initiate_pdelay_req(self, perfect_time):
        """
        Initiate a path delay request measurement.

        Args:
            perfect_time: Perfect time when measurement is initiated

        Returns:
            Event for scheduler or None
        """
        # Only measure if we have a peer
        if not self.peer_port:
            self.next_pdelay_time = perfect_time + self.pdelay_interval
            return None

        # Create Pdelay_Req message
        self.pdelay_sequence_id += 1
        req_message = PdelayReqMessage(
            self.port_identity,
            self.pdelay_sequence_id
        )

        # Record t1 (request send time)
        self.t1 = self.node.clock.get_time()
        self.current_pdelay_sequence_id = self.pdelay_sequence_id

        # Schedule next pdelay measurement
        self.next_pdelay_time = perfect_time + self.pdelay_interval

        # Send the request
        return self.send_message(req_message, perfect_time)

    def process_announce(self, message, reception_time, perfect_time):
        """
        Process a received Announce message.

        Args:
            message: The Announce message
            reception_time: Local time when message was received
            perfect_time: Perfect time when message was received

        Returns:
            List of new events triggered by this message reception
        """
        # Update announce receipt time
        self.last_announce_receipt_time = perfect_time

        events = []

        # Process BMCA (Best Master Clock Algorithm)
        dataset_changed = self.node.process_announce_bmca(message, self)

        # If we're not the grandmaster and this message affects our state
        if not self.node.is_grandmaster and dataset_changed:
            # Forward Announce message to other ports if needed
            if self.state == PortState.SLAVE:
                new_events = self.node.forward_announce(message, perfect_time, reception_time)
                if new_events:
                    events.extend(new_events)

        return events

    def process_signaling(self, message, reception_time, perfect_time):
        """
        Process a received Signaling message.

        Args:
            message: The Signaling message
            reception_time: Local time when message was received
            perfect_time: Perfect time when message was received

        Returns:
            List of new events triggered by this message reception
        """
        # Process TLVs in the signaling message
        # (In a complete implementation, this would handle various TLVs)
        return []

    def initiate_announce(self, perfect_time):
        """
        Initiate an Announce message.

        Args:
            perfect_time: Perfect time when Announce is initiated

        Returns:
            Event for scheduler or None
        """
        # Only send Announce if this port is in MASTER state
        if self.state != PortState.MASTER:
            self.next_announce_time = perfect_time + self.announce_interval
            return None

        # Create Announce message with current time properties dataset
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

        # Schedule next announce
        self.next_announce_time = perfect_time + self.announce_interval

        # Send the announce
        return self.send_message(announce_message, perfect_time)


class TimeAwareSystem:
    """Implementation of a time-aware system in the IEEE 802.1AS network."""

    def __init__(self, node_id, is_grandmaster=False, priority1=248, priority2=248,
                 phy_jitter_ns=8, num_ports=1, two_step_sync=True, two_step_pdelay=True):
        """
        Initialize a time-aware system.

        Args:
            node_id: Identifier of the node
            is_grandmaster: True if this node is the grandmaster
            priority1: Priority1 value for BMCA
            priority2: Priority2 value for BMCA
            phy_jitter_ns: Maximum PHY jitter in nanoseconds
            num_ports: Number of ports on this node
            two_step_sync: Whether to use two-step sync (True) or one-step (False)
            two_step_pdelay: Whether to use two-step pdelay (True) or one-step (False)
        """
        self.node_id = node_id
        self.is_grandmaster = is_grandmaster
        self.clock = Clock()
        self.clock_identity = ClockIdentity(node_id)
        self.max_phy_jitter = phy_jitter_ns * 1e-9  # Maximum PHY jitter in seconds

        # BMCA-related attributes
        self.priority1 = 128 if is_grandmaster else priority1
        self.priority2 = 128 if is_grandmaster else priority2
        self.clock_quality = ClockQuality(
            clock_class=6 if is_grandmaster else 248,
            clock_accuracy=0x20 if is_grandmaster else 0xFE,
            offset_scaled_log_variance=0x4100 if is_grandmaster else 0xFFFF
        )

        # Current time properties dataset
        self.current_gm_identity = str(self.clock_identity) if is_grandmaster else None
        self.steps_removed = 0 if is_grandmaster else 255

        # Ports
        self.ports = [Port(self, i + 1) for i in range(num_ports)]

        # Sync configuration
        self.sync_locked = True  # syncLocked flag (True for better precision)
        self.sync_interval = 1.0  # syncInterval (1 s)
        self.two_step_sync = two_step_sync  # Whether to use two-step sync mode
        self.two_step_pdelay = two_step_pdelay  # Whether to use two-step pdelay mode

        # Selected port roles
        self.slave_port = None  # Port receiving time from upstream

        # For statistics and analysis
        self.time_deviations = []  # Record of time deviations from GM
        self.sync_receptions = 0  # Count of sync messages received

    def record_time_deviation(self, perfect_time, deviation):
        """
        Record a time deviation for analysis.

        Args:
            perfect_time: Perfect time when the deviation was measured
            deviation: Time deviation from grandmaster
        """
        self.time_deviations.append((perfect_time, deviation))

    def process_announce_bmca(self, announce_msg, receive_port):
        """
        Process an Announce message for the Best Master Clock Algorithm.

        Args:
            announce_msg: The received Announce message
            receive_port: Port that received the Announce

        Returns:
            True if dataset changed, False otherwise
        """
        # Skip BMCA if we're the grandmaster
        if self.is_grandmaster:
            return False

        # Compare announced GM dataset with current dataset
        better_master = False

        # If we don't have a GM yet
        if self.current_gm_identity is None:
            better_master = True
        else:
            # Compare priority1
            if announce_msg.gm_priority1 < self.priority1:
                better_master = True
            elif announce_msg.gm_priority1 > self.priority1:
                better_master = False
            # Compare clock quality
            elif announce_msg.gm_clock_quality.is_better_than(self.clock_quality):
                better_master = True
            elif not announce_msg.gm_clock_quality.is_better_than(
                    self.clock_quality) and self.clock_quality.is_better_than(announce_msg.gm_clock_quality):
                better_master = False
            # Compare priority2
            elif announce_msg.gm_priority2 < self.priority2:
                better_master = True
            elif announce_msg.gm_priority2 > self.priority2:
                better_master = False
            # Compare GM identity
            elif announce_msg.gm_identity < self.current_gm_identity:
                better_master = True

        # If we found a better master
        if better_master:
            # Update dataset
            self.current_gm_identity = announce_msg.gm_identity
            self.steps_removed = announce_msg.steps_removed + 1

            # Update port states
            old_slave_port = self.slave_port
            self.slave_port = receive_port

            # Set receiving port to SLAVE
            receive_port.state = PortState.SLAVE

            # Set other ports appropriately
            for port in self.ports:
                if port != receive_port:
                    if port == old_slave_port:
                        port.state = PortState.MASTER
                    elif port.state != PortState.MASTER:
                        port.state = PortState.MASTER

            return True

        return False

    def initiate_sync(self, perfect_time):
        """
        Initiate a Sync message (only for grandmaster or ports in MASTER state).

        Args:
            perfect_time: Perfect time when sync is initiated

        Returns:
            List of sync events to be processed
        """
        sync_events = []

        # For grandmaster or nodes with master ports
        for port in self.ports:
            if port.state == PortState.MASTER:
                # Set precise origin timestamp to current time
                origin_timestamp = None if self.two_step_sync else self.clock.get_time()

                # Create Sync message
                port.sync_sequence_id += 1
                sync_message = SyncMessage(
                    port.port_identity,
                    port.sync_sequence_id,
                    origin_timestamp,
                    self.two_step_sync
                )

                # Store send time for Follow_Up if using two-step
                if self.two_step_sync:
                    port.sync_send_times[port.sync_sequence_id] = self.clock.get_time()

                # Send the sync
                sync_event = port.send_message(sync_message, perfect_time)
                if sync_event:
                    sync_events.append(sync_event)

                # If two-step, also create and send Follow_Up
                if self.two_step_sync:
                    # Create Follow_Up message
                    follow_up_message = FollowUpMessage(
                        port.port_identity,
                        port.sync_sequence_id,
                        port.sync_send_times[port.sync_sequence_id]
                    )

                    # Add small delay before sending follow-up
                    follow_up_send_time = perfect_time + 100e-9  # 100 ns after Sync
                    follow_up_event = port.send_message(follow_up_message, follow_up_send_time)
                    if follow_up_event:
                        sync_events.append(follow_up_event)

                    # Clean up stored send time
                    del port.sync_send_times[port.sync_sequence_id]

        return sync_events

    def forward_sync(self, sync_message, perfect_time, reception_time):
        """
        Forward a Sync message to downstream ports.

        Args:
            sync_message: The Sync message to forward
            perfect_time: Perfect time when message is forwarded
            reception_time: Local time when message was received

        Returns:
            List of sync events to be processed
        """
        sync_events = []

        # Calculate residence time
        residence_time = min(random.uniform(0, 1e-3), 1e-3)  # Up to 1 ms residence time

        # Forward to all MASTER ports
        for port in self.ports:
            if port.state == PortState.MASTER:
                # Create a new Sync message
                port.sync_sequence_id += 1

                # For one-step sync
                origin_timestamp = None
                if not self.two_step_sync and not sync_message.two_step_flag:
                    # Update correction field for one-step
                    correction_field = sync_message.correction_field + \
                                       port.mean_path_delay + \
                                       (residence_time * self.get_rate_ratio())

                    # Create one-step Sync with original timestamp
                    forwarded_sync = SyncMessage(
                        port.port_identity,
                        port.sync_sequence_id,
                        sync_message.origin_timestamp,
                        False
                    )
                    forwarded_sync.correction_field = correction_field

                # For two-step sync
                else:
                    # Create two-step Sync
                    forwarded_sync = SyncMessage(
                        port.port_identity,
                        port.sync_sequence_id,
                        None,
                        True
                    )
                    forwarded_sync.correction_field = sync_message.correction_field

                    # Store send time for Follow_Up
                    port.sync_send_times[port.sync_sequence_id] = self.clock.get_time()

                # Send the sync
                sync_event = port.send_message(forwarded_sync, perfect_time + residence_time)
                if sync_event:
                    sync_events.append(sync_event)

        return sync_events

    def forward_follow_up(self, follow_up_message, perfect_time, reception_time):
        """
        Forward a Follow_Up message to downstream ports.

        Args:
            follow_up_message: The Follow_Up message to forward
            perfect_time: Perfect time when message is forwarded
            reception_time: Local time when message was received

        Returns:
            List of events to be processed
        """
        events = []

        # Calculate residence time
        residence_time = min(random.uniform(0, 1e-4), 1e-4)  # Up to 100 µs residence time

        # Forward to all MASTER ports
        for port in self.ports:
            if port.state == PortState.MASTER and port.sync_sequence_id in port.sync_send_times:
                # Create a new Follow_Up message
                follow_up = FollowUpMessage(
                    port.port_identity,
                    port.sync_sequence_id,
                    port.sync_send_times[port.sync_sequence_id]
                )

                # Update correction field
                follow_up.correction_field = follow_up_message.correction_field + \
                                             self.slave_port.mean_path_delay + \
                                             (residence_time * self.get_rate_ratio())

                # Send the Follow_Up
                event = port.send_message(follow_up, perfect_time + residence_time)
                if event:
                    events.append(event)

                # Clean up stored send time
                del port.sync_send_times[port.sync_sequence_id]

        return events

    def forward_announce(self, announce_message, perfect_time, reception_time):
        """
        Forward an Announce message to downstream ports.

        Args:
            announce_message: The Announce message to forward
            perfect_time: Perfect time when message is forwarded
            reception_time: Local time when message was received

        Returns:
            List of events to be processed
        """
        events = []

        # Calculate residence time
        residence_time = min(random.uniform(0, 1e-4), 1e-4)  # Up to 100 µs residence time

        # Forward to all MASTER ports
        for port in self.ports:
            if port.state == PortState.MASTER:
                # Create a new Announce message
                port.announce_sequence_id += 1
                announce = AnnounceMessage(
                    port.port_identity,
                    port.announce_sequence_id,
                    announce_message.gm_priority1,
                    announce_message.gm_clock_quality,
                    announce_message.gm_priority2,
                    announce_message.gm_identity,
                    self.steps_removed  # Increment steps removed
                )

                # Send the Announce
                event = port.send_message(announce, perfect_time + residence_time)
                if event:
                    events.append(event)

        return events

    def get_rate_ratio(self):
        """Get the cumulative rate ratio from GM to this node."""
        if self.is_grandmaster or self.slave_port is None:
            return 1.0

        # In a real implementation, this would be computed based on observed frequency ratios
        rate_ratio = 1.0
        if self.slave_port:
            rate_ratio = self.slave_port.neighbor_rate_ratio

        return rate_ratio


class IEEE8021ASSimulation:
    """Simulation of IEEE 802.1AS network with complete message set."""

    def __init__(self, num_nodes=100, simulation_time=600.0, two_step_sync=True, linear_topology=True):
        """
        Initialize the simulation.

        Args:
            num_nodes: Number of time-aware systems in the network
            simulation_time: Total simulation time in seconds
            two_step_sync: Whether to use two-step sync (True) or one-step (False)
            linear_topology: Whether to use linear topology or more complex topology
        """
        self.num_nodes = num_nodes
        self.simulation_time = simulation_time
        self.perfect_time = 0.0
        self.nodes = []
        self.events = []  # (time, event_type, params)
        self.two_step_sync = two_step_sync

        # Create nodes
        for i in range(num_nodes):
            is_gm = (i == 0)
            node = TimeAwareSystem(
                i,
                is_grandmaster=is_gm,
                priority1=128 if is_gm else 248,
                priority2=128 if is_gm else 248,
                two_step_sync=two_step_sync
            )
            self.nodes.append(node)

        # Connect nodes in a linear topology
        if linear_topology:
            self._setup_linear_topology()
        else:
            self._setup_complex_topology()

    def _setup_linear_topology(self):
        """Set up a linear topology with connections between adjacent nodes."""
        for i in range(self.num_nodes - 1):
            # Connect port 1 of node i to port 1 of node i+1
            self.nodes[i].ports[0].connect(self.nodes[i + 1].ports[0])

            # Set initial port states
            if i == 0:  # GM
                self.nodes[i].ports[0].state = PortState.MASTER
            else:
                self.nodes[i].ports[0].state = PortState.SLAVE
                self.nodes[i].slave_port = self.nodes[i].ports[0]

            if i < self.num_nodes - 2:  # Not the last node
                self.nodes[i + 1].ports[0].state = PortState.MASTER
            else:  # Last node
                self.nodes[i + 1].ports[0].state = PortState.SLAVE
                self.nodes[i + 1].slave_port = self.nodes[i + 1].ports[0]

    def _setup_complex_topology(self):
        """
        Set up a more complex topology.
        This is a simplified example - in a full implementation,
        this would create a more complex network structure.
        """
        # For simplicity, still create a linear topology
        # A full implementation would create a more complex graph
        self._setup_linear_topology()

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
            if event_type == "message":
                self._process_message_event(*params)
            elif event_type == "initiate_sync":
                self._process_initiate_sync(params)
            elif event_type == "initiate_announce":
                self._process_initiate_announce(params)
            elif event_type == "initiate_pdelay":
                self._process_initiate_pdelay(params)

        # Collect and analyze results
        return self._analyze_results()

    def _schedule_initial_events(self):
        """Schedule initial events."""
        # Schedule initial sync from grandmaster and all master ports
        for i in range(self.num_nodes):
            for port in self.nodes[i].ports:
                if port.state == PortState.MASTER:
                    # Initialize sync
                    self.events.append((0.0, "initiate_sync", (i, port.port_number)))

                    # Initialize announce
                    announce_time = random.uniform(0, 0.1)  # Slightly randomize announce times
                    self.events.append((announce_time, "initiate_announce", (i, port.port_number)))

        # Schedule initial propagation delay measurements
        for i in range(self.num_nodes):
            for port in self.nodes[i].ports:
                # Stagger initial pdelay measurements
                pdelay_time = random.uniform(0, 1.0)
                self.events.append((pdelay_time, "initiate_pdelay", (i, port.port_number)))

    def _update_clocks(self, dt):
        """
        Update all clocks based on elapsed perfect time.

        Args:
            dt: Elapsed perfect time in seconds
        """
        for node in self.nodes:
            node.clock.update(dt)

    def _process_message_event(self, port, message):
        """Process a message reception event."""
        # Get receiving node and port
        receive_port = port
        node = receive_port.node

        # Process the message
        new_events = receive_port.receive_message(message, self.perfect_time)

        # Schedule any new events
        if new_events:
            for event in new_events:
                arrival_time, target_port, new_message = event
                self._insert_event((arrival_time, "message", (target_port, new_message)))

    def _process_initiate_sync(self, params):
        """Process a sync initiation event."""
        node_id, port_number = params
        node = self.nodes[node_id]

        # Initiate sync
        sync_events = node.initiate_sync(self.perfect_time)

        # Schedule any new events
        if sync_events:
            for event in sync_events:
                arrival_time, target_port, message = event
                self._insert_event((arrival_time, "message", (target_port, message)))

        # Schedule next sync initiation for all master ports
        for port in node.ports:
            if port.state == PortState.MASTER:
                next_sync_time = self.perfect_time + node.sync_interval
                self._insert_event((next_sync_time, "initiate_sync", (node_id, port.port_number)))

    def _process_initiate_announce(self, params):
        """Process an announce initiation event."""
        node_id, port_number = params
        node = self.nodes[node_id]
        port = node.ports[port_number - 1]  # Port numbers are 1-based

        # Initiate announce
        announce_event = port.initiate_announce(self.perfect_time)

        # Schedule any new events
        if announce_event:
            arrival_time, target_port, message = announce_event
            self._insert_event((arrival_time, "message", (target_port, message)))

        # Schedule next announce
        next_announce_time = port.next_announce_time
        self._insert_event((next_announce_time, "initiate_announce", (node_id, port_number)))

    def _process_initiate_pdelay(self, params):
        """Process a pdelay initiation event."""
        node_id, port_number = params
        node = self.nodes[node_id]
        port = node.ports[port_number - 1]  # Port numbers are 1-based

        # Initiate pdelay
        pdelay_event = port.initiate_pdelay_req(self.perfect_time)

        # Schedule any new events
        if pdelay_event:
            arrival_time, target_port, message = pdelay_event
            self._insert_event((arrival_time, "message", (target_port, message)))

        # Schedule next pdelay
        next_pdelay_time = port.next_pdelay_time
        self._insert_event((next_pdelay_time, "initiate_pdelay", (node_id, port_number)))

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
                for port in self.nodes[i].ports:
                    if port.state == PortState.SLAVE:
                        results["propagation_delays"].append(port.mean_path_delay)
                        break

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
    plt.show()


def compare_sync_modes(one_step_results, two_step_results):
    """
    Compare one-step and two-step synchronization modes.

    Args:
        one_step_results: Results from one-step simulation
        two_step_results: Results from two-step simulation
    """
    plt.figure(figsize=(14, 10))

    # Create subplots for each monitored node
    monitored_nodes = [10, 25, 50, 75, 100]
    num_nodes = len(monitored_nodes)

    for i, node_id in enumerate(monitored_nodes):
        plt.subplot(num_nodes, 1, i + 1)

        # Plot one-step deviations
        if node_id in one_step_results["node_deviations_time_series"]:
            time_series = one_step_results["node_deviations_time_series"][node_id]
            times = [t for t, _ in time_series]
            deviations = [d * 1e6 for _, d in time_series]  # Convert to microseconds
            plt.plot(times, deviations, 'r-', label='One-Step', alpha=0.7)

        # Plot two-step deviations
        if node_id in two_step_results["node_deviations_time_series"]:
            time_series = two_step_results["node_deviations_time_series"][node_id]
            times = [t for t, _ in time_series]
            deviations = [d * 1e6 for _, d in time_series]  # Convert to microseconds
            plt.plot(times, deviations, 'b-', label='Two-Step', alpha=0.7)

        plt.title(f'Node {node_id}')
        plt.ylabel('Deviation (µs)')
        plt.grid(True)

        if i == 0:
            plt.legend()

    plt.xlabel('Simulation Time (s)')
    plt.tight_layout()
    plt.savefig('one_step_vs_two_step.png')
    plt.show()


if __name__ == "__main__":
    print("Running IEEE 802.1AS simulation with complete message set and two-step sync...")

    # Run simulation with two-step sync (default)
    two_step_sim = IEEE8021ASSimulation(num_nodes=101, simulation_time=600.0, two_step_sync=True)
    two_step_results = two_step_sim.run()

    # Run simulation with one-step sync
    one_step_sim = IEEE8021ASSimulation(num_nodes=101, simulation_time=600.0, two_step_sync=False)
    one_step_results = one_step_sim.run()

    # Analyze results
    two_step_analysis = analyze_sync_precision(two_step_results)
    one_step_analysis = analyze_sync_precision(one_step_results)

    # Print summary for two-step
    print("\nTwo-Step Sync Results:")
    print(f"Simulation completed with {two_step_sim.num_nodes} nodes for {two_step_sim.simulation_time} seconds")
    print(f"Overall synchronization precision: {two_step_analysis['overall_precision'] * 1e6:.3f} µs")

    # Print summary for one-step
    print("\nOne-Step Sync Results:")
    print(f"Simulation completed with {one_step_sim.num_nodes} nodes for {one_step_sim.simulation_time} seconds")
    print(f"Overall synchronization precision: {one_step_analysis['overall_precision'] * 1e6:.3f} µs")

    # Print detailed results for specific nodes
    selected_nodes = [10, 25, 50, 75, 100]
    print("\nDetailed results for selected nodes (Two-Step):")
    for node in selected_nodes:
        if node in two_step_analysis["max_deviations"]:
            max_dev = two_step_analysis["max_deviations"][node]
            sync_prob = two_step_analysis["sync_probabilities"][node]
            print(f"Node {node}: Max deviation = {max_dev * 1e6:.3f} µs, "
                  f"Sync probability (1 µs) = {sync_prob:.2%}")

    print("\nDetailed results for selected nodes (One-Step):")
    for node in selected_nodes:
        if node in one_step_analysis["max_deviations"]:
            max_dev = one_step_analysis["max_deviations"][node]
            sync_prob = one_step_analysis["sync_probabilities"][node]
            print(f"Node {node}: Max deviation = {max_dev * 1e6:.3f} µs, "
                  f"Sync probability (1 µs) = {sync_prob:.2%}")

    # Plot time deviations for two-step
    print("\nPlotting time deviations (Two-Step)...")
    plot_time_deviations(two_step_results)

    # Compare one-step and two-step modes
    print("\nComparing one-step and two-step synchronization modes...")
    compare_sync_modes(one_step_results, two_step_results)