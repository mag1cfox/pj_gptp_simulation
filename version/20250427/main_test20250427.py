"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/4/27 13:36
*  @Project :   pj_gptp_simulation
*  @Description :   Description
*  @FileName:   main_test20250427.py
**************************************
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import random
from matplotlib.ticker import PercentFormatter


@dataclass
class SimulationParameters:
    """Parameters for the time synchronization simulation"""
    # Network configuration
    num_hops: int = 100  # Number of hops in the chain
    num_runs: int = 10000  # Number of Monte Carlo runs

    # Clock characteristics
    gm_clock_drift_max: float = 1.5  # Maximum GM clock drift in ppm/s
    gm_clock_drift_min: float = -1.5  # Minimum GM clock drift in ppm/s
    gm_clock_drift_fraction: float = 0.8  # Fraction of GM nodes with drift

    clock_drift_max: float = 1.5  # Maximum non-GM clock drift in ppm/s
    clock_drift_min: float = -1.5  # Minimum non-GM clock drift in ppm/s
    clock_drift_fraction: float = 0.8  # Fraction of non-GM nodes with drift

    # Temperature model parameters
    temp_max: float = 85.0  # Maximum temperature (°C)
    temp_min: float = -40.0  # Minimum temperature (°C)
    temp_ramp_rate: float = 1.0  # Temperature change rate (°C/s)
    temp_hold_period: float = 30.0  # Hold time at temperature extremes (s)
    gm_scaling_factor: float = 1.0  # GM drift scaling factor
    non_gm_scaling_factor: float = 1.0  # Non-GM drift scaling factor

    # Timestamp error characteristics
    tsge_tx: float = 4.0  # Timestamp Granularity Error for TX (±ns)
    tsge_rx: float = 4.0  # Timestamp Granularity Error for RX (±ns)
    dtse_tx: float = 4.0  # Dynamic Timestamp Error for TX (±ns)
    dtse_rx: float = 4.0  # Dynamic Timestamp Error for RX (±ns)

    # Message intervals
    pdelay_interval: float = 125.0  # pDelay message interval (ms)
    sync_interval: float = 125.0  # Sync message interval (ms)
    pdelay_turnaround: float = 10.0  # pDelay response time (ms)
    residence_time: float = 10.0  # Residence time within a node (ms)

    # Correction factors
    mean_link_delay_correction: float = 0.0  # Effectiveness of Mean Link Delay averaging
    nrr_drift_correction: float = 0.0  # NRR drift correction effectiveness
    rr_drift_correction: float = 0.0  # RR drift correction effectiveness
    pdelayresp_sync_correction: float = 0.0  # pDelay response-to-sync alignment factor

    # NRR smoothing parameters
    mnrr_smoothing_n: int = 1  # Number of previous pDelayResp to use
    mnrr_smoothing_m: int = 1  # For median calculation (not used in recommended settings)

    # Use temperature-based model or simple uniform model
    use_temperature_model: bool = True


@dataclass
class NodeState:
    """State of a node in the chain"""
    # Clock-related state
    clock_drift: float = 0.0  # Clock drift rate in ppm/s

    # Timestamp errors
    t1_pderror: float = 0.0  # TX timestamp error for pDelay request
    t2_pderror: float = 0.0  # RX timestamp error for pDelay request
    t3_pderror: float = 0.0  # TX timestamp error for pDelay response
    t4_pderror: float = 0.0  # RX timestamp error for pDelay response
    t3_pderror_prev: List[float] = field(default_factory=list)  # Previous t3 errors for NRR calculation
    t4_pderror_prev: List[float] = field(default_factory=list)  # Previous t4 errors for NRR calculation

    t2_sinerror: float = 0.0  # RX timestamp error for Sync
    t1_souterror: float = 0.0  # TX timestamp error for Sync

    # Error accumulation
    mnrr_error: float = 0.0  # Neighbor Rate Ratio error
    mnrr_error_ts: float = 0.0  # NRR error due to timestamp errors
    mnrr_error_cd: float = 0.0  # NRR error due to clock drift

    rr_error: float = 0.0  # Rate Ratio error
    rr_error_sum: float = 0.0  # Accumulated RR error
    rr_error_components: Dict[str, float] = field(default_factory=dict)  # Components of RR error

    mean_link_delay_error: float = 0.0  # Link delay measurement error
    mean_link_delay_components: Dict[str, float] = field(default_factory=dict)  # Components of MLD error

    residence_time_error: float = 0.0  # Residence time measurement error
    residence_time_components: Dict[str, float] = field(default_factory=dict)  # Components of RT error

    end_station_error: float = 0.0  # End station error (only for last hop)
    end_station_components: Dict[str, float] = field(default_factory=dict)  # Components of ES error

    dte: float = 0.0  # Dynamic Time Error at this node


class TimeSyncSimulation:
    """Time synchronization simulation for IEEE 802.1AS in IEC/IEEE 60802"""

    def __init__(self, params: SimulationParameters):
        self.params = params
        self.results = {
            'dte_max': [],  # Maximum DTE across all runs
            'dte_7sigma': [],  # 7-sigma value of DTE
            'dte_per_hop': np.zeros((params.num_runs, params.num_hops))  # DTE at each hop for each run
        }

        # Create output directories if they don't exist
        os.makedirs('output_data', exist_ok=True)
        os.makedirs('output_image', exist_ok=True)

    def generate_timestamp_error(self, is_tx: bool) -> float:
        """Generate a random timestamp error based on parameters"""
        if is_tx:
            tsge = np.random.uniform(-self.params.tsge_tx, self.params.tsge_tx)
            dtse = np.random.uniform(-self.params.dtse_tx, self.params.dtse_tx)
        else:
            tsge = np.random.uniform(-self.params.tsge_rx, self.params.tsge_rx)
            dtse = np.random.uniform(-self.params.dtse_rx, self.params.dtse_rx)
        return tsge + dtse

    def generate_clock_drift(self, is_gm: bool) -> float:
        """Generate random clock drift based on parameters"""
        if self.params.use_temperature_model:
            return self.generate_clock_drift_temperature_based(is_gm)
        else:
            # Simple uniform distribution model
            if is_gm:
                if np.random.random() <= self.params.gm_clock_drift_fraction:
                    return np.random.uniform(self.params.gm_clock_drift_min, self.params.gm_clock_drift_max)
                return 0.0
            else:
                if np.random.random() <= self.params.clock_drift_fraction:
                    return np.random.uniform(self.params.clock_drift_min, self.params.clock_drift_max)
                return 0.0

    def generate_clock_drift_temperature_based(self, is_gm: bool) -> float:
        """Generate clock drift based on temperature model"""
        # Determine if this node should have drift
        if (is_gm and np.random.random() > self.params.gm_clock_drift_fraction) or \
                (not is_gm and np.random.random() > self.params.clock_drift_fraction):
            return 0.0

        # Calculate temperature cycle parameters
        temp_cycle_period = ((self.params.temp_max - self.params.temp_min) / self.params.temp_ramp_rate) * 2 + \
                            2 * self.params.temp_hold_period

        # Random point in temperature cycle
        t = np.random.uniform(0, temp_cycle_period)

        # Find section boundaries
        section_a = (self.params.temp_max - self.params.temp_min) / self.params.temp_ramp_rate
        section_b = section_a + self.params.temp_hold_period
        section_c = section_b + section_a

        # Calculate temperature and temperature rate of change
        if t < section_a:
            # Ramp up
            temp_xo = self.params.temp_min + self.params.temp_ramp_rate * t
            temp_roc = self.params.temp_ramp_rate
        elif t < section_b:
            # Hold at max
            temp_xo = self.params.temp_max
            temp_roc = 0
        elif t < section_c:
            # Ramp down
            temp_xo = self.params.temp_max - self.params.temp_ramp_rate * (t - section_b)
            temp_roc = -self.params.temp_ramp_rate
        else:
            # Hold at min
            temp_xo = self.params.temp_min
            temp_roc = 0

        # Cubic model constants
        a, b, c, d = 0.00012, -0.01005, -0.0305, 5.73845

        # Calculate clock drift
        clock_drift = (3 * a * temp_xo ** 2 + 2 * b * temp_xo + c) * temp_roc

        # Apply scaling factor
        if is_gm:
            clock_drift *= self.params.gm_scaling_factor
        else:
            clock_drift *= self.params.non_gm_scaling_factor

        return clock_drift

    def generate_pdelay_interval(self) -> float:
        """Generate random pDelay interval within spec"""
        return np.random.uniform(0.9 * self.params.pdelay_interval,
                                 1.3 * self.params.pdelay_interval)

    def run_simulation(self):
        """Run the time sync simulation"""
        for run in range(self.params.num_runs):
            if run % 1000 == 0:
                print(f"Running simulation {run}/{self.params.num_runs}...")

            # Reset for new run
            nodes = [NodeState() for _ in range(self.params.num_hops + 1)]  # +1 for GM

            # Generate clock drifts for all nodes
            nodes[0].clock_drift = self.generate_clock_drift(is_gm=True)  # GM
            for i in range(1, self.params.num_hops + 1):
                nodes[i].clock_drift = self.generate_clock_drift(is_gm=False)

            # Calculate errors across all hops
            dte = 0.0
            for hop in range(1, self.params.num_hops + 1):
                # Generate timestamp errors
                nodes[hop].t1_pderror = self.generate_timestamp_error(is_tx=True)
                nodes[hop].t2_pderror = self.generate_timestamp_error(is_tx=False)
                nodes[hop].t3_pderror = self.generate_timestamp_error(is_tx=True)
                nodes[hop].t4_pderror = self.generate_timestamp_error(is_tx=False)
                nodes[hop].t1_souterror = self.generate_timestamp_error(is_tx=True)
                nodes[hop].t2_sinerror = self.generate_timestamp_error(is_tx=False)

                # Generate previous timestamps for NRR calculation
                for n in range(1, self.params.mnrr_smoothing_n):
                    nodes[hop].t3_pderror_prev.append(self.generate_timestamp_error(is_tx=True))
                    nodes[hop].t4_pderror_prev.append(self.generate_timestamp_error(is_tx=False))

                # Calculate NRR error components
                self.calculate_mnrr_errors(nodes, hop)

                # Calculate RR error
                self.calculate_rr_error(nodes, hop)

                # Calculate Mean Link Delay error
                self.calculate_mean_link_delay_error(nodes, hop)

                # Calculate Residence Time error or End Station error
                if hop < self.params.num_hops:  # Not the last hop
                    self.calculate_residence_time_error(nodes, hop)
                    nodes[hop].dte = dte + nodes[hop].mean_link_delay_error + nodes[hop].residence_time_error
                else:  # Last hop (End Station)
                    self.calculate_end_station_error(nodes, hop)
                    nodes[hop].dte = dte + nodes[hop].mean_link_delay_error + nodes[hop].end_station_error

                # Update accumulated DTE for next hop
                dte = nodes[hop].dte

                # Store results
                self.results['dte_per_hop'][run, hop - 1] = dte

        # Calculate statistics
        for hop in range(self.params.num_hops):
            dte_at_hop = self.results['dte_per_hop'][:, hop]
            max_abs_dte = np.max(np.abs(dte_at_hop))
            self.results['dte_max'].append(max_abs_dte)
            self.results['dte_7sigma'].append(np.std(dte_at_hop) * 7)

        # Save results to CSV
        self.save_results_to_csv()

    def calculate_mnrr_errors(self, nodes: List[NodeState], hop: int):
        """Calculate mNRR error components for a given hop"""
        # Calculate effective pDelay interval based on mNRR smoothing
        tpdelay2pdelay = 0
        for n in range(self.params.mnrr_smoothing_n):
            tpdelay2pdelay += self.generate_pdelay_interval()

        # Calculate timestamp-induced mNRR error
        if self.params.mnrr_smoothing_n > 1 and len(nodes[hop].t3_pderror_prev) >= self.params.mnrr_smoothing_n - 1:
            # Use previous timestamps for NRR calculation
            t3pd_diff = nodes[hop].t3_pderror - nodes[hop].t3_pderror_prev[-1]
            t4pd_diff = nodes[hop].t4_pderror - nodes[hop].t4_pderror_prev[-1]
        else:
            # Default calculation with most recent timestamps
            t3pd_diff = nodes[hop].t3_pderror - 0  # Assuming previous sample has 0 error (simplified)
            t4pd_diff = nodes[hop].t4_pderror - 0

        nodes[hop].mnrr_error_ts = (t3pd_diff - t4pd_diff) / tpdelay2pdelay

        # Calculate clock-drift-induced mNRR error
        nodes[hop].mnrr_error_cd = (tpdelay2pdelay * (nodes[hop].clock_drift - nodes[hop - 1].clock_drift) / (
                    2 * 1000)) * (1.0 - self.params.nrr_drift_correction)

        # Total mNRR error
        nodes[hop].mnrr_error = nodes[hop].mnrr_error_ts + nodes[hop].mnrr_error_cd

    def calculate_rr_error(self, nodes: List[NodeState], hop: int):
        """Calculate RR error components with improved model"""
        if hop == 1:
            # First hop RR error is just the NRR error
            nodes[hop].rr_error = nodes[hop].mnrr_error
            nodes[hop].rr_error_components = {
                'mnrr_ts': nodes[hop].mnrr_error_ts,
                'mnrr_cd': nodes[hop].mnrr_error_cd,
                'cd_direct': 0.0,
                'gm_impact': 0.0
            }
        else:
            # Special handling of GM clock drift impact
            gm_impact = 0
            for h in range(1, hop):
                gm_impact += self.params.residence_time * (nodes[0].clock_drift - nodes[h].clock_drift) / 1000 * (
                            1.0 - self.params.rr_drift_correction)

            # Clock drift between NRR measurement and Sync
            pdelay_to_sync = np.random.uniform(0, self.params.pdelay_interval) * (
                        1.0 - self.params.pdelayresp_sync_correction)
            cd_direct = (pdelay_to_sync * (nodes[hop].clock_drift - nodes[hop - 1].clock_drift) / 1000) * (
                        1.0 - self.params.nrr_drift_correction)

            # Calculate accumulated RR error
            nodes[hop].rr_error = nodes[hop - 1].rr_error + nodes[hop].mnrr_error + cd_direct + gm_impact

            # Store components for analysis
            nodes[hop].rr_error_components = {
                'mnrr_ts': nodes[hop].mnrr_error_ts,
                'mnrr_cd': nodes[hop].mnrr_error_cd,
                'cd_direct': cd_direct,
                'gm_impact': gm_impact,
                'upstream_rr': nodes[hop - 1].rr_error
            }

    def calculate_mean_link_delay_error(self, nodes: List[NodeState], hop: int):
        """Calculate Mean Link Delay error components"""
        # Timestamp error component
        pdelay_error_ts = (nodes[hop].t4_pderror - nodes[hop].t1_pderror - nodes[hop].t3_pderror + nodes[
            hop].t2_pderror) / 2
        pdelay_error_ts *= (1.0 - self.params.mean_link_delay_correction)

        # NRR error component
        pdelay_error_nrr = -self.params.pdelay_turnaround * nodes[hop].mnrr_error / 2
        pdelay_error_nrr *= (1.0 - self.params.mean_link_delay_correction)

        # Combined error
        nodes[hop].mean_link_delay_error = pdelay_error_ts + pdelay_error_nrr

        # Store components
        nodes[hop].mean_link_delay_components = {
            'ts_direct': pdelay_error_ts,
            'nrr': pdelay_error_nrr
        }

    def calculate_residence_time_error(self, nodes: List[NodeState], hop: int):
        """Calculate Residence Time error components"""
        # Direct timestamp error
        rt_error_ts_direct = nodes[hop].t1_souterror - nodes[hop].t2_sinerror

        # RR-induced error
        rt_error_rr = self.params.residence_time * nodes[hop].rr_error

        # Clock drift direct effect
        rt_error_cd_direct = (self.params.residence_time ** 2 * (nodes[hop].clock_drift - nodes[0].clock_drift) / (
                    2 * 1000)) * (1.0 - self.params.rr_drift_correction)

        # Combined error
        nodes[hop].residence_time_error = rt_error_ts_direct + rt_error_rr + rt_error_cd_direct

        # Store components
        nodes[hop].residence_time_components = {
            'ts_direct': rt_error_ts_direct,
            'rr': rt_error_rr,
            'cd_direct': rt_error_cd_direct
        }

    def calculate_end_station_error(self, nodes: List[NodeState], hop: int):
        """Calculate End Station error components with improved model"""
        # Use gamma distribution for sync interval
        sync_interval = np.random.gamma(270.5532, self.params.sync_interval / 270.5532)

        # RR component
        es_error_rr = sync_interval * nodes[hop].rr_error

        # Clock drift direct effect
        es_error_cd_direct = (sync_interval / 2 * (nodes[hop].clock_drift - nodes[0].clock_drift) / 1000) * (
                    1.0 - self.params.rr_drift_correction)

        # Frequency offset effect
        a, b, c, d = 0.00012, -0.01005, -0.0305, 5.73845
        temp = np.random.uniform(self.params.temp_min, self.params.temp_max)
        freq_offset = a * temp ** 3 + b * temp ** 2 + c * temp + d
        es_error_freq_offset = sync_interval * freq_offset / 1e6

        # Combined error
        nodes[hop].end_station_error = es_error_rr + es_error_cd_direct + es_error_freq_offset

        # Store components
        nodes[hop].end_station_components = {
            'rr': es_error_rr,
            'cd_direct': es_error_cd_direct,
            'freq_offset': es_error_freq_offset
        }

    def save_results_to_csv(self):
        """Save simulation results to CSV files"""
        # Create DataFrame for DTE at each hop for each run
        dte_df = pd.DataFrame(self.results['dte_per_hop'])
        dte_df.columns = [f'Hop_{i + 1}' for i in range(self.params.num_hops)]

        # Save to CSV
        dte_df.to_csv('output_data/dte_all_runs.csv', index=False)

        # Create summary statistics DataFrame
        summary_df = pd.DataFrame({
            'Hop': range(1, self.params.num_hops + 1),
            'Max_Abs_DTE': self.results['dte_max'],
            'DTE_7sigma': self.results['dte_7sigma'],
            'Mean_DTE': [np.mean(self.results['dte_per_hop'][:, i]) for i in range(self.params.num_hops)],
            'StdDev_DTE': [np.std(self.results['dte_per_hop'][:, i]) for i in range(self.params.num_hops)]
        })

        # Save to CSV
        summary_df.to_csv('output_data/dte_summary_stats.csv', index=False)

        print(f"Results saved to output_data folder.")

    def plot_results(self):
        """Plot and save simulation results"""
        self.plot_dte_distribution()
        self.plot_dte_growth()
        self.plot_early_hops()
        self.plot_selected_hops_line()
        self.plot_selected_hops_cdf()

    def plot_dte_distribution(self):
        """Plot DTE distribution at final hop"""
        plt.figure(figsize=(10, 6))

        final_hop_dte = self.results['dte_per_hop'][:, -1]
        plt.hist(final_hop_dte, bins=50, alpha=0.7, color='royalblue')

        plt.axvline(x=self.results['dte_7sigma'][-1], color='crimson', linestyle='--',
                    label=f'7σ: {self.results["dte_7sigma"][-1]:.1f} ns')
        plt.axvline(x=-self.results['dte_7sigma'][-1], color='crimson', linestyle='--')

        plt.axvline(x=1000, color='forestgreen', linestyle=':', label='±1μs target')
        plt.axvline(x=-1000, color='forestgreen', linestyle=':')

        plt.xlabel('Dynamic Time Error (ns)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title(f'DTE Distribution at Hop {self.params.num_hops}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)

        plt.tight_layout()
        plt.savefig('output_image/dte_distribution.png', dpi=300)
        plt.close()

    def plot_dte_growth(self):
        """Plot DTE growth across all hops"""
        plt.figure(figsize=(12, 7))

        hops = np.arange(1, self.params.num_hops + 1)
        plt.plot(hops, self.results['dte_7sigma'], 'b-', linewidth=2, label='7σ DTE')

        # Add target line
        plt.axhline(y=1000, color='forestgreen', linestyle=':', linewidth=2, label='±1μs target')

        # Add max line
        plt.plot(hops, self.results['dte_max'], 'r--', alpha=0.6, linewidth=1.5, label='Max Abs DTE')

        plt.xlabel('Hop Number', fontsize=12)
        plt.ylabel('Dynamic Time Error (ns)', fontsize=12)
        plt.title('DTE Growth Across Hops', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)

        plt.tight_layout()
        plt.savefig('output_image/dte_growth.png', dpi=300)
        plt.close()

    def plot_early_hops(self):
        """Plot DTE for hops 1-7"""
        plt.figure(figsize=(12, 7))

        # Get data for hops 1-7
        hops = np.arange(1, 8)
        dte_values = self.results['dte_7sigma'][:7]
        max_values = self.results['dte_max'][:7]

        # Create plot
        plt.plot(hops, dte_values, 'bo-', linewidth=2, label='7σ DTE')
        plt.plot(hops, max_values, 'ro--', alpha=0.6, linewidth=1.5, label='Max Abs DTE')

        for i, (dte, max_val) in enumerate(zip(dte_values, max_values)):
            plt.annotate(f'{dte:.1f}', (hops[i], dte), textcoords="offset points",
                         xytext=(0, 10), ha='center', fontsize=9)
            plt.annotate(f'{max_val:.1f}', (hops[i], max_val), textcoords="offset points",
                         xytext=(0, 10), ha='center', fontsize=9)

        plt.xlabel('Hop Number', fontsize=12)
        plt.ylabel('Dynamic Time Error (ns)', fontsize=12)
        plt.title('DTE for First 7 Hops', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)

        plt.tight_layout()
        plt.savefig('output_image/dte_early_hops.png', dpi=300)
        plt.close()

    def plot_selected_hops_line(self):
        """Plot line graph for hops 10, 25, 50, 75, 100"""
        plt.figure(figsize=(12, 7))

        selected_hops = [10, 25, 50, 75, 100]

        for hop in selected_hops:
            if hop <= self.params.num_hops:
                dte_values = self.results['dte_per_hop'][:, hop - 1]
                plt.plot(np.sort(dte_values), label=f'Hop {hop}')

        plt.axhline(y=1000, color='forestgreen', linestyle=':', linewidth=2, label='±1μs target')
        plt.axhline(y=-1000, color='forestgreen', linestyle=':', linewidth=2)

        plt.xlabel('Sorted Run Index', fontsize=12)
        plt.ylabel('Dynamic Time Error (ns)', fontsize=12)
        plt.title('DTE Values for Selected Hops', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)

        plt.tight_layout()
        plt.savefig('output_image/dte_selected_hops_line.png', dpi=300)
        plt.close()

    def plot_selected_hops_cdf(self):
        """Plot CDF for hops 10, 25, 50, 75, 100"""
        plt.figure(figsize=(12, 7))

        selected_hops = [10, 25, 50, 75, 100]

        for hop in selected_hops:
            if hop <= self.params.num_hops:
                dte_values = self.results['dte_per_hop'][:, hop - 1]
                sorted_data = np.sort(dte_values)
                yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                plt.plot(sorted_data, yvals, label=f'Hop {hop}')

        plt.axvline(x=1000, color='forestgreen', linestyle=':', linewidth=2, label='±1μs target')
        plt.axvline(x=-1000, color='forestgreen', linestyle=':', linewidth=2)

        plt.xlabel('Dynamic Time Error (ns)', fontsize=12)
        plt.ylabel('Cumulative Probability', fontsize=12)
        plt.title('CDF of DTE for Selected Hops', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)

        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

        plt.tight_layout()
        plt.savefig('output_image/dte_selected_hops_cdf.png', dpi=300)
        plt.close()


def main():
    # Create parameters with non-optimized settings to match Fig 5
    params = SimulationParameters(
        num_hops=100,
        num_runs=10000,  # Can adjust for faster runs during testing

        # Clock characteristics
        gm_clock_drift_max=1.5,
        gm_clock_drift_min=-1.5,
        gm_clock_drift_fraction=0.8,
        clock_drift_max=1.5,
        clock_drift_min=-1.5,
        clock_drift_fraction=0.8,

        # Temperature model parameters
        temp_max=85.0,
        temp_min=-40.0,
        temp_ramp_rate=1.0,
        temp_hold_period=30.0,
        gm_scaling_factor=1.0,
        non_gm_scaling_factor=1.0,
        use_temperature_model=True,  # Use temperature-based model

        # Timestamp errors
        tsge_tx=4.0,
        tsge_rx=4.0,
        dtse_tx=4.0,
        dtse_rx=4.0,

        # Message intervals
        pdelay_interval=125.0,
        sync_interval=125.0,
        pdelay_turnaround=10.0,
        residence_time=10.0,

        # Correction factors - all disabled to match Fig 5
        mean_link_delay_correction=0.0,
        nrr_drift_correction=0.0,
        rr_drift_correction=0.0,
        pdelayresp_sync_correction=0.0,
        mnrr_smoothing_n=1,
        mnrr_smoothing_m=1
    )

    # Create and run simulation
    sim = TimeSyncSimulation(params)
    print("Running simulation with non-optimized parameters to match Fig 5...")
    sim.run_simulation()

    # Output results
    max_dte = max(sim.results['dte_max'])
    final_7sigma = sim.results['dte_7sigma'][-1]

    print(f"Simulation complete!")
    print(f"Maximum DTE: {max_dte:.1f} ns")
    print(f"7-sigma DTE at hop {params.num_hops}: {final_7sigma:.1f} ns")
    print(f"Target (<1000 ns): {'PASSED' if final_7sigma < 1000 else 'FAILED'}")

    # Plot results
    print("Generating plots...")
    sim.plot_results()
    print("Plots saved to output_image folder.")


if __name__ == "__main__":
    main()