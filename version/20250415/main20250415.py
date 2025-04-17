"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/4/15 8:42
*  @Project :   pj_gptp_simulation
*  @Description :   IEEE 802.1AS Multi-Domain Simulation
*  @FileName:   main20250415_multi_domain.py
**************************************
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math

# Create the output directories
os.makedirs("output_data", exist_ok=True)
os.makedirs("output_image", exist_ok=True)

# Parameters from the paper
PHY_JITTER_MAX = 8e-9  # 8 ns
CLOCK_GRANULARITY = 8e-9  # 8 ns
MAX_DRIFT_RATE = 10e-6  # 10 ppm
NR_ERROR = 0.1e-6  # 0.1 ppm
RESIDENCE_TIME_MAX = 1e-3  # 1 ms
PROPAGATION_DELAY = 25e-9  # 25 ns
SYNC_INTERVAL = 31.25e-3  # 31.25 ms
NUM_SAMPLES = 1000  # Samples per hop

# Parameters for multi-domain simulation
DOMAIN_SIZE = 10  # Number of hops per domain
DOMAIN_BOUNDARY_ERROR_MAX = 30e-9  # Maximum error at domain boundaries (30 ns)


def calculate_time_error(hops, sync_interval=SYNC_INTERVAL):
    """
    Calculate time synchronization error based on the paper's equations for single domain.
    """
    # Basic error from clock drift (equation 11)
    gm_drift = np.random.uniform(-MAX_DRIFT_RATE, MAX_DRIFT_RATE)
    node_drift = np.random.uniform(-MAX_DRIFT_RATE, MAX_DRIFT_RATE)
    basic_error = (node_drift - gm_drift) * sync_interval

    # Errors due to PHY jitter and clock granularity
    if hops > 1:
        # Timestamp error
        timestamp_error = np.random.uniform(0, PHY_JITTER_MAX) + np.random.uniform(0, CLOCK_GRANULARITY)

        # Error propagation factor based on equations (16)-(23)
        error_factor = np.random.uniform(0.5, 1.0)

        # Calculate accumulated error (grows with hop count)
        accumulated_error = timestamp_error * error_factor * hops

        # From Figure 10 - error grows faster after ~30 hops
        if hops > 30:
            accumulated_error *= 1 + (hops - 30) / 100
    else:
        accumulated_error = 0

    # Total error
    total_error = basic_error + accumulated_error

    # Randomize sign (error can be positive or negative)
    if np.random.random() < 0.5:
        total_error = -total_error

    return total_error


def calculate_multi_domain_time_error(total_hops, domain_size=DOMAIN_SIZE):
    """
    Calculate time synchronization error with multi-domain approach.

    In this approach:
    - Each domain has its own GM
    - Error only accumulates within each domain (max domain_size hops)
    - Additional errors occur at domain boundaries
    """
    # Calculate which domain and which hop within that domain
    current_domain = (total_hops - 1) // domain_size + 1
    hop_within_domain = ((total_hops - 1) % domain_size) + 1

    # Calculate domain-internal error (only from current domain's GM)
    domain_error = calculate_time_error(hop_within_domain)

    # Add boundary errors from all previous domain crossings
    boundary_errors = 0
    if current_domain > 1:
        # Accumulate errors from each boundary crossing
        for i in range(current_domain - 1):
            boundary_error = np.random.uniform(0, DOMAIN_BOUNDARY_ERROR_MAX)
            # In real systems, boundary errors can be positive or negative
            if np.random.random() < 0.5:
                boundary_error = -boundary_error
            boundary_errors += boundary_error

    # Total error is domain-internal error plus all boundary crossing errors
    total_error = domain_error + boundary_errors

    return total_error


# Generate data for single domain approach (original method)
print("Generating data for single domain approach...")
single_domain_data = {}
for h in range(1, 101):
    print(f"Simulating single domain hop {h}...")
    hop_errors = [calculate_time_error(h) for _ in range(NUM_SAMPLES)]
    single_domain_data[h] = hop_errors

# Save to CSV
single_df = pd.DataFrame({hop: single_domain_data[hop] for hop in range(1, 101)})
single_df.to_csv("output_data/single_domain_data_v2.csv", index=False)

# Generate data for multi-domain approach
print("Generating data for multi-domain approach...")
multi_domain_data = {}
for h in range(1, 101):
    print(f"Simulating multi-domain hop {h}...")
    hop_errors = [calculate_multi_domain_time_error(h) for _ in range(NUM_SAMPLES)]
    multi_domain_data[h] = hop_errors

# Save multi-domain data to CSV
multi_df = pd.DataFrame({hop: multi_domain_data[hop] for hop in range(1, 101)})
multi_df.to_csv("output_data/multi_domain_data_v2.csv", index=False)

# Plot 1: Comparison of Time Error CDF for hop 100
plt.figure(figsize=(12, 8))

# Single domain data for hop 100
errors_single = np.array(single_domain_data[100])
errors_single_us = errors_single * 1e6  # Convert to microseconds
sorted_errors_single = np.sort(errors_single_us)
cumulative_prob_single = np.linspace(0, 1, len(sorted_errors_single))

# Multi domain data for hop 100
errors_multi = np.array(multi_domain_data[100])
errors_multi_us = errors_multi * 1e6  # Convert to microseconds
sorted_errors_multi = np.sort(errors_multi_us)
cumulative_prob_multi = np.linspace(0, 1, len(sorted_errors_multi))

# Calculate statistics for annotations
single_median = np.median(np.abs(errors_single_us))
multi_median = np.median(np.abs(errors_multi_us))
single_max = np.max(np.abs(errors_single_us))
multi_max = np.max(np.abs(errors_multi_us))

# Plot both distributions
plt.plot(sorted_errors_single, cumulative_prob_single,
         label=f'Single Domain (Median: {single_median:.2f}μs, Max: {single_max:.2f}μs)',
         color='#E41A1C',  # red
         linewidth=2)

plt.plot(sorted_errors_multi, cumulative_prob_multi,
         label=f'Multi Domain (Median: {multi_median:.2f}μs, Max: {multi_max:.2f}μs)',
         color='#377EB8',  # blue
         linewidth=2)

# Configure plot appearance
plt.grid(True, alpha=0.3)
plt.xlabel('Time Error (μs)', fontname='Times New Roman', fontsize=20, fontweight='bold')
plt.ylabel('CDF', fontname='Times New Roman', fontsize=20, fontweight='bold')
plt.legend(fontsize=16, frameon=True, shadow=False, edgecolor='black', prop={'family': 'Times New Roman'})
plt.xticks(fontname='Times New Roman', fontsize=20)
plt.yticks(fontname='Times New Roman', fontsize=20)
plt.title('Time Error Comparison: Single Domain vs. Multi-Domain (Hop 100)', fontname='Times New Roman', fontsize=22, fontweight='bold')
plt.tight_layout()
plt.savefig("output_image/domain_comparison_cdf_v2.png", dpi=600)
plt.close()

# Plot 2: Time Error vs Hop Number for both approaches
plt.figure(figsize=(14, 10))

# Prepare data for boxplot - single domain
single_domain_abs_errors = {}
for h in range(1, 101, 5):  # Sample every 5 hops for clarity
    errors = np.array(single_domain_data[h])
    errors_us = errors * 1e6  # Convert to microseconds
    single_domain_abs_errors[h] = np.abs(errors_us)  # Use absolute errors

# Prepare data for boxplot - multi domain
multi_domain_abs_errors = {}
for h in range(1, 101, 5):  # Sample every 5 hops for clarity
    errors = np.array(multi_domain_data[h])
    errors_us = errors * 1e6  # Convert to microseconds
    multi_domain_abs_errors[h] = np.abs(errors_us)  # Use absolute errors

# Plot median values with error bars (more readable than raw boxplots)
hops = list(single_domain_abs_errors.keys())

single_medians = [np.median(single_domain_abs_errors[h]) for h in hops]
single_q1 = [np.percentile(single_domain_abs_errors[h], 25) for h in hops]
single_q3 = [np.percentile(single_domain_abs_errors[h], 75) for h in hops]

multi_medians = [np.median(multi_domain_abs_errors[h]) for h in hops]
multi_q1 = [np.percentile(multi_domain_abs_errors[h], 25) for h in hops]
multi_q3 = [np.percentile(multi_domain_abs_errors[h], 75) for h in hops]

plt.plot(hops, single_medians, 'o-', color='#E41A1C', label='Single Domain', linewidth=2, markersize=8)
plt.fill_between(hops, single_q1, single_q3, color='#E41A1C', alpha=0.2)

plt.plot(hops, multi_medians, 's-', color='#377EB8', label='Multi Domain', linewidth=2, markersize=8)
plt.fill_between(hops, multi_q1, multi_q3, color='#377EB8', alpha=0.2)

# Add vertical lines at domain boundaries
for i in range(1, 10):
    boundary = i * DOMAIN_SIZE
    plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.7)

# Configure plot appearance
plt.grid(True, alpha=0.3)
plt.xlabel('Hop Number', fontname='Times New Roman', fontsize=20, fontweight='bold')
plt.ylabel('Absolute Time Error (μs)', fontname='Times New Roman', fontsize=20, fontweight='bold')
plt.legend(fontsize=18, frameon=True, shadow=False, edgecolor='black', prop={'family': 'Times New Roman'})
plt.xticks(fontname='Times New Roman', fontsize=20)
plt.yticks(fontname='Times New Roman', fontsize=20)
plt.title('Time Error vs. Hop Number', fontname='Times New Roman', fontsize=22, fontweight='bold')
plt.tight_layout()
plt.savefig("output_image/hop_vs_error_v2.png", dpi=600)
plt.close()

# Plot 3: Target hops CDF comparison
target_hops = [10, 50, 100]
colors = {
    '10': '#E41A1C',   # red
    '50': '#4DAF4A',   # green
    '100': '#984EA3',  # purple
}

plt.figure(figsize=(14, 10))

# Plot for each target hop (both single and multi domain)
for hop in target_hops:
    # Single domain
    errors_single = np.array(single_domain_data[hop])
    errors_single_us = errors_single * 1e6
    sorted_errors_single = np.sort(errors_single_us)
    cumulative_prob_single = np.linspace(0, 1, len(sorted_errors_single))

    plt.plot(sorted_errors_single, cumulative_prob_single,
             label=f'Single Domain - Hop {hop}',
             color=colors[str(hop)],
             linestyle='-',
             linewidth=2)

    # Multi domain
    errors_multi = np.array(multi_domain_data[hop])
    errors_multi_us = errors_multi * 1e6
    sorted_errors_multi = np.sort(errors_multi_us)
    cumulative_prob_multi = np.linspace(0, 1, len(sorted_errors_multi))

    plt.plot(sorted_errors_multi, cumulative_prob_multi,
             label=f'Multi Domain - Hop {hop}',
             color=colors[str(hop)],
             linestyle='--',
             linewidth=2)

# Configure plot appearance
plt.grid(True, alpha=0.3)
plt.xlabel('Time Error (μs)', fontname='Times New Roman', fontsize=20, fontweight='bold')
plt.ylabel('CDF', fontname='Times New Roman', fontsize=20, fontweight='bold')
plt.legend(fontsize=16, frameon=True, shadow=False, edgecolor='black', prop={'family': 'Times New Roman'})
plt.xticks(fontname='Times New Roman', fontsize=20)
plt.yticks(fontname='Times New Roman', fontsize=20)
plt.title('Time Error CDF: Single vs. Multi-Domain Comparison', fontname='Times New Roman', fontsize=22, fontweight='bold')
plt.tight_layout()
plt.savefig("output_image/target_hops_comparison_cdf_v2.png", dpi=600)
plt.close()

print("Simulation complete. Results saved to output_data/ and output_image/ directories.")