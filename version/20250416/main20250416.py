"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/4/16 19:22
*  @Project :   pj_gptp_simulation
*  @Description :   将不同时间间隔的 折线图（Line Chart），并结合了 区间填充（Shaded Area Plot）
*  @FileName:   main20250416.py
**************************************
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import time

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
TOTAL_HOPS = 100  # Total number of hops to simulate

# Parameters for multi-domain simulation
DOMAIN_BOUNDARY_ERROR_MAX = 30e-9  # Maximum error at domain boundaries (30 ns)

# Define different domain size configurations to test
DOMAIN_SIZE_CONFIGS = {
    'small': 5,  # 5 hops per domain (20 domains total)
    'medium': 10,  # 10 hops per domain (10 domains total)
    'large': 20,  # 20 hops per domain (5 domains total)
    'xlarge': 50,  # 50 hops per domain (2 domains total)
}


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


def calculate_multi_domain_time_error(total_hops, domain_size):
    """
    Calculate time synchronization error with multi-domain approach.

    In this approach:
    - Each domain has its own GM
    - Error only accumulates within each domain (max domain_size hops)
    - Additional errors occur at domain boundaries

    Args:
        total_hops: The total number of hops from the network entry
        domain_size: Number of hops per domain
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
start_time = time.time()
single_domain_data = {}
for h in range(1, TOTAL_HOPS + 1):
    print(f"Simulating single domain hop {h}...")
    hop_errors = [calculate_time_error(h) for _ in range(NUM_SAMPLES)]
    single_domain_data[h] = hop_errors

# Save to CSV
single_df = pd.DataFrame({hop: single_domain_data[hop] for hop in range(1, TOTAL_HOPS + 1)})
single_df.to_csv("output_data/single_domain_data_v3.csv", index=False)
print(f"Single domain simulation completed in {time.time() - start_time:.2f} seconds")

# Generate data for multi-domain approach with different domain sizes
multi_domain_results = {}

for config_name, domain_size in DOMAIN_SIZE_CONFIGS.items():
    print(f"Generating data for multi-domain approach with {domain_size} hops per domain ({config_name})...")
    start_time = time.time()

    multi_domain_data = {}
    for h in range(1, TOTAL_HOPS + 1):
        print(f"Simulating multi-domain hop {h} with domain size {domain_size}...")
        hop_errors = [calculate_multi_domain_time_error(h, domain_size) for _ in range(NUM_SAMPLES)]
        multi_domain_data[h] = hop_errors

    # Store the results for this configuration
    multi_domain_results[config_name] = multi_domain_data

    # Save multi-domain data to CSV
    multi_df = pd.DataFrame({hop: multi_domain_data[hop] for hop in range(1, TOTAL_HOPS + 1)})
    multi_df.to_csv(f"output_data/multi_domain_data_{config_name}_v3.csv", index=False)
    print(f"{config_name} domain simulation completed in {time.time() - start_time:.2f} seconds")

# PLOTTING SECTION

# Plot 1: Comparison of Time Error CDF for hop 100 across all configurations
plt.figure(figsize=(14, 10))

# Single domain data for hop 100
errors_single = np.array(single_domain_data[TOTAL_HOPS])
errors_single_us = errors_single * 1e6  # Convert to microseconds
sorted_errors_single = np.sort(errors_single_us)
cumulative_prob_single = np.linspace(0, 1, len(sorted_errors_single))

plt.plot(sorted_errors_single, cumulative_prob_single,
         label=f'Single Domain',
         color='#E41A1C',  # red
         linewidth=2)

# Color map for multi-domain configurations
config_colors = {
    'small': '#377EB8',  # blue
    'medium': '#4DAF4A',  # green
    'large': '#984EA3',  # purple
    'xlarge': '#FF7F00',  # orange
}

# Plot each multi-domain configuration
for config_name, domain_size in DOMAIN_SIZE_CONFIGS.items():
    errors_multi = np.array(multi_domain_results[config_name][TOTAL_HOPS])
    errors_multi_us = errors_multi * 1e6  # Convert to microseconds
    sorted_errors_multi = np.sort(errors_multi_us)
    cumulative_prob_multi = np.linspace(0, 1, len(sorted_errors_multi))

    # Calculate statistics for annotations
    single_median = np.median(np.abs(errors_single_us))
    multi_median = np.median(np.abs(errors_multi_us))

    plt.plot(sorted_errors_multi, cumulative_prob_multi,
             label=f'Domain Size {domain_size} ({config_name}) - Median: {multi_median:.2f}μs',
             color=config_colors[config_name],
             linewidth=2)

# Configure plot appearance
plt.grid(True, alpha=0.3)
plt.xlabel('Time Error (μs)', fontname='Times New Roman', fontsize=20, fontweight='bold')
plt.ylabel('CDF', fontname='Times New Roman', fontsize=20, fontweight='bold')
plt.legend(fontsize=16, frameon=True, shadow=False, edgecolor='black', prop={'family': 'Times New Roman'})
plt.xticks(fontname='Times New Roman', fontsize=20)
plt.yticks(fontname='Times New Roman', fontsize=20)
plt.title(f'Time Error Comparison: Different Domain Sizes (Hop {TOTAL_HOPS})',
          fontname='Times New Roman', fontsize=22, fontweight='bold')
plt.tight_layout()
plt.savefig("output_image/domain_size_comparison_cdf_v3.png", dpi=600)
plt.close()

# Plot 2: Time Error vs Hop Number for all approaches (both absolute and real values)
for error_type in ['absolute', 'real']:
    plt.figure(figsize=(16, 12))

    # Prepare data for single domain
    single_domain_errors = {}
    for h in range(1, TOTAL_HOPS + 1, 5):  # Sample every 5 hops for clarity
        errors = np.array(single_domain_data[h])
        errors_us = errors * 1e6  # Convert to microseconds

        if error_type == 'absolute':
            single_domain_errors[h] = np.abs(errors_us)
        else:
            single_domain_errors[h] = errors_us

    # Plot for single domain
    hops = list(single_domain_errors.keys())
    single_medians = [np.median(single_domain_errors[h]) for h in hops]
    single_q1 = [np.percentile(single_domain_errors[h], 25) for h in hops]
    single_q3 = [np.percentile(single_domain_errors[h], 75) for h in hops]

    plt.plot(hops, single_medians, 'o-', color='#E41A1C', label='Single Domain',
             linewidth=2, markersize=8, zorder=10)
    plt.fill_between(hops, single_q1, single_q3, color='#E41A1C', alpha=0.2)

    # Plot for each multi-domain configuration
    for config_name, domain_size in DOMAIN_SIZE_CONFIGS.items():
        multi_domain_errors = {}
        for h in range(1, TOTAL_HOPS + 1, 5):
            errors = np.array(multi_domain_results[config_name][h])
            errors_us = errors * 1e6

            if error_type == 'absolute':
                multi_domain_errors[h] = np.abs(errors_us)
            else:
                multi_domain_errors[h] = errors_us

        multi_medians = [np.median(multi_domain_errors[h]) for h in hops]
        multi_q1 = [np.percentile(multi_domain_errors[h], 25) for h in hops]
        multi_q3 = [np.percentile(multi_domain_errors[h], 75) for h in hops]

        plt.plot(hops, multi_medians, 's-', color=config_colors[config_name],
                 label=f'Domain Size {domain_size} ({config_name})',
                 linewidth=2, markersize=8, zorder=5)
        plt.fill_between(hops, multi_q1, multi_q3, color=config_colors[config_name], alpha=0.2)

        # Add vertical lines at domain boundaries for this configuration
        if config_name == 'medium':  # Only show domain boundaries for medium (default) to avoid clutter
            num_domains = TOTAL_HOPS // domain_size
            for i in range(1, num_domains):
                boundary = i * domain_size
                plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)

    # Configure plot appearance
    plt.grid(True, alpha=0.3)
    plt.xlabel('Hop Number', fontname='Times New Roman', fontsize=20, fontweight='bold')

    if error_type == 'absolute':
        plt.ylabel('Absolute Time Error (μs)', fontname='Times New Roman', fontsize=20, fontweight='bold')
        plt.title('Absolute Time Error vs. Hop Number', fontname='Times New Roman', fontsize=22, fontweight='bold')
        plt.savefig("output_image/hop_vs_absolute_error_v3.png", dpi=600)
    else:
        plt.ylabel('Time Error (μs)', fontname='Times New Roman', fontsize=20, fontweight='bold')
        plt.title('Real Time Error vs. Hop Number', fontname='Times New Roman', fontsize=22, fontweight='bold')
        plt.savefig("output_image/hop_vs_real_error_v3.png", dpi=600)

    plt.legend(fontsize=16, frameon=True, shadow=False, edgecolor='black',
               prop={'family': 'Times New Roman'}, loc='best')
    plt.xticks(fontname='Times New Roman', fontsize=20)
    plt.yticks(fontname='Times New Roman', fontsize=20)
    plt.tight_layout()
    plt.close()

# Plot 3: Domain Size Impact Analysis (Box plot at hop 100)
plt.figure(figsize=(14, 10))

# Prepare data for boxplot
box_data = []
labels = ['Single Domain']

# First, add single domain data
single_errors = np.array(single_domain_data[TOTAL_HOPS]) * 1e6  # Convert to microseconds
box_data.append(np.abs(single_errors))  # Use absolute errors

# Then add data for each domain size configuration
for config_name, domain_size in sorted(DOMAIN_SIZE_CONFIGS.items(), key=lambda x: x[1]):
    multi_errors = np.array(multi_domain_results[config_name][TOTAL_HOPS]) * 1e6
    box_data.append(np.abs(multi_errors))
    labels.append(f'Size {domain_size}\n({config_name})')

# Create boxplot
boxplot = plt.boxplot(box_data, patch_artist=True, showfliers=False,
                      medianprops={'color': 'black', 'linewidth': 2})

# Color boxes
colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00']
for patch, color in zip(boxplot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Configure plot appearance
plt.grid(True, alpha=0.3, axis='y')
plt.xlabel('Domain Configuration', fontname='Times New Roman', fontsize=20, fontweight='bold')
plt.ylabel('Absolute Time Error (μs)', fontname='Times New Roman', fontsize=20, fontweight='bold')
plt.xticks(range(1, len(labels) + 1), labels, fontname='Times New Roman', fontsize=18)
plt.yticks(fontname='Times New Roman', fontsize=20)
plt.title(f'Impact of Domain Size on Time Error (Hop {TOTAL_HOPS})',
          fontname='Times New Roman', fontsize=22, fontweight='bold')

# Add numeric annotations for median values
for i, data in enumerate(box_data):
    median = np.median(data)
    plt.text(i + 1, median + 0.5, f'{median:.2f}μs',
             horizontalalignment='center', fontsize=14, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))

plt.tight_layout()
plt.savefig("output_image/domain_size_impact_boxplot_v3.png", dpi=600)
plt.close()

# Plot 4: Optimal Domain Size Analysis
# Calculate median error at hop 100 for each configuration
all_configs = ['Single'] + list(DOMAIN_SIZE_CONFIGS.keys())
median_errors = []
num_domains = []

# Single domain (1 domain total)
single_errors = np.abs(np.array(single_domain_data[TOTAL_HOPS]) * 1e6)
median_errors.append(np.median(single_errors))
num_domains.append(1)

# Multi-domain configurations
for config_name, domain_size in sorted(DOMAIN_SIZE_CONFIGS.items(), key=lambda x: x[1]):
    multi_errors = np.abs(np.array(multi_domain_results[config_name][TOTAL_HOPS]) * 1e6)
    median_errors.append(np.median(multi_errors))
    num_domains.append(TOTAL_HOPS // domain_size)  # Number of domains

# Plot domain count vs. error
plt.figure(figsize=(14, 10))
plt.plot(num_domains, median_errors, 'o-', color='#377EB8', linewidth=3, markersize=12)

# Add labels for each point
for i, (x, y, config) in enumerate(zip(num_domains, median_errors, all_configs)):
    plt.annotate(f'{config}\n({TOTAL_HOPS / num_domains[i]:.0f} hops/domain)',
                 (x, y), textcoords="offset points",
                 xytext=(0, 10), ha='center',
                 fontsize=14, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# Find and mark the optimal point
optimal_idx = np.argmin(median_errors)
plt.plot(num_domains[optimal_idx], median_errors[optimal_idx], 'o',
         color='red', markersize=15, label='Optimal')

# Configure plot appearance
plt.grid(True, alpha=0.3)
plt.xlabel('Number of Domains', fontname='Times New Roman', fontsize=20, fontweight='bold')
plt.ylabel('Median Absolute Time Error (μs)', fontname='Times New Roman', fontsize=20, fontweight='bold')
plt.xticks(fontname='Times New Roman', fontsize=20)
plt.yticks(fontname='Times New Roman', fontsize=20)
plt.title(f'Optimal Domain Count Analysis (Total {TOTAL_HOPS} hops)',
          fontname='Times New Roman', fontsize=22, fontweight='bold')
plt.tight_layout()
plt.savefig("output_image/optimal_domain_count_v3.png", dpi=600)
plt.close()

print("Simulation complete. Results saved to output_data/ and output_image/ directories.")