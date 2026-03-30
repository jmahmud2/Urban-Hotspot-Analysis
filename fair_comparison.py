"""
FAIR COMPARISON: Naive DBSCAN vs Optimized DBSCAN
Both implementations use EPSG:2263 projection + Euclidean distance in meters
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.optimize import curve_fit
from load_uber_data import load_uber_2014_data, prepare_gps_data
from src.naive_dbscan import NaiveDBSCAN
from src.optimized_dbscan import OptimizedDBSCAN

# Set style for professional visualizations
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

# Neutral color palette
COLORS = {
    'naive': '#E63946',
    'optimized': '#2E8B57',
    'speedup': '#1E88E5',
    'grid': '#CCCCCC'
}

def compute_cluster_metrics(labels, X_proj):
    """Compute cluster quality metrics with memory-efficient sampling"""
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    noise_pct = n_noise / len(labels) * 100
    
    # Cluster sizes
    cluster_sizes = []
    cluster_centroids = []
    for label in set(labels):
        if label != -1:
            size = np.sum(labels == label)
            cluster_sizes.append(size)
            centroid = np.mean(X_proj[labels == label], axis=0)
            cluster_centroids.append(centroid)
    
    # Average cluster size
    avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
    cluster_size_std = np.std(cluster_sizes) if cluster_sizes else 0
    largest_cluster = max(cluster_sizes) if cluster_sizes else 0
    smallest_cluster = min(cluster_sizes) if cluster_sizes else 0
    
    # Intra-cluster distances - SAMPLING to avoid memory explosion
    intra_distances = []
    for label in set(labels):
        if label != -1:
            cluster_points = X_proj[labels == label]
            if len(cluster_points) > 1:
                # Sample up to 500 points per cluster for distance calculation
                if len(cluster_points) > 500:
                    sample_idx = np.random.choice(len(cluster_points), 500, replace=False)
                    cluster_points = cluster_points[sample_idx]
                if len(cluster_points) > 1:
                    distances = pdist(cluster_points)
                    intra_distances.extend(distances)
    
    avg_intra_distance = np.mean(intra_distances) if intra_distances else 0
    
    # Inter-cluster distances (centroids only, no memory issue)
    inter_distances = []
    for i, c1 in enumerate(cluster_centroids):
        for j, c2 in enumerate(cluster_centroids):
            if i < j:
                dist = np.linalg.norm(c1 - c2)
                inter_distances.append(dist)
    
    avg_inter_distance = np.mean(inter_distances) if inter_distances else 0
    
    # Separation ratio (higher is better)
    separation_ratio = avg_inter_distance / (avg_intra_distance + 1e-10) if avg_intra_distance > 0 else 0
    
    # Compactness ratio (lower is better)
    compactness_ratio = avg_intra_distance / (avg_inter_distance + 1e-10) if avg_inter_distance > 0 else 0
    
    return {
        'n_clusters': n_clusters,
        'noise_pct': noise_pct,
        'avg_cluster_size': avg_cluster_size,
        'cluster_size_std': cluster_size_std,
        'largest_cluster': largest_cluster,
        'smallest_cluster': smallest_cluster,
        'avg_intra_distance': avg_intra_distance,
        'avg_inter_distance': avg_inter_distance,
        'separation_ratio': separation_ratio,
        'compactness_ratio': compactness_ratio
    }

def generate_cluster_map(X, labels, method_name, n_points, output_path):
    """Generate a neutral cluster map without subjective landmarks"""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    unique_labels = np.unique(labels)
    cluster_labels = unique_labels[unique_labels != -1]
    
    # Use a high-contrast colormap
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(cluster_labels))))
    
    # Plot clusters
    for i, label in enumerate(cluster_labels):
        cluster_points = X[labels == label]
        ax.scatter(cluster_points[:, 1], cluster_points[:, 0], 
                  c=[colors[i % len(colors)]], s=10, alpha=0.7, 
                  label=f'Cluster {label}', edgecolors='white', linewidth=0.3)
    
    # Plot noise
    noise_points = X[labels == -1]
    if len(noise_points) > 0:
        ax.scatter(noise_points[:, 1], noise_points[:, 0], 
                  c='lightgray', s=3, alpha=0.3, label='Noise')
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'{method_name} - Clustering Results ({n_points:,} points)\n{len(cluster_labels)} Clusters Found', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(-74.05, -73.85)
    ax.set_ylim(40.6, 40.9)
    ax.grid(True, alpha=0.2)
    
    # Add legend (limited to top 10 to avoid clutter)
    handles = []
    for i, label in enumerate(cluster_labels[:10]):
        handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=colors[i % len(colors)], 
                                  markersize=8, label=f'Cluster {label}'))
    if len(noise_points) > 0:
        handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor='lightgray', markersize=8, label='Noise'))
    ax.legend(handles=handles, loc='upper right', fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def print_size_independent_observations(results_df):
    """Print observations that don't depend on specific sample sizes"""
    
    print("\n" + "="*80)
    print("SIZE-INDEPENDENT OBSERVATIONS")
    print("="*80)
    
    # 1. Crossover point
    cross_df = results_df[results_df['speedup'] >= 1]
    if len(cross_df) > 0:
        cross_point = cross_df['n_points'].iloc[0]
        print(f"\n1. CROSSOVER BEHAVIOR:")
        print(f"   Optimized becomes faster than naive at approximately {cross_point:,} points")
        print(f"   Before this, grid index overhead dominates; after, spatial indexing pays off")
    
    # 2. Scaling exponents
    x = np.log(results_df['n_points'])
    naive_log = np.log(results_df['naive_time'])
    opt_log = np.log(results_df['optimized_time'])
    
    naive_exp = np.polyfit(x, naive_log, 1)[0]
    opt_exp = np.polyfit(x, opt_log, 1)[0]
    
    print(f"\n2. ASYMPTOTIC SCALING:")
    print(f"   Naive exponent: {naive_exp:.2f} (theoretical O(n²) = 2.0)")
    print(f"   Optimized exponent: {opt_exp:.2f} (theoretical O(n log n) ≈ 1.1-1.3)")
    
    # 3. Efficiency ratio
    efficiency_ratio = (results_df['optimized_time'] / results_df['naive_time']).iloc[-1]
    print(f"\n3. EFFICIENCY RATIO:")
    print(f"   At largest scale, optimized takes {efficiency_ratio:.2%} of naive time")
    
    # 4. Cluster stability
    cluster_cv = results_df['optimized_clusters'].std() / results_df['optimized_clusters'].mean()
    print(f"\n4. CLUSTER STABILITY:")
    print(f"   Coefficient of variation: {cluster_cv:.3f}")
    
    # 5. Noise reduction trend
    noise_start = results_df['optimized_noise_pct'].iloc[0]
    noise_end = results_df['optimized_noise_pct'].iloc[-1]
    noise_reduction = (noise_start - noise_end) / noise_start
    print(f"\n5. NOISE REDUCTION:")
    print(f"   Noise decreased from {noise_start:.1f}% to {noise_end:.1f}%")
    print(f"   → {noise_reduction:.1%} reduction as dataset size increases")
    
    # 6. Consistency metric
    cluster_diff = (results_df['naive_clusters'] - results_df['optimized_clusters']).abs()
    max_diff = cluster_diff.max()
    print(f"\n6. IMPLEMENTATION CONSISTENCY:")
    print(f"   Maximum cluster count difference: {max_diff:.0f}")
    
    # 7. Processing rate trend
    rate_start = (results_df['n_points'].iloc[0] / results_df['optimized_time'].iloc[0])
    rate_end = (results_df['n_points'].iloc[-1] / results_df['optimized_time'].iloc[-1])
    rate_change = (rate_end - rate_start) / rate_start
    print(f"\n7. PROCESSING RATE SCALING:")
    print(f"   Processing rate changed from {rate_start:.0f} to {rate_end:.0f} pts/sec")

def generate_execution_time_plot(results_df):
    """Execution time comparison"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = results_df['n_points']
    
    ax.plot(x, results_df['naive_time'], 'o-', 
            label='Naive O(n²)', linewidth=2.5, markersize=9,
            color=COLORS['naive'], markerfacecolor='white', markeredgewidth=2)
    
    ax.plot(x, results_df['optimized_time'], 's-', 
            label='Optimized O(n log n)', linewidth=2.5, markersize=9,
            color=COLORS['optimized'], markerfacecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Execution Time: Naive vs Optimized DBSCAN', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('results/execution_time_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_speedup_plot(results_df):
    """Speedup factor plot"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(results_df['n_points'], results_df['speedup'], 'd-', 
            color=COLORS['speedup'], linewidth=2.5, markersize=9,
            markerfacecolor='white', markeredgewidth=2)
    ax.axhline(y=1, color=COLORS['naive'], linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline (1x)')
    
    ax.fill_between(results_df['n_points'], 1, results_df['speedup'], 
                     where=(results_df['speedup'] > 1), color=COLORS['speedup'], alpha=0.25)
    
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_ylabel('Speedup Factor', fontsize=12)
    ax.set_title(f'Spatial Indexing Speedup\n{results_df["speedup"].max():.1f}x at {results_df.loc[results_df["speedup"].idxmax(), "n_points"]:.0f} points', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    for _, row in results_df.iterrows():
        ax.annotate(f'{row["speedup"]:.1f}x', 
                   (row['n_points'], row['speedup']),
                   textcoords="offset points", xytext=(5, 8), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/speedup_factor.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_cluster_comparison_plot(results_df):
    """Cluster count comparison"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    mask = results_df['naive_clusters'] > 0
    filtered = results_df[mask]
    
    ax.plot(filtered['n_points'], filtered['naive_clusters'], 'o-', 
            label='Naive DBSCAN', linewidth=2.5, markersize=9,
            color=COLORS['naive'], markerfacecolor='white', markeredgewidth=2)
    
    ax.plot(filtered['n_points'], filtered['optimized_clusters'], 's-', 
            label='Optimized DBSCAN', linewidth=2.5, markersize=9,
            color=COLORS['optimized'], markerfacecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_ylabel('Number of Clusters', fontsize=12)
    ax.set_title('Clusters Found: Naive vs Optimized DBSCAN', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    for _, row in filtered.iterrows():
        ax.annotate(f'{row["naive_clusters"]:.0f}', 
                   (row['n_points'], row['naive_clusters']),
                   textcoords="offset points", xytext=(5, 8), ha='center', fontsize=9)
        ax.annotate(f'{row["optimized_clusters"]:.0f}', 
                   (row['n_points'], row['optimized_clusters']),
                   textcoords="offset points", xytext=(5, -12), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/cluster_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_noise_comparison_plot(results_df):
    """Noise ratio comparison"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    mask = results_df['naive_clusters'] > 0
    filtered = results_df[mask]
    
    ax.plot(filtered['n_points'], filtered['naive_noise_pct'], 'o-', 
            label='Naive DBSCAN', linewidth=2.5, markersize=9,
            color=COLORS['naive'], markerfacecolor='white', markeredgewidth=2)
    
    ax.plot(filtered['n_points'], filtered['optimized_noise_pct'], 's-', 
            label='Optimized DBSCAN', linewidth=2.5, markersize=9,
            color=COLORS['optimized'], markerfacecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_ylabel('Noise Percentage (%)', fontsize=12)
    ax.set_title('Noise Ratio: Naive vs Optimized DBSCAN', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    for _, row in filtered.iterrows():
        ax.annotate(f'{row["naive_noise_pct"]:.0f}%', 
                   (row['n_points'], row['naive_noise_pct']),
                   textcoords="offset points", xytext=(5, 8), ha='center', fontsize=9)
        ax.annotate(f'{row["optimized_noise_pct"]:.0f}%', 
                   (row['n_points'], row['optimized_noise_pct']),
                   textcoords="offset points", xytext=(5, -12), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/noise_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_cluster_quality_plot(results_df):
    """Cluster quality metrics (separation ratio)"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    mask = results_df['naive_clusters'] > 0
    filtered = results_df[mask]
    
    ax.plot(filtered['n_points'], filtered['naive_separation_ratio'], 'o-', 
            label='Naive DBSCAN', linewidth=2.5, markersize=9,
            color=COLORS['naive'], markerfacecolor='white', markeredgewidth=2)
    
    ax.plot(filtered['n_points'], filtered['optimized_separation_ratio'], 's-', 
            label='Optimized DBSCAN', linewidth=2.5, markersize=9,
            color=COLORS['optimized'], markerfacecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_ylabel('Separation Ratio (Inter/Intra Distance)', fontsize=12)
    ax.set_title('Cluster Quality: Separation Ratio (higher = better)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('results/cluster_quality.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_cluster_size_plot(results_df):
    """Cluster size comparison"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    mask = results_df['naive_clusters'] > 0
    filtered = results_df[mask]
    
    ax.plot(filtered['n_points'], filtered['naive_avg_cluster_size'], 'o-', 
            label='Naive DBSCAN', linewidth=2.5, markersize=9,
            color=COLORS['naive'], markerfacecolor='white', markeredgewidth=2)
    
    ax.plot(filtered['n_points'], filtered['optimized_avg_cluster_size'], 's-', 
            label='Optimized DBSCAN', linewidth=2.5, markersize=9,
            color=COLORS['optimized'], markerfacecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_ylabel('Average Cluster Size (points)', fontsize=12)
    ax.set_title('Average Cluster Size: Naive vs Optimized DBSCAN', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('results/cluster_size_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_processing_rate_plot(results_df):
    """Processing rate (points per second)"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    naive_rate = results_df['n_points'] / results_df['naive_time']
    opt_rate = results_df['n_points'] / results_df['optimized_time']
    
    ax.plot(results_df['n_points'], naive_rate, 'o-', 
            label='Naive O(n²)', linewidth=2.5, markersize=9,
            color=COLORS['naive'], markerfacecolor='white', markeredgewidth=2)
    
    ax.plot(results_df['n_points'], opt_rate, 's-', 
            label='Optimized O(n log n)', linewidth=2.5, markersize=9,
            color=COLORS['optimized'], markerfacecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_ylabel('Points Processed per Second', fontsize=12)
    ax.set_title('Processing Rate: Naive vs Optimized DBSCAN', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('results/processing_rate.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_scalability_analysis(results_df):
    """Scalability analysis with theoretical O(n²) reference"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = results_df['n_points']
    
    ax.loglog(x, results_df['naive_time'], 'o-', 
              label='Naive (Actual)', linewidth=2.5, markersize=9,
              color=COLORS['naive'], markerfacecolor='white', markeredgewidth=2)
    
    ax.loglog(x, results_df['optimized_time'], 's-', 
              label='Optimized (Actual)', linewidth=2.5, markersize=9,
              color=COLORS['optimized'], markerfacecolor='white', markeredgewidth=2)
    
    # Theoretical O(n²) reference
    t_ref = results_df['naive_time'].iloc[0] * (x / x.iloc[0])**2
    ax.loglog(x, t_ref, '--', label='Theoretical O(n²)', alpha=0.6, linewidth=1.5, color=COLORS['grid'])
    
    ax.set_xlabel('Number of Points (log scale)', fontsize=12)
    ax.set_ylabel('Time (seconds) (log scale)', fontsize=12)
    ax.set_title('Scalability Analysis: O(n²) vs O(n log n)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('results/scalability_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_scaling_exponent_plot(results_df):
    """Plot actual scaling vs theoretical curves with O(n log n) reference"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = results_df['n_points']
    
    # Actual data
    ax.loglog(x, results_df['naive_time'], 'o-', 
              label='Naive (Actual)', linewidth=2.5, color=COLORS['naive'])
    ax.loglog(x, results_df['optimized_time'], 's-', 
              label='Optimized (Actual)', linewidth=2.5, color=COLORS['optimized'])
    
    # Theoretical O(n²)
    t_quadratic = results_df['naive_time'].iloc[0] * (x / x.iloc[0])**2
    ax.loglog(x, t_quadratic, '--', label='Theoretical O(n²)', alpha=0.7, color='gray')
    
    # Theoretical O(n log n)
    t_expected = results_df['optimized_time'].iloc[0] * (x / x.iloc[0]) * np.log2(x / x.iloc[0])
    ax.loglog(x, t_expected, ':', label='Theoretical O(n log n)', alpha=0.7, color='darkgray')
    
    ax.set_xlabel('Number of Points (log scale)', fontsize=12)
    ax.set_ylabel('Time (seconds) (log scale)', fontsize=12)
    ax.set_title('Asymptotic Scaling Analysis', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/scaling_exponent_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_speedup_trend_plot(results_df):
    """Speedup with logarithmic trendline"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = results_df['n_points']
    y = results_df['speedup']
    
    ax.plot(x, y, 'd-', label='Actual Speedup', linewidth=2.5, 
            color=COLORS['speedup'], markersize=9, markerfacecolor='white')
    
    # Fit logarithmic trend
    def log_func(x, a, b):
        return a * np.log(x) + b
    
    try:
        popt, _ = curve_fit(log_func, x, y)
        x_smooth = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
        y_fit = log_func(x_smooth, *popt)
        ax.plot(x_smooth, y_fit, '--', label=f'Logarithmic Trend: {popt[0]:.2f}·log(n) + {popt[1]:.2f}', 
                alpha=0.7, color='darkgray')
    except:
        pass
    
    ax.axhline(y=1, color=COLORS['naive'], linestyle=':', alpha=0.7, label='Baseline (1x)')
    ax.fill_between(x, 1, y, where=(y > 1), color=COLORS['speedup'], alpha=0.2)
    
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_ylabel('Speedup Factor', fontsize=12)
    ax.set_title('Speedup Trend Analysis', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('results/speedup_trend.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_performance_heatmap(results_df):
    """Heatmap showing relative performance"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate ratio (optimized/naive)
    ratio = results_df['optimized_time'] / results_df['naive_time']
    
    # Create heatmap data
    heatmap_data = ratio.values.reshape(1, -1)
    
    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=2)
    
    # Add labels
    ax.set_xticks(range(len(results_df['n_points'])))
    ax.set_xticklabels([f'{int(x):,}' for x in results_df['n_points']], rotation=45, ha='right')
    ax.set_yticks([0])
    ax.set_yticklabels(['Optimized / Naive'])
    
    # Add value labels
    for j, val in enumerate(ratio):
        color = 'white' if val < 0.7 or val > 1.3 else 'black'
        ax.text(j, 0, f'{val:.2f}x', ha='center', va='center', color=color, fontweight='bold')
    
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_title('Relative Performance (Lower = Better for Optimized)', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Time Ratio (Optimized / Naive)')
    plt.tight_layout()
    plt.savefig('results/performance_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_efficiency_convergence_plot(results_df):
    """Processing rate comparison"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = results_df['n_points']
    naive_rate = results_df['n_points'] / results_df['naive_time']
    opt_rate = results_df['n_points'] / results_df['optimized_time']
    
    ax.plot(x, naive_rate, 'o-', label='Naive O(n²)', linewidth=2.5,
            color=COLORS['naive'], markersize=9, markerfacecolor='white')
    ax.plot(x, opt_rate, 's-', label='Optimized O(n log n)', linewidth=2.5,
            color=COLORS['optimized'], markersize=9, markerfacecolor='white')
    
    # Add efficiency ratio annotation
    for i, (x_val, n_rate, o_rate) in enumerate(zip(x, naive_rate, opt_rate)):
        ratio = o_rate / n_rate
        ax.annotate(f'{ratio:.1f}x', (x_val, max(n_rate, o_rate)),
                   xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.7)
    
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_ylabel('Points Processed per Second', fontsize=12)
    ax.set_title('Processing Rate: Efficiency Convergence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('results/efficiency_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_overhead_analysis_plot(results_df):
    """Visualize overhead cost vs benefit"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = results_df['n_points']
    speedup = results_df['speedup']
    
    # Left: Speedup with crossover highlight
    ax1.plot(x, speedup, 'd-', color=COLORS['speedup'], linewidth=2.5, markersize=8)
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Break-even (1x)')
    
    # Highlight crossover region
    cross_idx = np.argmax(speedup >= 1)
    if cross_idx > 0:
        ax1.axvspan(x.iloc[0], x.iloc[cross_idx], alpha=0.2, color='red', label='Overhead Dominant')
        ax1.axvspan(x.iloc[cross_idx], x.iloc[-1], alpha=0.2, color='green', label='Benefit Dominant')
    
    ax1.set_xlabel('Number of Points', fontsize=12)
    ax1.set_ylabel('Speedup Factor', fontsize=12)
    ax1.set_title('Optimization Overhead vs Benefit', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Right: Time ratio as percentage
    time_ratio = results_df['optimized_time'] / results_df['naive_time']
    ax2.bar(range(len(x)), time_ratio, color=COLORS['optimized'], alpha=0.7)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    ax2.set_xticks(range(len(x)))
    ax2.set_xticklabels([f'{int(v):,}' for v in x], rotation=45, ha='right')
    ax2.set_xlabel('Number of Points', fontsize=12)
    ax2.set_ylabel('Time Ratio (Optimized / Naive)', fontsize=12)
    ax2.set_title('Relative Time: Lower is Better', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/overhead_benefit_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_cluster_quality_dashboard(results_df):
    """Dashboard of all cluster quality metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    mask = results_df['naive_clusters'] > 0
    filtered = results_df[mask]
    x = filtered['n_points']
    
    # 1. Separation Ratio
    axes[0, 0].plot(x, filtered['naive_separation_ratio'], 'o-', 
                    label='Naive', color=COLORS['naive'], linewidth=2)
    axes[0, 0].plot(x, filtered['optimized_separation_ratio'], 's-', 
                    label='Optimized', color=COLORS['optimized'], linewidth=2)
    axes[0, 0].set_xlabel('Points')
    axes[0, 0].set_ylabel('Separation Ratio')
    axes[0, 0].set_title('Cluster Separation (higher = better)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale('log')
    
    # 2. Compactness
    axes[0, 1].plot(x, filtered['naive_compactness_ratio'], 'o-', 
                    label='Naive', color=COLORS['naive'], linewidth=2)
    axes[0, 1].plot(x, filtered['optimized_compactness_ratio'], 's-', 
                    label='Optimized', color=COLORS['optimized'], linewidth=2)
    axes[0, 1].set_xlabel('Points')
    axes[0, 1].set_ylabel('Compactness Ratio')
    axes[0, 1].set_title('Cluster Compactness (lower = better)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log')
    
    # 3. Noise Reduction
    axes[1, 0].plot(x, filtered['naive_noise_pct'], 'o-', 
                    label='Naive', color=COLORS['naive'], linewidth=2)
    axes[1, 0].plot(x, filtered['optimized_noise_pct'], 's-', 
                    label='Optimized', color=COLORS['optimized'], linewidth=2)
    axes[1, 0].set_xlabel('Points')
    axes[1, 0].set_ylabel('Noise (%)')
    axes[1, 0].set_title('Noise Reduction Trend')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log')
    
    # 4. Cluster Size Distribution
    axes[1, 1].plot(x, filtered['naive_avg_cluster_size'], 'o-', 
                    label='Naive', color=COLORS['naive'], linewidth=2)
    axes[1, 1].plot(x, filtered['optimized_avg_cluster_size'], 's-', 
                    label='Optimized', color=COLORS['optimized'], linewidth=2)
    axes[1, 1].set_xlabel('Points')
    axes[1, 1].set_ylabel('Average Cluster Size')
    axes[1, 1].set_title('Average Cluster Size Growth')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    
    plt.suptitle('Cluster Quality Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/cluster_quality_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_scaling_comparison_plot(results_df):
    """Compare scaling exponents"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = results_df['n_points']
    
    # Normalize times to start at 1 for fair comparison
    naive_norm = results_df['naive_time'] / results_df['naive_time'].iloc[0]
    opt_norm = results_df['optimized_time'] / results_df['optimized_time'].iloc[0]
    
    ax.plot(x, naive_norm, 'o-', label='Naive (Actual)', linewidth=2.5, color=COLORS['naive'])
    ax.plot(x, opt_norm, 's-', label='Optimized (Actual)', linewidth=2.5, color=COLORS['optimized'])
    
    # Theoretical curves (normalized)
    x_norm = x / x.iloc[0]
    ax.plot(x, x_norm**2, '--', label='O(n²) Reference', alpha=0.7, color='gray')
    ax.plot(x, x_norm * np.log2(x_norm), ':', label='O(n log n) Reference', alpha=0.7, color='darkgray')
    
    ax.set_xlabel('Number of Points (log scale)', fontsize=12)
    ax.set_ylabel('Normalized Time (log scale)', fontsize=12)
    ax.set_title('Scaling Behavior Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('results/scaling_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_comprehensive_dashboard(results_df):
    """Comprehensive dashboard with all metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    x = results_df['n_points']
    mask = results_df['naive_clusters'] > 0
    filtered = results_df[mask]
    
    # 1. Execution Time
    ax1 = axes[0, 0]
    ax1.plot(x, results_df['naive_time'], 'o-', label='Naive', linewidth=2, color=COLORS['naive'])
    ax1.plot(x, results_df['optimized_time'], 's-', label='Optimized', linewidth=2, color=COLORS['optimized'])
    ax1.set_xlabel('Points')
    ax1.set_ylabel('Time (s)')
    ax1.set_title('Execution Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # 2. Speedup
    ax2 = axes[0, 1]
    ax2.plot(x, results_df['speedup'], 'd-', color=COLORS['speedup'], linewidth=2)
    ax2.axhline(y=1, color=COLORS['naive'], linestyle='--', alpha=0.7)
    ax2.set_xlabel('Points')
    ax2.set_ylabel('Speedup')
    ax2.set_title(f'Speedup: {results_df["speedup"].max():.1f}x max')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # 3. Clusters
    ax3 = axes[0, 2]
    ax3.plot(filtered['n_points'], filtered['naive_clusters'], 'o-', label='Naive', linewidth=2, color=COLORS['naive'])
    ax3.plot(filtered['n_points'], filtered['optimized_clusters'], 's-', label='Optimized', linewidth=2, color=COLORS['optimized'])
    ax3.set_xlabel('Points')
    ax3.set_ylabel('Clusters')
    ax3.set_title('Clusters Found')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # 4. Noise Ratio
    ax4 = axes[1, 0]
    ax4.plot(filtered['n_points'], filtered['naive_noise_pct'], 'o-', label='Naive', linewidth=2, color=COLORS['naive'])
    ax4.plot(filtered['n_points'], filtered['optimized_noise_pct'], 's-', label='Optimized', linewidth=2, color=COLORS['optimized'])
    ax4.set_xlabel('Points')
    ax4.set_ylabel('Noise (%)')
    ax4.set_title('Noise Ratio')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    
    # 5. Cluster Quality
    ax5 = axes[1, 1]
    ax5.plot(filtered['n_points'], filtered['naive_separation_ratio'], 'o-', label='Naive', linewidth=2, color=COLORS['naive'])
    ax5.plot(filtered['n_points'], filtered['optimized_separation_ratio'], 's-', label='Optimized', linewidth=2, color=COLORS['optimized'])
    ax5.set_xlabel('Points')
    ax5.set_ylabel('Separation Ratio')
    ax5.set_title('Cluster Quality (higher = better)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xscale('log')
    
    # 6. Processing Rate
    ax6 = axes[1, 2]
    naive_rate = results_df['n_points'] / results_df['naive_time']
    opt_rate = results_df['n_points'] / results_df['optimized_time']
    ax6.plot(x, naive_rate, 'o-', label='Naive', linewidth=2, color=COLORS['naive'])
    ax6.plot(x, opt_rate, 's-', label='Optimized', linewidth=2, color=COLORS['optimized'])
    ax6.set_xlabel('Points')
    ax6.set_ylabel('Points/sec')
    ax6.set_title('Processing Rate')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xscale('log')
    ax6.set_yscale('log')
    
    plt.suptitle('DBSCAN Optimization: Comprehensive Performance Dashboard', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/comprehensive_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_performance_table(results_df):
    """Generate a performance summary table as an image"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for _, row in results_df.iterrows():
        table_data.append([
            f"{int(row['n_points']):,}",
            f"{row['naive_time']:.2f}s",
            f"{row['optimized_time']:.2f}s",
            f"{row['speedup']:.1f}x",
            f"{row['naive_clusters']:.0f}",
            f"{row['optimized_clusters']:.0f}",
            f"{row['naive_noise_pct']:.1f}%",
            f"{row['optimized_noise_pct']:.1f}%",
            f"{row['naive_separation_ratio']:.1f}",
            f"{row['optimized_separation_ratio']:.1f}"
        ])
    
    columns = ['Points', 'Naive Time', 'Opt Time', 'Speedup', 
               'Naive Clusters', 'Opt Clusters', 'Naive Noise', 'Opt Noise',
               'Naive Sep Ratio', 'Opt Sep Ratio']
    
    table = ax.table(cellText=table_data, colLabels=columns, 
                     cellLoc='center', loc='center',
                     colColours=[COLORS['optimized']] * len(columns))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    # Highlight best speedup row
    best_idx = results_df['speedup'].idxmax()
    for j in range(len(columns)):
        table[(best_idx + 1, j)].set_facecolor('#E8F5E9')
    
    ax.set_title('Performance Comparison Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('results/performance_table.png', dpi=150, bbox_inches='tight')
    plt.close()

def run_comparison():
    print("="*80)
    print("FAIR COMPARISON: Naive vs Optimized DBSCAN")
    print("Both implementations use EPSG:2263 projection + Euclidean distance in meters")
    print("="*80)
    
    # Test sizes - you can modify this list
    test_sizes = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    results = []
    
    # Create cluster map folder
    os.makedirs('results/cluster_maps', exist_ok=True)
    
    for n_points in test_sizes:
        print(f"\n{'='*60}")
        print(f"Testing with {n_points:,} points")
        print(f"{'='*60}")
        
        print("Loading data...")
        df = load_uber_2014_data(sample_size=n_points, filter_nyc=True)
        
        if df is None:
            continue
        
        X, timestamps, _ = prepare_gps_data(df)
        
        eps_km = 0.5
        min_samples = 30
        
        print(f"\nParameters: eps={eps_km}km (500 meters), min_samples={min_samples}, Points={len(X):,}")
        
        # ========== NAIVE DBSCAN ==========
        print("\n[1/2] Running Naive DBSCAN (Vectorized O(n²))...")
        naive_dbscan = NaiveDBSCAN(eps=eps_km, min_samples=min_samples)
        
        start = time.time()
        naive_labels = naive_dbscan.fit_predict(X)
        naive_time = time.time() - start
        
        # Get projection for metrics
        X_proj = naive_dbscan._project(X)
        naive_metrics = compute_cluster_metrics(naive_labels, X_proj)
        
        print(f"  Time: {naive_time:.2f}s | Clusters: {naive_metrics['n_clusters']:.0f} | Noise: {naive_metrics['noise_pct']:.1f}%")
        
        # Generate cluster map for naive (only for sizes up to 50k to save time)
        if n_points <= 50000:
            generate_cluster_map(X, naive_labels, 'Naive DBSCAN', n_points, 
                                f'results/cluster_maps/naive_{n_points}.png')
        
        # ========== OPTIMIZED DBSCAN ==========
        print("\n[2/2] Running Optimized DBSCAN (Custom Grid Index)...")
        opt_dbscan = OptimizedDBSCAN(eps=eps_km, min_samples=min_samples)
        
        start = time.time()
        opt_labels = opt_dbscan.fit_predict(X)
        opt_time = time.time() - start
        
        # Get projection for metrics
        X_proj = opt_dbscan._project(X)
        opt_metrics = compute_cluster_metrics(opt_labels, X_proj)
        
        print(f"  Time: {opt_time:.2f}s | Clusters: {opt_metrics['n_clusters']:.0f} | Noise: {opt_metrics['noise_pct']:.1f}%")
        
        # Generate cluster map for optimized (only for sizes up to 50k to save time)
        if n_points <= 50000:
            generate_cluster_map(X, opt_labels, 'Optimized DBSCAN', n_points,
                                f'results/cluster_maps/optimized_{n_points}.png')
        
        speedup = naive_time / opt_time
        print(f"\n🚀 SPEEDUP: {speedup:.1f}x faster!")
        
        results.append({
            'n_points': len(X),
            'naive_time': naive_time,
            'optimized_time': opt_time,
            'speedup': speedup,
            'naive_clusters': naive_metrics['n_clusters'],
            'optimized_clusters': opt_metrics['n_clusters'],
            'naive_noise_pct': naive_metrics['noise_pct'],
            'optimized_noise_pct': opt_metrics['noise_pct'],
            'naive_avg_cluster_size': naive_metrics['avg_cluster_size'],
            'optimized_avg_cluster_size': opt_metrics['avg_cluster_size'],
            'naive_separation_ratio': naive_metrics['separation_ratio'],
            'optimized_separation_ratio': opt_metrics['separation_ratio'],
            'naive_compactness_ratio': naive_metrics['compactness_ratio'],
            'optimized_compactness_ratio': opt_metrics['compactness_ratio']
        })
    
    if results:
        results_df = pd.DataFrame(results)
        os.makedirs('results', exist_ok=True)
        results_df.to_csv('results/fair_comparison_results.csv', index=False)
        
        # Generate all visualizations
        generate_execution_time_plot(results_df)
        generate_speedup_plot(results_df)
        generate_cluster_comparison_plot(results_df)
        generate_noise_comparison_plot(results_df)
        generate_cluster_quality_plot(results_df)
        generate_cluster_size_plot(results_df)
        generate_processing_rate_plot(results_df)
        generate_scalability_analysis(results_df)
        generate_scaling_exponent_plot(results_df)
        generate_speedup_trend_plot(results_df)
        generate_performance_heatmap(results_df)
        generate_efficiency_convergence_plot(results_df)
        generate_overhead_analysis_plot(results_df)
        generate_cluster_quality_dashboard(results_df)
        generate_scaling_comparison_plot(results_df)
        generate_comprehensive_dashboard(results_df)
        generate_performance_table(results_df)
        
        # Print size-independent observations
        print_size_independent_observations(results_df)
        
        print("\n" + "="*80)
        print("✅ All visualizations saved to results/ folder")
        print("="*80)
        print("\nGenerated files:")
        print("  1. execution_time_comparison.png")
        print("  2. speedup_factor.png")
        print("  3. cluster_comparison.png")
        print("  4. noise_comparison.png")
        print("  5. cluster_quality.png")
        print("  6. cluster_size_comparison.png")
        print("  7. processing_rate.png")
        print("  8. scalability_analysis.png")
        print("  9. scaling_exponent_analysis.png")
        print("  10. speedup_trend.png")
        print("  11. performance_heatmap.png")
        print("  12. efficiency_convergence.png")
        print("  13. overhead_benefit_analysis.png")
        print("  14. cluster_quality_dashboard.png")
        print("  15. scaling_comparison.png")
        print("  16. comprehensive_dashboard.png")
        print("  17. performance_table.png")
        print("  18. cluster_maps/ - cluster maps (up to 50k points)")
        
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(results_df[['n_points', 'naive_time', 'optimized_time', 'speedup', 
                          'naive_clusters', 'optimized_clusters']].to_string(index=False))
        print(f"\nBest Speedup: {results_df['speedup'].max():.1f}x at {results_df.loc[results_df['speedup'].idxmax(), 'n_points']:.0f} points")
        print(f"Avg Speedup: {results_df['speedup'].mean():.1f}x")
        
        return results_df
    return None

if __name__ == "__main__":
    run_comparison()