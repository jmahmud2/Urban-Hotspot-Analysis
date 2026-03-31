"""
FAIR COMPARISON: Naive DBSCAN vs Optimized DBSCAN
- Multiple runs (5 runs per test size) with mean ± std
- ARI between naive and optimized for correctness
- Phase-wise timing breakdown
- Both implementations have duplicate prevention
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score
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

# Number of runs for statistical significance
N_RUNS = 5

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
    
    avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
    cluster_size_std = np.std(cluster_sizes) if cluster_sizes else 0
    
    # Intra-cluster distances - SAMPLING to avoid memory explosion
    intra_distances = []
    for label in set(labels):
        if label != -1:
            cluster_points = X_proj[labels == label]
            if len(cluster_points) > 1:
                if len(cluster_points) > 500:
                    sample_idx = np.random.choice(len(cluster_points), 500, replace=False)
                    cluster_points = cluster_points[sample_idx]
                if len(cluster_points) > 1:
                    distances = pdist(cluster_points)
                    intra_distances.extend(distances)
    
    avg_intra_distance = np.mean(intra_distances) if intra_distances else 0
    
    # Inter-cluster distances (centroids only)
    inter_distances = []
    for i, c1 in enumerate(cluster_centroids):
        for j, c2 in enumerate(cluster_centroids):
            if i < j:
                dist = np.linalg.norm(c1 - c2)
                inter_distances.append(dist)
    
    avg_inter_distance = np.mean(inter_distances) if inter_distances else 0
    separation_ratio = avg_inter_distance / (avg_intra_distance + 1e-10) if avg_intra_distance > 0 else 0
    
    return {
        'n_clusters': n_clusters,
        'noise_pct': noise_pct,
        'avg_cluster_size': avg_cluster_size,
        'cluster_size_std': cluster_size_std,
        'separation_ratio': separation_ratio
    }

def generate_cluster_map(X, labels, method_name, n_points, output_path):
    """Generate a neutral cluster map without subjective landmarks"""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    unique_labels = np.unique(labels)
    cluster_labels = unique_labels[unique_labels != -1]
    
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(cluster_labels))))
    
    for i, label in enumerate(cluster_labels):
        cluster_points = X[labels == label]
        ax.scatter(cluster_points[:, 1], cluster_points[:, 0], 
                  c=[colors[i % len(colors)]], s=10, alpha=0.7, 
                  label=f'Cluster {label}', edgecolors='white', linewidth=0.3)
    
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

def print_observations(results_df):
    """Print key observations with statistical rigor"""
    
    print("\n" + "="*80)
    print("KEY OBSERVATIONS (Based on {} runs per test size)".format(N_RUNS))
    print("="*80)
    
    # Crossover point (where speedup >= 1 consistently)
    cross_df = results_df[results_df['speedup_mean'] >= 1]
    if len(cross_df) > 0:
        print(f"\n1. CROSSOVER: Optimized becomes consistently faster at {cross_df['n_points'].iloc[0]:,} points")
    
    # Scaling exponents
    x = np.log(results_df['n_points'])
    naive_exp = np.polyfit(x, np.log(results_df['naive_time_mean']), 1)[0]
    opt_exp = np.polyfit(x, np.log(results_df['optimized_time_mean']), 1)[0]
    print(f"\n2. SCALING: Naive exponent: {naive_exp:.2f} | Optimized: {opt_exp:.2f} (theoretical O(n²)=2.0)")
    
    # Best speedup
    best_idx = results_df['speedup_mean'].idxmax()
    best_speedup = results_df.loc[best_idx, 'speedup_mean']
    best_speedup_std = results_df.loc[best_idx, 'speedup_std']
    best_points = results_df.loc[best_idx, 'n_points']
    print(f"\n3. BEST SPEEDUP: {best_speedup:.1f} ± {best_speedup_std:.1f}x at {best_points:.0f} points")
    
    # Noise reduction
    noise_start = results_df['optimized_noise_pct'].iloc[0]
    noise_end = results_df['optimized_noise_pct'].iloc[-1]
    print(f"\n4. NOISE: Decreased from {noise_start:.1f}% to {noise_end:.1f}%")
    
    # ARI (correctness)
    ari_mean = results_df['ari_mean'].mean()
    print(f"\n5. CORRECTNESS: Avg ARI between naive and optimized: {ari_mean:.4f} (1.0 = identical)")

def run_comparison():
    print("="*80)
    print("FAIR COMPARISON: Naive vs Optimized DBSCAN")
    print(f"Each test size: {N_RUNS} runs (mean ± std reported)")
    print("="*80)
    
    # Test sizes - modify as needed
    test_sizes = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    results = []
    
    # Create folders
    os.makedirs('results/cluster_maps', exist_ok=True)
    
    for n_points in test_sizes:
        print(f"\n{'='*60}")
        print(f"Testing with {n_points:,} points ({N_RUNS} runs each)")
        print(f"{'='*60}")
        
        print("Loading data...")
        df = load_uber_2014_data(sample_size=n_points, filter_nyc=True)
        
        if df is None:
            continue
        
        X, timestamps, _ = prepare_gps_data(df)
        
        eps_km = 0.5
        min_samples = 30
        
        print(f"\nParameters: eps={eps_km}km, min_samples={min_samples}, Points={len(X):,}")
        
        # ========== MULTIPLE RUNS ==========
        naive_times = []
        naive_clusters = []
        naive_noise = []
        naive_sep = []
        
        opt_times = []
        opt_clusters = []
        opt_noise = []
        opt_sep = []
        
        ari_scores = []
        
        # Phase timing storage
        naive_proj_times = []
        naive_clust_times = []
        opt_proj_times = []
        opt_index_times = []
        opt_clust_times = []
        
        for run in range(N_RUNS):
            print(f"\n  Run {run + 1}/{N_RUNS}...")
            
            # ========== NAIVE DBSCAN with phase timing ==========
            naive_dbscan = NaiveDBSCAN(eps=eps_km, min_samples=min_samples)
            
            # Phase 1: Projection
            proj_start = time.time()
            X_proj = naive_dbscan._project(X)
            proj_time = time.time() - proj_start
            
            # Phase 2: Clustering
            clust_start = time.time()
            naive_labels = naive_dbscan.fit_predict_phased(X, X_proj)
            clust_time = time.time() - clust_start
            
            naive_total = proj_time + clust_time
            naive_times.append(naive_total)
            naive_proj_times.append(proj_time)
            naive_clust_times.append(clust_time)
            
            naive_metrics = compute_cluster_metrics(naive_labels, X_proj)
            naive_clusters.append(naive_metrics['n_clusters'])
            naive_noise.append(naive_metrics['noise_pct'])
            naive_sep.append(naive_metrics['separation_ratio'])
            
            # ========== OPTIMIZED DBSCAN with phase timing ==========
            opt_dbscan = OptimizedDBSCAN(eps=eps_km, min_samples=min_samples)
            
            # Phase 1: Projection
            proj_start = time.time()
            X_proj = opt_dbscan._project(X)
            proj_time = time.time() - proj_start
            
            # Phase 2: Build grid index
            index_start = time.time()
            opt_dbscan.grid_index = opt_dbscan._build_grid_index(X_proj)
            index_time = time.time() - index_start
            
            # Phase 3: Clustering
            clust_start = time.time()
            opt_labels = opt_dbscan.fit_predict_phased(X_proj)
            clust_time = time.time() - clust_start
            
            opt_total = proj_time + index_time + clust_time
            opt_times.append(opt_total)
            opt_proj_times.append(proj_time)
            opt_index_times.append(index_time)
            opt_clust_times.append(clust_time)
            
            opt_metrics = compute_cluster_metrics(opt_labels, X_proj)
            opt_clusters.append(opt_metrics['n_clusters'])
            opt_noise.append(opt_metrics['noise_pct'])
            opt_sep.append(opt_metrics['separation_ratio'])
            
            # ARI between naive and optimized
            ari = adjusted_rand_score(naive_labels, opt_labels)
            ari_scores.append(ari)
            
            print(f"    Naive: {naive_total:.2f}s | Opt: {opt_total:.2f}s | ARI: {ari:.4f}")
        
        # Calculate statistics
        naive_mean = np.mean(naive_times)
        naive_std = np.std(naive_times)
        opt_mean = np.mean(opt_times)
        opt_std = np.std(opt_times)
        speedup_mean = naive_mean / opt_mean
        speedup_std = speedup_mean * np.sqrt((naive_std/naive_mean)**2 + (opt_std/opt_mean)**2)
        
        ari_mean = np.mean(ari_scores)
        ari_std = np.std(ari_scores)
        
        print(f"\n📊 SUMMARY:")
        print(f"   Naive: {naive_mean:.2f}s ± {naive_std:.2f}s")
        print(f"   Optimized: {opt_mean:.2f}s ± {opt_std:.2f}s")
        print(f"   Speedup: {speedup_mean:.1f}x ± {speedup_std:.1f}x")
        print(f"   ARI (Naive vs Opt): {ari_mean:.4f} ± {ari_std:.4f}")
        
        # Generate cluster maps for selected sizes
        if n_points in [10000, 50000, 100000]:
            # Use last run's labels for map
            generate_cluster_map(X, naive_labels, 'Naive DBSCAN', n_points, 
                                f'results/cluster_maps/naive_{n_points}.png')
            generate_cluster_map(X, opt_labels, 'Optimized DBSCAN', n_points,
                                f'results/cluster_maps/optimized_{n_points}.png')
        
        results.append({
            'n_points': len(X),
            'naive_time_mean': naive_mean,
            'naive_time_std': naive_std,
            'optimized_time_mean': opt_mean,
            'optimized_time_std': opt_std,
            'speedup_mean': speedup_mean,
            'speedup_std': speedup_std,
            'ari_mean': ari_mean,
            'ari_std': ari_std,
            'naive_clusters': np.mean(naive_clusters),
            'optimized_clusters': np.mean(opt_clusters),
            'naive_noise_pct': np.mean(naive_noise),
            'optimized_noise_pct': np.mean(opt_noise),
            'naive_separation_ratio': np.mean(naive_sep),
            'optimized_separation_ratio': np.mean(opt_sep),
            'naive_proj_time': np.mean(naive_proj_times),
            'naive_clust_time': np.mean(naive_clust_times),
            'opt_proj_time': np.mean(opt_proj_times),
            'opt_index_time': np.mean(opt_index_times),
            'opt_clust_time': np.mean(opt_clust_times)
        })
    
    if results:
        results_df = pd.DataFrame(results)
        os.makedirs('results', exist_ok=True)
        results_df.to_csv('results/fair_comparison_results.csv', index=False)
        
        # Generate visualizations
        generate_execution_time_plot(results_df)
        generate_speedup_plot(results_df)
        generate_ari_plot(results_df)
        generate_phase_timing_plot(results_df)
        generate_processing_rate_plot(results_df)
        generate_cluster_comparison_plot(results_df)
        generate_noise_comparison_plot(results_df)
        generate_cluster_quality_plot(results_df)
        generate_performance_table(results_df)
        
        print_observations(results_df)
        
        print("\n" + "="*80)
        print("✅ Results saved to results/ folder")
        print("="*80)
        print("\nGenerated files:")
        print("  1. execution_time_comparison.png (with error bars)")
        print("  2. speedup_factor.png (with error bars)")
        print("  3. ari_comparison.png (correctness verification)")
        print("  4. phase_timing_breakdown.png")
        print("  5. processing_rate.png")
        print("  6. cluster_comparison.png")
        print("  7. noise_comparison.png")
        print("  8. cluster_quality.png")
        print("  9. performance_table.png")
        print("  10. fair_comparison_results.csv")
        print("  11. cluster_maps/ - 6 cluster maps (3 naive + 3 optimized)")
        
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(results_df[['n_points', 'naive_time_mean', 'optimized_time_mean', 'speedup_mean', 
                          'ari_mean', 'naive_clusters', 'optimized_clusters']].to_string(index=False))
        
        return results_df
    return None

def generate_execution_time_plot(results_df):
    """Execution time comparison with error bars"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = results_df['n_points']
    
    ax.errorbar(x, results_df['naive_time_mean'], yerr=results_df['naive_time_std'],
                label='Naive O(n²)', linewidth=2.5, marker='o', markersize=9,
                capsize=5, color=COLORS['naive'], markerfacecolor='white', markeredgewidth=2)
    
    ax.errorbar(x, results_df['optimized_time_mean'], yerr=results_df['optimized_time_std'],
                label='Optimized O(n log n)', linewidth=2.5, marker='s', markersize=9,
                capsize=5, color=COLORS['optimized'], markerfacecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Execution Time: Naive vs Optimized DBSCAN (with ±1σ error bars)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('results/execution_time_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_speedup_plot(results_df):
    """Speedup factor plot with error bars"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = results_df['n_points']
    
    ax.errorbar(x, results_df['speedup_mean'], yerr=results_df['speedup_std'],
                label='Speedup', linewidth=2.5, marker='d', markersize=9,
                capsize=5, color=COLORS['speedup'], markerfacecolor='white', markeredgewidth=2)
    
    ax.axhline(y=1, color=COLORS['naive'], linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline (1x)')
    ax.fill_between(x, 1, results_df['speedup_mean'], 
                     where=(results_df['speedup_mean'] > 1), color=COLORS['speedup'], alpha=0.25)
    
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_ylabel('Speedup Factor', fontsize=12)
    ax.set_title(f'Spatial Indexing Speedup\n{results_df["speedup_mean"].max():.1f}x at {results_df.loc[results_df["speedup_mean"].idxmax(), "n_points"]:.0f} points', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('results/speedup_factor.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_ari_plot(results_df):
    """ARI plot showing correctness (naive vs optimized produce same clusters)"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = results_df['n_points']
    
    ax.errorbar(x, results_df['ari_mean'], yerr=results_df['ari_std'],
                label='Adjusted Rand Index', linewidth=2.5, marker='o', markersize=9,
                capsize=5, color=COLORS['optimized'], markerfacecolor='white', markeredgewidth=2)
    ax.axhline(y=0.99, color=COLORS['naive'], linestyle='--', linewidth=1.5, alpha=0.7, label='Near Perfect (0.99)')
    
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_ylabel('Adjusted Rand Index', fontsize=12)
    ax.set_title('Correctness: Naive vs Optimized Produce Identical Clusters', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_ylim(0.98, 1.01)
    
    plt.tight_layout()
    plt.savefig('results/ari_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_phase_timing_plot(results_df):
    """Phase-wise timing breakdown for optimized implementation"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = results_df['n_points']
    
    ax.plot(x, results_df['opt_proj_time'], 'o-', 
            label='Projection', linewidth=2, markersize=6, color='#1E88E5')
    ax.plot(x, results_df['opt_index_time'], 's-', 
            label='Grid Index Build', linewidth=2, markersize=6, color='#FFB347')
    ax.plot(x, results_df['opt_clust_time'], '^-', 
            label='Clustering', linewidth=2, markersize=6, color='#2E8B57')
    
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Optimized DBSCAN: Phase-Wise Timing Breakdown', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('results/phase_timing_breakdown.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_processing_rate_plot(results_df):
    """Processing rate (points per second) with error bars"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = results_df['n_points']
    
    naive_rate = results_df['n_points'] / results_df['naive_time_mean']
    naive_rate_std = naive_rate * (results_df['naive_time_std'] / results_df['naive_time_mean'])
    opt_rate = results_df['n_points'] / results_df['optimized_time_mean']
    opt_rate_std = opt_rate * (results_df['optimized_time_std'] / results_df['optimized_time_mean'])
    
    ax.errorbar(x, naive_rate, yerr=naive_rate_std,
                label='Naive O(n²)', linewidth=2.5, marker='o', markersize=9,
                capsize=5, color=COLORS['naive'], markerfacecolor='white', markeredgewidth=2)
    
    ax.errorbar(x, opt_rate, yerr=opt_rate_std,
                label='Optimized O(n log n)', linewidth=2.5, marker='s', markersize=9,
                capsize=5, color=COLORS['optimized'], markerfacecolor='white', markeredgewidth=2)
    
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
    
    plt.tight_layout()
    plt.savefig('results/noise_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_cluster_quality_plot(results_df):
    """Cluster quality (separation ratio)"""
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
    ax.set_ylabel('Separation Ratio (higher = better)', fontsize=12)
    ax.set_title('Cluster Quality: Separation Ratio', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('results/cluster_quality.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_performance_table(results_df):
    """Generate performance summary table"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    
    table_data = []
    for _, row in results_df.iterrows():
        table_data.append([
            f"{int(row['n_points']):,}",
            f"{row['naive_time_mean']:.2f}s ± {row['naive_time_std']:.2f}",
            f"{row['optimized_time_mean']:.2f}s ± {row['optimized_time_std']:.2f}",
            f"{row['speedup_mean']:.1f}x ± {row['speedup_std']:.1f}",
            f"{row['ari_mean']:.4f} ± {row['ari_std']:.4f}",
            f"{row['naive_clusters']:.0f}",
            f"{row['optimized_clusters']:.0f}",
            f"{row['naive_noise_pct']:.1f}%",
            f"{row['optimized_noise_pct']:.1f}%"
        ])
    
    columns = ['Points', 'Naive Time', 'Opt Time', 'Speedup', 'ARI', 
               'Naive Clusters', 'Opt Clusters', 'Naive Noise', 'Opt Noise']
    
    table = ax.table(cellText=table_data, colLabels=columns, 
                     cellLoc='center', loc='center',
                     colColours=[COLORS['optimized']] * len(columns))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    # Highlight best speedup row
    best_idx = results_df['speedup_mean'].idxmax()
    for j in range(len(columns)):
        table[(best_idx + 1, j)].set_facecolor('#E8F5E9')
    
    ax.set_title('Performance Comparison Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('results/performance_table.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_comparison()