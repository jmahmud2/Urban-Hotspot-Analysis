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
    
    avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
    cluster_size_std = np.std(cluster_sizes) if cluster_sizes else 0
    
    # Intra-cluster distances - SAMPLING to avoid memory explosion
    intra_distances = []
    for label in set(labels):
        if label != -1:
            cluster_points = X_proj[labels == label]
            if len(cluster_points) > 1:
                # Sample up to 500 points per cluster
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
    
    # Add legend (limited to top 10)
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
    """Print key observations"""
    
    print("\n" + "="*80)
    print("KEY OBSERVATIONS")
    print("="*80)
    
    # Crossover point
    cross_df = results_df[results_df['speedup'] >= 1]
    if len(cross_df) > 0:
        print(f"\n1. CROSSOVER: Optimized becomes faster at {cross_df['n_points'].iloc[0]:,} points")
    
    # Scaling exponents
    x = np.log(results_df['n_points'])
    naive_exp = np.polyfit(x, np.log(results_df['naive_time']), 1)[0]
    opt_exp = np.polyfit(x, np.log(results_df['optimized_time']), 1)[0]
    print(f"\n2. SCALING: Naive exponent: {naive_exp:.2f} | Optimized: {opt_exp:.2f}")
    
    # Speedup
    best_idx = results_df['speedup'].idxmax()
    print(f"\n3. BEST SPEEDUP: {results_df['speedup'].max():.1f}x at {results_df.loc[best_idx, 'n_points']:.0f} points")
    
    # Noise reduction
    noise_start = results_df['optimized_noise_pct'].iloc[0]
    noise_end = results_df['optimized_noise_pct'].iloc[-1]
    print(f"\n4. NOISE: Decreased from {noise_start:.1f}% to {noise_end:.1f}%")
    
    # Processing rate
    naive_rate_start = results_df['n_points'].iloc[0] / results_df['naive_time'].iloc[0]
    naive_rate_end = results_df['n_points'].iloc[-1] / results_df['naive_time'].iloc[-1]
    opt_rate_start = results_df['n_points'].iloc[0] / results_df['optimized_time'].iloc[0]
    opt_rate_end = results_df['n_points'].iloc[-1] / results_df['optimized_time'].iloc[-1]
    print(f"\n5. PROCESSING RATE:")
    print(f"   Naive: {naive_rate_start:.0f} → {naive_rate_end:.0f} pts/sec")
    print(f"   Optimized: {opt_rate_start:.0f} → {opt_rate_end:.0f} pts/sec")

def run_comparison():
    print("="*80)
    print("FAIR COMPARISON: Naive vs Optimized DBSCAN")
    print("="*80)
    
    # Test sizes - modify as needed
    test_sizes = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    results = []
    
    # Create folders
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
        
        print(f"\nParameters: eps={eps_km}km, min_samples={min_samples}, Points={len(X):,}")
        
        # ========== NAIVE DBSCAN ==========
        print("\n[1/2] Running Naive DBSCAN...")
        naive_dbscan = NaiveDBSCAN(eps=eps_km, min_samples=min_samples)
        
        start = time.time()
        naive_labels = naive_dbscan.fit_predict(X)
        naive_time = time.time() - start
        
        X_proj = naive_dbscan._project(X)
        naive_metrics = compute_cluster_metrics(naive_labels, X_proj)
        
        naive_rate = len(X) / naive_time
        print(f"  Time: {naive_time:.2f}s | Rate: {naive_rate:.0f} pts/sec | Clusters: {naive_metrics['n_clusters']:.0f} | Noise: {naive_metrics['noise_pct']:.1f}%")
        
        # ========== OPTIMIZED DBSCAN ==========
        print("\n[2/2] Running Optimized DBSCAN...")
        opt_dbscan = OptimizedDBSCAN(eps=eps_km, min_samples=min_samples)
        
        start = time.time()
        opt_labels = opt_dbscan.fit_predict(X)
        opt_time = time.time() - start
        
        X_proj = opt_dbscan._project(X)
        opt_metrics = compute_cluster_metrics(opt_labels, X_proj)
        
        opt_rate = len(X) / opt_time
        print(f"  Time: {opt_time:.2f}s | Rate: {opt_rate:.0f} pts/sec | Clusters: {opt_metrics['n_clusters']:.0f} | Noise: {opt_metrics['noise_pct']:.1f}%")
        
        # Generate cluster maps for selected sizes
        if n_points in [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]:
            generate_cluster_map(X, naive_labels, 'Naive DBSCAN', n_points, 
                                f'results/cluster_maps/naive_{n_points}.png')
            generate_cluster_map(X, opt_labels, 'Optimized DBSCAN', n_points,
                                f'results/cluster_maps/optimized_{n_points}.png')
        
        speedup = naive_time / opt_time
        print(f"\n🚀 SPEEDUP: {speedup:.1f}x faster!")
        
        results.append({
            'n_points': len(X),
            'naive_time': naive_time,
            'optimized_time': opt_time,
            'naive_rate': naive_rate,
            'optimized_rate': opt_rate,
            'speedup': speedup,
            'naive_clusters': naive_metrics['n_clusters'],
            'optimized_clusters': opt_metrics['n_clusters'],
            'naive_noise_pct': naive_metrics['noise_pct'],
            'optimized_noise_pct': opt_metrics['noise_pct'],
            'naive_avg_cluster_size': naive_metrics['avg_cluster_size'],
            'optimized_avg_cluster_size': opt_metrics['avg_cluster_size'],
            'naive_separation_ratio': naive_metrics['separation_ratio'],
            'optimized_separation_ratio': opt_metrics['separation_ratio']
        })
    
    if results:
        results_df = pd.DataFrame(results)
        os.makedirs('results', exist_ok=True)
        results_df.to_csv('results/fair_comparison_results.csv', index=False)
        
        # Generate essential visualizations
        generate_execution_time_plot(results_df)
        generate_speedup_plot(results_df)
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
        print("  1. execution_time_comparison.png")
        print("  2. speedup_factor.png")
        print("  3. processing_rate.png")
        print("  4. cluster_comparison.png")
        print("  5. noise_comparison.png")
        print("  6. cluster_quality.png")
        print("  7. performance_table.png")
        print("  8. fair_comparison_results.csv")
        print("  9. cluster_maps/ - 6 cluster maps (3 naive + 3 optimized)")
        
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(results_df[['n_points', 'naive_time', 'optimized_time', 'naive_rate', 'optimized_rate', 
                          'speedup', 'naive_clusters', 'optimized_clusters']].to_string(index=False))
        
        return results_df
    return None

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
    
    plt.tight_layout()
    plt.savefig('results/speedup_factor.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_processing_rate_plot(results_df):
    """Processing rate (points per second) comparison"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = results_df['n_points']
    
    ax.plot(x, results_df['naive_rate'], 'o-', 
            label='Naive O(n²)', linewidth=2.5, markersize=9,
            color=COLORS['naive'], markerfacecolor='white', markeredgewidth=2)
    ax.plot(x, results_df['optimized_rate'], 's-', 
            label='Optimized O(n log n)', linewidth=2.5, markersize=9,
            color=COLORS['optimized'], markerfacecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_ylabel('Points Processed per Second', fontsize=12)
    ax.set_title('Processing Rate: Naive vs Optimized DBSCAN', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add value labels for optimized rate
    for _, row in results_df.iterrows():
        ax.annotate(f'{row["optimized_rate"]:.0f}', 
                   (row['n_points'], row['optimized_rate']),
                   textcoords="offset points", xytext=(5, 5), ha='center', fontsize=9)
    
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
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    table_data = []
    for _, row in results_df.iterrows():
        table_data.append([
            f"{int(row['n_points']):,}",
            f"{row['naive_time']:.2f}s",
            f"{row['optimized_time']:.2f}s",
            f"{row['naive_rate']:.0f}",
            f"{row['optimized_rate']:.0f}",
            f"{row['speedup']:.1f}x",
            f"{row['naive_clusters']:.0f}",
            f"{row['optimized_clusters']:.0f}",
            f"{row['naive_noise_pct']:.1f}%",
            f"{row['optimized_noise_pct']:.1f}%"
        ])
    
    columns = ['Points', 'Naive Time', 'Opt Time', 'Naive Rate', 'Opt Rate',
               'Speedup', 'Naive Clusters', 'Opt Clusters', 'Naive Noise', 'Opt Noise']
    
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

if __name__ == "__main__":
    run_comparison()