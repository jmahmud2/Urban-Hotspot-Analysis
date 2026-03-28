"""
FAIR COMPARISON: Naive DBSCAN (O(n²)) vs Optimized DBSCAN (O(n log n))
Comprehensive analysis with multiple visualizations and proper metrics
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, silhouette_score
from load_uber_data import load_uber_2014_data, prepare_gps_data
from src.naive_dbscan import NaiveDBSCAN
from src.optimized_dbscan import OptimizedDBSCAN

# Set style for professional visualizations
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

def run_comparison():
    print("="*80)
    print("FAIR COMPARISON: Naive DBSCAN vs Optimized DBSCAN")
    print("Both implementations use identical inputs and metrics")
    print("="*80)
    
    # Test sizes - progressive
    test_sizes = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    results = []
    
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
        
        naive_clusters = naive_dbscan.n_clusters_
        naive_noise = np.sum(naive_labels == -1)
        naive_noise_pct = naive_noise / len(X) * 100
        
        print(f"  Time: {naive_time:.2f}s | Clusters: {naive_clusters} | Noise: {naive_noise_pct:.1f}%")
        
        # ========== OPTIMIZED DBSCAN ==========
        print("\n[2/2] Running Optimized DBSCAN (Custom Grid Index)...")
        opt_dbscan = OptimizedDBSCAN(eps=eps_km, min_samples=min_samples)
        
        start = time.time()
        opt_labels = opt_dbscan.fit_predict(X)
        opt_time = time.time() - start
        
        opt_clusters = opt_dbscan.n_clusters_
        opt_noise = np.sum(opt_labels == -1)
        opt_noise_pct = opt_noise / len(X) * 100
        
        print(f"  Time: {opt_time:.2f}s | Clusters: {opt_clusters} | Noise: {opt_noise_pct:.1f}%")
        
        # ========== CLUSTER SIMILARITY METRICS ==========
        ari = adjusted_rand_score(naive_labels, opt_labels)
        
        # Silhouette score only if there are clusters (not all noise)
        try:
            if opt_clusters > 1 and len(np.unique(opt_labels)) > 1:
                # Project coordinates for silhouette score
                from src.optimized_dbscan import OptimizedDBSCAN as OptTemp
                temp = OptTemp(eps=eps_km, min_samples=min_samples)
                X_proj = temp._project(X)
                silhouette = silhouette_score(X_proj, opt_labels)
            else:
                silhouette = 0.0
        except:
            silhouette = 0.0
        
        speedup = naive_time / opt_time if opt_time > 0 else 0
        
        print(f"\n🚀 SPEEDUP: {speedup:.1f}x faster!")
        print(f"📊 Adjusted Rand Index: {ari:.4f} (1.0 = identical clusters)")
        print(f"📊 Silhouette Score: {silhouette:.4f} (higher = better clusters)")
        
        results.append({
            'n_points': len(X),
            'naive_time': naive_time,
            'optimized_time': opt_time,
            'speedup': speedup,
            'adjusted_rand_index': ari,
            'silhouette_score': silhouette,
            'naive_clusters': naive_clusters,
            'optimized_clusters': opt_clusters,
            'naive_noise_pct': naive_noise_pct,
            'optimized_noise_pct': opt_noise_pct,
            'naive_pts_per_sec': len(X) / naive_time,
            'optimized_pts_per_sec': len(X) / opt_time
        })
    
    if results:
        results_df = pd.DataFrame(results)
        os.makedirs('results', exist_ok=True)
        results_df.to_csv('results/fair_comparison_results.csv', index=False)
        
        # ========== GENERATE ALL VISUALIZATIONS ==========
        generate_execution_time_plot(results_df)
        generate_speedup_plot(results_df)
        generate_cluster_comparison_plot(results_df)
        generate_noise_comparison_plot(results_df)
        generate_processing_rate_plot(results_df)
        generate_scalability_analysis(results_df)
        generate_comprehensive_dashboard(results_df)
        generate_performance_table(results_df)
        
        print("\n" + "="*80)
        print("✅ All visualizations saved to results/ folder")
        print("="*80)
        print("\nGenerated files:")
        print("  1. execution_time_comparison.png")
        print("  2. speedup_factor.png")
        print("  3. cluster_comparison.png")
        print("  4. noise_comparison.png")
        print("  5. processing_rate.png")
        print("  6. scalability_analysis.png")
        print("  7. comprehensive_dashboard.png")
        print("  8. performance_table.png")
        print("  9. fair_comparison_results.csv")
        
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(results_df[['n_points', 'naive_time', 'optimized_time', 'speedup', 
                          'adjusted_rand_index', 'naive_clusters', 'optimized_clusters']].to_string(index=False))
        print(f"\nBest Speedup: {results_df['speedup'].max():.1f}x at {results_df.loc[results_df['speedup'].idxmax(), 'n_points']:.0f} points")
        print(f"Avg Adjusted Rand Index: {results_df['adjusted_rand_index'].mean():.4f}")
        
        return results_df
    return None

def generate_execution_time_plot(results_df):
    """Execution time comparison plot"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(results_df['n_points'], results_df['naive_time'], 'o-', 
            label='Naive O(n²)', linewidth=2, markersize=8, color='#E63946')
    ax.plot(results_df['n_points'], results_df['optimized_time'], 's-', 
            label='Optimized O(n log n)', linewidth=2, markersize=8, color='#2E8B57')
    
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Execution Time: Naive vs Optimized DBSCAN', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add value labels for optimized times
    for _, row in results_df.iterrows():
        ax.annotate(f'{row["optimized_time"]:.2f}s', 
                   (row['n_points'], row['optimized_time']),
                   textcoords="offset points", xytext=(5, 5), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/execution_time_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_speedup_plot(results_df):
    """Speedup factor plot"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(results_df['n_points'], results_df['speedup'], 'd-', 
            color='#1E88E5', linewidth=2, markersize=8)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Baseline (1x)')
    ax.fill_between(results_df['n_points'], 1, results_df['speedup'], 
                     where=(results_df['speedup'] > 1), color='#90EE90', alpha=0.3)
    
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_ylabel('Speedup Factor', fontsize=12)
    ax.set_title(f'Spatial Indexing Speedup\n{results_df["speedup"].max():.1f}x at {results_df.loc[results_df["speedup"].idxmax(), "n_points"]:.0f} points', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Add value labels
    for _, row in results_df.iterrows():
        ax.annotate(f'{row["speedup"]:.1f}x', 
                   (row['n_points'], row['speedup']),
                   textcoords="offset points", xytext=(5, 5), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/speedup_factor.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_cluster_comparison_plot(results_df):
    """Cluster count comparison plot"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(results_df['n_points'], results_df['naive_clusters'], 'o-', 
            label='Naive DBSCAN', linewidth=2, markersize=8, color='#E63946')
    ax.plot(results_df['n_points'], results_df['optimized_clusters'], 's-', 
            label='Optimized DBSCAN', linewidth=2, markersize=8, color='#2E8B57')
    
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_ylabel('Number of Clusters', fontsize=12)
    ax.set_title('Clusters Found: Naive vs Optimized', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('results/cluster_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_noise_comparison_plot(results_df):
    """Noise ratio comparison plot"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(results_df['n_points'], results_df['naive_noise_pct'], 'o-', 
            label='Naive DBSCAN', linewidth=2, markersize=8, color='#E63946')
    ax.plot(results_df['n_points'], results_df['optimized_noise_pct'], 's-', 
            label='Optimized DBSCAN', linewidth=2, markersize=8, color='#2E8B57')
    
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_ylabel('Noise Percentage (%)', fontsize=12)
    ax.set_title('Noise Ratio: Naive vs Optimized', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('results/noise_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_processing_rate_plot(results_df):
    """Processing rate (points per second) comparison"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(results_df['n_points'], results_df['naive_pts_per_sec'], 'o-', 
            label='Naive O(n²)', linewidth=2, markersize=8, color='#E63946')
    ax.plot(results_df['n_points'], results_df['optimized_pts_per_sec'], 's-', 
            label='Optimized O(n log n)', linewidth=2, markersize=8, color='#2E8B57')
    
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_ylabel('Points Processed per Second', fontsize=12)
    ax.set_title('Processing Rate: Naive vs Optimized', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('results/processing_rate.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_scalability_analysis(results_df):
    """Scalability analysis with theoretical curves"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Actual results
    ax.loglog(results_df['n_points'], results_df['naive_time'], 'o-', 
              label='Naive (Actual)', linewidth=2, markersize=8, color='#E63946')
    ax.loglog(results_df['n_points'], results_df['optimized_time'], 's-', 
              label='Optimized (Actual)', linewidth=2, markersize=8, color='#2E8B57')
    
    # Theoretical O(n²) reference
    n_ref = results_df['n_points'].values
    t_ref = results_df['naive_time'].iloc[0] * (n_ref / results_df['n_points'].iloc[0])**2
    ax.loglog(n_ref, t_ref, '--', label='Theoretical O(n²)', alpha=0.5, color='gray')
    
    ax.set_xlabel('Number of Points (log scale)', fontsize=12)
    ax.set_ylabel('Time (seconds) (log scale)', fontsize=12)
    ax.set_title('Scalability Analysis: O(n²) vs O(n log n)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('results/scalability_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_comprehensive_dashboard(results_df):
    """Comprehensive dashboard with all metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. Time Comparison
    axes[0, 0].plot(results_df['n_points'], results_df['naive_time'], 'o-', 
                     label='Naive', linewidth=2, color='#E63946')
    axes[0, 0].plot(results_df['n_points'], results_df['optimized_time'], 's-', 
                     label='Optimized', linewidth=2, color='#2E8B57')
    axes[0, 0].set_xlabel('Points')
    axes[0, 0].set_ylabel('Time (s)')
    axes[0, 0].set_title('Execution Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')
    
    # 2. Speedup
    axes[0, 1].plot(results_df['n_points'], results_df['speedup'], 'd-', 
                     color='#1E88E5', linewidth=2, markersize=8)
    axes[0, 1].axhline(y=1, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Points')
    axes[0, 1].set_ylabel('Speedup')
    axes[0, 1].set_title(f'Speedup: {results_df["speedup"].max():.1f}x max')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log')
    
    # 3. Cluster Similarity (ARI)
    axes[0, 2].plot(results_df['n_points'], results_df['adjusted_rand_index'], 'o-', 
                     color='#2E8B57', linewidth=2, markersize=8)
    axes[0, 2].axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='Target (0.95)')
    axes[0, 2].set_xlabel('Points')
    axes[0, 2].set_ylabel('Adjusted Rand Index')
    axes[0, 2].set_title('Cluster Identity (1.0 = identical)')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_xscale('log')
    
    # 4. Clusters Found
    axes[1, 0].plot(results_df['n_points'], results_df['naive_clusters'], 'o-', 
                     label='Naive', linewidth=2, color='#E63946')
    axes[1, 0].plot(results_df['n_points'], results_df['optimized_clusters'], 's-', 
                     label='Optimized', linewidth=2, color='#2E8B57')
    axes[1, 0].set_xlabel('Points')
    axes[1, 0].set_ylabel('Clusters')
    axes[1, 0].set_title('Clusters Found')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log')
    
    # 5. Noise Ratio
    axes[1, 1].plot(results_df['n_points'], results_df['naive_noise_pct'], 'o-', 
                     label='Naive', linewidth=2, color='#E63946')
    axes[1, 1].plot(results_df['n_points'], results_df['optimized_noise_pct'], 's-', 
                     label='Optimized', linewidth=2, color='#2E8B57')
    axes[1, 1].set_xlabel('Points')
    axes[1, 1].set_ylabel('Noise (%)')
    axes[1, 1].set_title('Noise Ratio')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log')
    
    # 6. Processing Rate
    axes[1, 2].plot(results_df['n_points'], results_df['optimized_pts_per_sec'], 's-', 
                     label='Optimized', linewidth=2, color='#2E8B57')
    axes[1, 2].plot(results_df['n_points'], results_df['naive_pts_per_sec'], 'o-', 
                     label='Naive', linewidth=2, color='#E63946')
    axes[1, 2].set_xlabel('Points')
    axes[1, 2].set_ylabel('Points/sec')
    axes[1, 2].set_title('Processing Rate')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xscale('log')
    axes[1, 2].set_yscale('log')
    
    plt.suptitle('DBSCAN Optimization: Comprehensive Performance Dashboard', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/comprehensive_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_performance_table(results_df):
    """Generate a performance summary table as an image"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for _, row in results_df.iterrows():
        table_data.append([
            f"{int(row['n_points']):,}",
            f"{row['naive_time']:.2f}s",
            f"{row['optimized_time']:.2f}s",
            f"{row['speedup']:.1f}x",
            f"{row['adjusted_rand_index']:.4f}",
            f"{row['naive_clusters']}",
            f"{row['optimized_clusters']}",
            f"{row['naive_noise_pct']:.1f}%",
            f"{row['optimized_noise_pct']:.1f}%"
        ])
    
    columns = ['Points', 'Naive Time', 'Optimized Time', 'Speedup', 'ARI', 
               'Naive Clusters', 'Opt Clusters', 'Naive Noise', 'Opt Noise']
    
    table = ax.table(cellText=table_data, colLabels=columns, 
                     cellLoc='center', loc='center',
                     colColours=['#2E8B57'] * len(columns))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    ax.set_title('Performance Comparison Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('results/performance_table.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_comparison()