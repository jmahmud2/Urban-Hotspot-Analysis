"""
FAIR COMPARISON: Naive DBSCAN (O(n²)) vs Optimized DBSCAN (O(n log n))
Comprehensive analysis with high-contrast colors
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

# High-contrast color palette - Naive brighter/prominent, Optimized darker/smoother
COLORS = {
    'naive': '#E63946',      # Bright Red - prominent, stands out
    'optimized': '#1F7A5A',   # Darker Teal - smoother, recedes
    'speedup': '#1E88E5',     # Bright Blue
    'accent': '#FFB347',      # Orange accent
    'grid': '#CCCCCC',        # Light Gray
    'text': '#2C3E50'         # Dark blue-gray
}

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
    """Execution time comparison plot - Naive prominent, Optimized smoother"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Naive: Bright Red, thicker line, larger markers - stands out
    ax.plot(results_df['n_points'], results_df['naive_time'], 'o-', 
            label='Naive O(n²)', linewidth=2.5, markersize=9, 
            color=COLORS['naive'], markerfacecolor=COLORS['naive'], 
            markeredgewidth=1.5, markeredgecolor='white')
    
    # Optimized: Darker Teal, standard line, smaller markers - recedes
    ax.plot(results_df['n_points'], results_df['optimized_time'], 's-', 
            label='Optimized O(n log n)', linewidth=2, markersize=8,
            color=COLORS['optimized'], markerfacecolor=COLORS['optimized'],
            markeredgewidth=1, markeredgecolor='white', alpha=0.85)
    
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Execution Time: Naive vs Optimized DBSCAN', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add value labels for optimized times (smaller font)
    for _, row in results_df.iterrows():
        if row['optimized_time'] > 0:
            ax.annotate(f'{row["optimized_time"]:.2f}s', 
                       (row['n_points'], row['optimized_time']),
                       textcoords="offset points", xytext=(5, 5), 
                       ha='center', fontsize=8, color=COLORS['optimized'])
    
    plt.tight_layout()
    plt.savefig('results/execution_time_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_speedup_plot(results_df):
    """Speedup factor plot"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(results_df['n_points'], results_df['speedup'], 'd-', 
            color=COLORS['speedup'], linewidth=2.5, markersize=9,
            markerfacecolor=COLORS['speedup'], markeredgewidth=1.5, 
            markeredgecolor='white')
    
    ax.axhline(y=1, color=COLORS['naive'], linestyle='--', 
               linewidth=1.5, alpha=0.7, label='Baseline (1x)')
    
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_ylabel('Speedup Factor', fontsize=12)
    ax.set_title(f'Spatial Indexing Speedup\n{results_df["speedup"].max():.1f}x at {results_df.loc[results_df["speedup"].idxmax(), "n_points"]:.0f} points', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xscale('log')
    
    # Add value labels
    for _, row in results_df.iterrows():
        ax.annotate(f'{row["speedup"]:.1f}x', 
                   (row['n_points'], row['speedup']),
                   textcoords="offset points", xytext=(5, 8), 
                   ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/speedup_factor.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_cluster_comparison_plot(results_df):
    """Cluster count comparison - Naive prominent, Optimized smoother"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Filter out points where clusters = 0
    mask = results_df['naive_clusters'] > 0
    filtered_df = results_df[mask]
    
    if len(filtered_df) > 0:
        # Naive: Bright Red, larger markers, thicker line - stands out
        ax.plot(filtered_df['n_points'], filtered_df['naive_clusters'], 'o-', 
                label='Naive DBSCAN', linewidth=2.5, markersize=9,
                color=COLORS['naive'], markerfacecolor=COLORS['naive'],
                markeredgewidth=1.5, markeredgecolor='white')
        
        # Optimized: Darker Teal, smaller markers, smoother - recedes
        ax.plot(filtered_df['n_points'], filtered_df['optimized_clusters'], 's-', 
                label='Optimized DBSCAN', linewidth=2, markersize=8,
                color=COLORS['optimized'], markerfacecolor=COLORS['optimized'],
                markeredgewidth=1, markeredgecolor='white', alpha=0.85)
    
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_ylabel('Number of Clusters', fontsize=12)
    ax.set_title('Clusters Found: Naive vs Optimized DBSCAN', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xscale('log')
    
    # Add value labels - Naive labels prominent, Optimized smaller
    for _, row in filtered_df.iterrows():
        if row['naive_clusters'] > 0:
            ax.annotate(f'{row["naive_clusters"]}', 
                       (row['n_points'], row['naive_clusters']),
                       textcoords="offset points", xytext=(5, 8), 
                       ha='center', fontsize=10, fontweight='bold', color=COLORS['naive'])
            ax.annotate(f'{row["optimized_clusters"]}', 
                       (row['n_points'], row['optimized_clusters']),
                       textcoords="offset points", xytext=(5, -10), 
                       ha='center', fontsize=9, color=COLORS['optimized'])
    
    plt.tight_layout()
    plt.savefig('results/cluster_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_noise_comparison_plot(results_df):
    """Noise ratio comparison - Naive prominent, Optimized smoother"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Filter out points where clusters > 0 for meaningful comparison
    mask = results_df['naive_clusters'] > 0
    filtered_df = results_df[mask]
    
    if len(filtered_df) > 0:
        # Naive: Bright Red, larger markers - stands out
        ax.plot(filtered_df['n_points'], filtered_df['naive_noise_pct'], 'o-', 
                label='Naive DBSCAN', linewidth=2.5, markersize=9,
                color=COLORS['naive'], markerfacecolor=COLORS['naive'],
                markeredgewidth=1.5, markeredgecolor='white')
        
        # Optimized: Darker Teal, smaller markers - recedes
        ax.plot(filtered_df['n_points'], filtered_df['optimized_noise_pct'], 's-', 
                label='Optimized DBSCAN', linewidth=2, markersize=8,
                color=COLORS['optimized'], markerfacecolor=COLORS['optimized'],
                markeredgewidth=1, markeredgecolor='white', alpha=0.85)
    
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_ylabel('Noise Percentage (%)', fontsize=12)
    ax.set_title('Noise Ratio: Naive vs Optimized DBSCAN', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xscale('log')
    
    # Add value labels
    for _, row in filtered_df.iterrows():
        ax.annotate(f'{row["naive_noise_pct"]:.0f}%', 
                   (row['n_points'], row['naive_noise_pct']),
                   textcoords="offset points", xytext=(5, 8), 
                   ha='center', fontsize=9, fontweight='bold', color=COLORS['naive'])
        ax.annotate(f'{row["optimized_noise_pct"]:.0f}%', 
                   (row['n_points'], row['optimized_noise_pct']),
                   textcoords="offset points", xytext=(5, -10), 
                   ha='center', fontsize=8, color=COLORS['optimized'])
    
    plt.tight_layout()
    plt.savefig('results/noise_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_processing_rate_plot(results_df):
    """Processing rate comparison"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(results_df['n_points'], results_df['naive_pts_per_sec'], 'o-', 
            label='Naive O(n²)', linewidth=2.5, markersize=9,
            color=COLORS['naive'], markerfacecolor=COLORS['naive'],
            markeredgewidth=1.5, markeredgecolor='white')
    
    ax.plot(results_df['n_points'], results_df['optimized_pts_per_sec'], 's-', 
            label='Optimized O(n log n)', linewidth=2, markersize=8,
            color=COLORS['optimized'], markerfacecolor=COLORS['optimized'],
            markeredgewidth=1, markeredgecolor='white', alpha=0.85)
    
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_ylabel('Points Processed per Second', fontsize=12)
    ax.set_title('Processing Rate: Naive vs Optimized DBSCAN', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
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
              label='Naive (Actual)', linewidth=2.5, markersize=9,
              color=COLORS['naive'], markerfacecolor=COLORS['naive'],
              markeredgewidth=1.5, markeredgecolor='white')
    
    ax.loglog(results_df['n_points'], results_df['optimized_time'], 's-', 
              label='Optimized (Actual)', linewidth=2, markersize=8,
              color=COLORS['optimized'], markerfacecolor=COLORS['optimized'],
              markeredgewidth=1, markeredgecolor='white', alpha=0.85)
    
    # Theoretical O(n²) reference
    n_ref = results_df['n_points'].values
    t_ref = results_df['naive_time'].iloc[0] * (n_ref / results_df['n_points'].iloc[0])**2
    ax.loglog(n_ref, t_ref, '--', label='Theoretical O(n²)', 
              alpha=0.5, linewidth=1.5, color=COLORS['grid'])
    
    ax.set_xlabel('Number of Points (log scale)', fontsize=12)
    ax.set_ylabel('Time (seconds) (log scale)', fontsize=12)
    ax.set_title('Scalability Analysis: O(n²) vs O(n log n)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    
    plt.tight_layout()
    plt.savefig('results/scalability_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_comprehensive_dashboard(results_df):
    """Comprehensive dashboard with all metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. Time Comparison
    ax1 = axes[0, 0]
    ax1.plot(results_df['n_points'], results_df['naive_time'], 'o-', 
             label='Naive', linewidth=2, markersize=6, color=COLORS['naive'])
    ax1.plot(results_df['n_points'], results_df['optimized_time'], 's-', 
             label='Optimized', linewidth=2, markersize=5, color=COLORS['optimized'])
    ax1.set_xlabel('Points')
    ax1.set_ylabel('Time (s)')
    ax1.set_title('Execution Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # 2. Speedup
    ax2 = axes[0, 1]
    ax2.plot(results_df['n_points'], results_df['speedup'], 'd-', 
             color=COLORS['speedup'], linewidth=2, markersize=6)
    ax2.axhline(y=1, color=COLORS['naive'], linestyle='--', alpha=0.7)
    ax2.set_xlabel('Points')
    ax2.set_ylabel('Speedup')
    ax2.set_title(f'Speedup: {results_df["speedup"].max():.1f}x max')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # 3. Cluster Similarity
    ax3 = axes[0, 2]
    ax3.plot(results_df['n_points'], results_df['adjusted_rand_index'], 'o-', 
             color=COLORS['optimized'], linewidth=2, markersize=6)
    ax3.axhline(y=0.99, color=COLORS['naive'], linestyle='--', alpha=0.7)
    ax3.set_xlabel('Points')
    ax3.set_ylabel('Adjusted Rand Index')
    ax3.set_title('Cluster Identity (1.0 = identical)')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    ax3.set_ylim(0.99, 1.01)
    
    # 4. Clusters Found
    ax4 = axes[1, 0]
    mask = results_df['naive_clusters'] > 0
    filtered = results_df[mask]
    ax4.plot(filtered['n_points'], filtered['naive_clusters'], 'o-', 
             label='Naive', linewidth=2, markersize=6, color=COLORS['naive'])
    ax4.plot(filtered['n_points'], filtered['optimized_clusters'], 's-', 
             label='Optimized', linewidth=2, markersize=5, color=COLORS['optimized'])
    ax4.set_xlabel('Points')
    ax4.set_ylabel('Clusters')
    ax4.set_title('Clusters Found')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    
    # 5. Noise Ratio
    ax5 = axes[1, 1]
    ax5.plot(filtered['n_points'], filtered['naive_noise_pct'], 'o-', 
             label='Naive', linewidth=2, markersize=6, color=COLORS['naive'])
    ax5.plot(filtered['n_points'], filtered['optimized_noise_pct'], 's-', 
             label='Optimized', linewidth=2, markersize=5, color=COLORS['optimized'])
    ax5.set_xlabel('Points')
    ax5.set_ylabel('Noise (%)')
    ax5.set_title('Noise Ratio')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xscale('log')
    
    # 6. Processing Rate
    ax6 = axes[1, 2]
    ax6.plot(results_df['n_points'], results_df['optimized_pts_per_sec'], 's-', 
             label='Optimized', linewidth=2, markersize=5, color=COLORS['optimized'])
    ax6.plot(results_df['n_points'], results_df['naive_pts_per_sec'], 'o-', 
             label='Naive', linewidth=2, markersize=6, color=COLORS['naive'])
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
    fig, ax = plt.subplots(figsize=(14, 6))
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
                     colColours=[COLORS['optimized']] * len(columns))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
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