"""
FAIR COMPARISON: Naive DBSCAN vs Optimized DBSCAN vs Sklearn DBSCAN
Rigorous benchmarking with multiple runs, standard deviations, and proper validation
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from load_uber_data import load_uber_2014_data, prepare_gps_data
from src.naive_dbscan import NaiveDBSCAN
from src.optimized_dbscan import OptimizedDBSCAN

# Set style for professional visualizations (neutral colors)
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

# Neutral color palette
COLORS = {
    'naive': '#E63946',
    'optimized': '#2E8B57',
    'sklearn': '#1E88E5',
    'grid': '#CCCCCC'
}

def run_comparison():
    print("="*80)
    print("FAIR COMPARISON: Naive vs Optimized vs Sklearn DBSCAN")
    print("Multiple runs per test size with standard deviation")
    print("="*80)
    
    test_sizes = [1000, 2000, 5000, 10000, 20000, 50000]
    n_runs = 3  # Multiple runs for robustness
    
    results = []
    
    for n_points in test_sizes:
        print(f"\n{'='*60}")
        print(f"Testing with {n_points:,} points ({n_runs} runs each)")
        print(f"{'='*60}")
        
        print("Loading data...")
        df = load_uber_2014_data(sample_size=n_points, filter_nyc=True)
        
        if df is None:
            continue
        
        X, timestamps, _ = prepare_gps_data(df)
        
        eps_km = 0.5
        eps_degrees = eps_km / 111.0
        min_samples = 30
        
        print(f"\nParameters: eps={eps_km}km, min_samples={min_samples}, Points={len(X):,}")
        
        # ========== SKLEARN BASELINE (Ground Truth) ==========
        print("\n[1/3] Running Sklearn DBSCAN (Ground Truth)...")
        sklearn_times = []
        sklearn_labels = None
        
        for run in range(n_runs):
            sklearn_dbscan = SklearnDBSCAN(eps=eps_degrees, min_samples=min_samples)
            start = time.time()
            labels = sklearn_dbscan.fit_predict(X)
            sklearn_times.append(time.time() - start)
            if run == 0:
                sklearn_labels = labels
        
        sklearn_mean = np.mean(sklearn_times)
        sklearn_std = np.std(sklearn_times)
        sklearn_clusters = len(set(sklearn_labels)) - (1 if -1 in sklearn_labels else 0)
        sklearn_noise = np.sum(sklearn_labels == -1)
        
        print(f"  Time: {sklearn_mean:.2f}s ± {sklearn_std:.2f}s | Clusters: {sklearn_clusters} | Noise: {sklearn_noise/len(X)*100:.1f}%")
        
        # ========== NAIVE DBSCAN ==========
        print("\n[2/3] Running Naive DBSCAN (Vectorized O(n²))...")
        naive_times = []
        naive_labels = None
        
        for run in range(n_runs):
            naive_dbscan = NaiveDBSCAN(eps=eps_km, min_samples=min_samples)
            start = time.time()
            labels = naive_dbscan.fit_predict(X)
            naive_times.append(time.time() - start)
            if run == 0:
                naive_labels = labels
        
        naive_mean = np.mean(naive_times)
        naive_std = np.std(naive_times)
        naive_clusters = naive_dbscan.n_clusters_
        naive_noise = np.sum(naive_labels == -1)
        
        print(f"  Time: {naive_mean:.2f}s ± {naive_std:.2f}s | Clusters: {naive_clusters} | Noise: {naive_noise/len(X)*100:.1f}%")
        
        # ========== OPTIMIZED DBSCAN ==========
        print("\n[3/3] Running Optimized DBSCAN (Custom Grid Index)...")
        opt_times = []
        opt_labels = None
        
        for run in range(n_runs):
            opt_dbscan = OptimizedDBSCAN(eps=eps_km, min_samples=min_samples)
            start = time.time()
            labels = opt_dbscan.fit_predict(X)
            opt_times.append(time.time() - start)
            if run == 0:
                opt_labels = labels
        
        opt_mean = np.mean(opt_times)
        opt_std = np.std(opt_times)
        opt_clusters = opt_dbscan.n_clusters_
        opt_noise = np.sum(opt_labels == -1)
        
        print(f"  Time: {opt_mean:.2f}s ± {opt_std:.2f}s | Clusters: {opt_clusters} | Noise: {opt_noise/len(X)*100:.1f}%")
        
        # ========== VALIDATION METRICS ==========
        # Compare against sklearn (ground truth)
        ari_naive_vs_sklearn = adjusted_rand_score(sklearn_labels, naive_labels)
        ari_opt_vs_sklearn = adjusted_rand_score(sklearn_labels, opt_labels)
        
        # Silhouette score (excluding noise)
        try:
            mask = opt_labels != -1
            if np.sum(mask) > opt_clusters and opt_clusters > 1:
                from src.optimized_dbscan import OptimizedDBSCAN as OptTemp
                temp = OptTemp(eps=eps_km, min_samples=min_samples)
                X_proj = temp._project(X)
                silhouette = silhouette_score(X_proj[mask], opt_labels[mask])
            else:
                silhouette = 0.0
        except Exception as e:
            print(f"  Silhouette calculation failed: {e}")
            silhouette = 0.0
        
        speedup_vs_naive = naive_mean / opt_mean if opt_mean > 0 else 0
        speedup_vs_sklearn = sklearn_mean / opt_mean if opt_mean > 0 else 0
        
        print(f"\n🚀 SPEEDUP vs Naive: {speedup_vs_naive:.1f}x")
        print(f"🚀 SPEEDUP vs Sklearn: {speedup_vs_sklearn:.1f}x")
        print(f"📊 ARI (Naive vs Sklearn): {ari_naive_vs_sklearn:.4f}")
        print(f"📊 ARI (Optimized vs Sklearn): {ari_opt_vs_sklearn:.4f}")
        
        results.append({
            'n_points': len(X),
            'sklearn_time': sklearn_mean,
            'sklearn_std': sklearn_std,
            'naive_time': naive_mean,
            'naive_std': naive_std,
            'optimized_time': opt_mean,
            'optimized_std': opt_std,
            'speedup_vs_naive': speedup_vs_naive,
            'speedup_vs_sklearn': speedup_vs_sklearn,
            'ari_naive_vs_sklearn': ari_naive_vs_sklearn,
            'ari_opt_vs_sklearn': ari_opt_vs_sklearn,
            'silhouette_score': silhouette,
            'sklearn_clusters': sklearn_clusters,
            'naive_clusters': naive_clusters,
            'optimized_clusters': opt_clusters
        })
    
    if results:
        results_df = pd.DataFrame(results)
        os.makedirs('results', exist_ok=True)
        results_df.to_csv('results/rigorous_benchmark.csv', index=False)
        
        generate_benchmark_plots(results_df)
        
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(results_df[['n_points', 'sklearn_time', 'naive_time', 'optimized_time', 
                          'speedup_vs_naive', 'ari_opt_vs_sklearn']].to_string(index=False))
        print(f"\nBest Speedup vs Naive: {results_df['speedup_vs_naive'].max():.1f}x")
        print(f"Best Speedup vs Sklearn: {results_df['speedup_vs_sklearn'].max():.1f}x")
        print(f"Avg ARI (Optimized vs Sklearn): {results_df['ari_opt_vs_sklearn'].mean():.4f}")
        
        return results_df
    return None

def generate_benchmark_plots(results_df):
    """Generate benchmark plots with error bars"""
    
    # 1. Execution Time Comparison with Error Bars
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = results_df['n_points']
    
    ax.errorbar(x, results_df['sklearn_time'], yerr=results_df['sklearn_std'],
                label='Sklearn DBSCAN', linewidth=2, marker='o', capsize=5, color=COLORS['sklearn'])
    ax.errorbar(x, results_df['naive_time'], yerr=results_df['naive_std'],
                label='Naive O(n²)', linewidth=2, marker='s', capsize=5, color=COLORS['naive'])
    ax.errorbar(x, results_df['optimized_time'], yerr=results_df['optimized_std'],
                label='Optimized Grid Index', linewidth=2, marker='^', capsize=5, color=COLORS['optimized'])
    
    ax.set_xlabel('Number of Points')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Execution Time Comparison (with std deviation)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('results/benchmark_execution_time.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. ARI Validation (Full Range 0-1)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(x, results_df['ari_naive_vs_sklearn'], 'o-',
            label='Naive vs Sklearn', linewidth=2, markersize=8, color=COLORS['naive'])
    ax.plot(x, results_df['ari_opt_vs_sklearn'], 's-',
            label='Optimized vs Sklearn', linewidth=2, markersize=8, color=COLORS['optimized'])
    ax.axhline(y=0.99, color='gray', linestyle='--', alpha=0.7, label='Near Perfect (0.99)')
    
    ax.set_xlabel('Number of Points')
    ax.set_ylabel('Adjusted Rand Index')
    ax.set_title('Validation: Comparison with Sklearn (Ground Truth)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_ylim(0, 1.05)  # Full range 0-1, not compressed
    
    plt.tight_layout()
    plt.savefig('results/benchmark_ari_validation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Speedup Comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(x, results_df['speedup_vs_naive'], 'o-',
            label='vs Naive Baseline', linewidth=2, markersize=8, color=COLORS['optimized'])
    ax.plot(x, results_df['speedup_vs_sklearn'], 's-',
            label='vs Sklearn', linewidth=2, markersize=8, color=COLORS['sklearn'])
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline (1x)')
    
    ax.set_xlabel('Number of Points')
    ax.set_ylabel('Speedup Factor')
    ax.set_title('Speedup Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('results/benchmark_speedup.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_comparison()