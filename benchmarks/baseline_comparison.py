"""
Baseline Comparison: Optimized DBSCAN vs Scikit-learn DBSCAN
"""
import sys
import os
# Add parent directory to path so we can import from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from load_uber_data import load_uber_2014_data, prepare_gps_data
from src.optimized_dbscan import OptimizedDBSCAN

def run_comparison():
    """Run comparison between optimized and baseline DBSCAN"""
    
    print("="*60)
    print("BASELINE COMPARISON: Optimized vs Sklearn DBSCAN")
    print("="*60)
    
    # Test with increasing dataset sizes (start smaller for faster testing)
    test_sizes = [10000, 50000, 100000]  # Removed 200k for now
    results = []
    
    for n_points in test_sizes:
        print(f"\n{'='*50}")
        print(f"Testing with {n_points:,} points")
        print(f"{'='*50}")
        
        # Load data
        print("Loading data...")
        df = load_uber_2014_data(sample_size=n_points, filter_nyc=True)
        
        if df is None:
            print(f"⚠️ Could not load {n_points} points, skipping...")
            continue
        
        X, timestamps, _ = prepare_gps_data(df)
        
        # Parameters (adjusted for NYC scale)
        eps_km = 0.5  # 500 meters
        eps_degrees = eps_km / 111  # Convert km to degrees for sklearn
        min_samples = 30
        
        # Test 1: Sklearn DBSCAN (Baseline)
        print("\n[1/2] Running Sklearn DBSCAN...")
        sklearn_dbscan = SklearnDBSCAN(
            eps=eps_degrees,
            min_samples=min_samples,
            metric='euclidean'
        )
        
        start_time = time.time()
        sklearn_labels = sklearn_dbscan.fit_predict(X)
        sklearn_time = time.time() - start_time
        
        sklearn_clusters = len(set(sklearn_labels)) - (1 if -1 in sklearn_labels else 0)
        
        print(f"  Time: {sklearn_time:.2f} seconds")
        print(f"  Clusters: {sklearn_clusters}")
        
        # Test 2: Optimized DBSCAN
        print("\n[2/2] Running Optimized DBSCAN...")
        opt_dbscan = OptimizedDBSCAN(
            eps=eps_km,
            min_samples=min_samples,
            algorithm='ball_tree'
        )
        
        start_time = time.time()
        opt_labels = opt_dbscan.fit_predict(X)
        opt_time = time.time() - start_time
        
        opt_clusters = opt_dbscan.n_clusters_
        
        print(f"  Time: {opt_time:.2f} seconds")
        print(f"  Clusters: {opt_clusters}")
        
        # Calculate speedup
        speedup = sklearn_time / opt_time if opt_time > 0 else 0
        
        print(f"\n🚀 Speedup: {speedup:.2f}x faster!")
        
        # Store results
        results.append({
            'n_points': n_points,
            'sklearn_time': sklearn_time,
            'optimized_time': opt_time,
            'speedup': speedup,
            'sklearn_clusters': sklearn_clusters,
            'optimized_clusters': opt_clusters
        })
    
    # Create results dataframe
    if results:
        results_df = pd.DataFrame(results)
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Save results
        results_df.to_csv('results/baseline_comparison.csv', index=False)
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(results_df.to_string(index=False))
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Time comparison
        axes[0].plot(results_df['n_points'], results_df['sklearn_time'], 
                     'o-', label='Sklearn DBSCAN', linewidth=2, markersize=8)
        axes[0].plot(results_df['n_points'], results_df['optimized_time'], 
                     's-', label='Optimized DBSCAN', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of Points')
        axes[0].set_ylabel('Time (seconds)')
        axes[0].set_title('Execution Time Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        
        # Speedup
        axes[1].plot(results_df['n_points'], results_df['speedup'], 
                     'd-', color='green', linewidth=2, markersize=8)
        axes[1].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Baseline (1x)')
        axes[1].set_xlabel('Number of Points')
        axes[1].set_ylabel('Speedup Factor')
        axes[1].set_title('Speedup vs Scikit-learn')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xscale('log')
        
        plt.tight_layout()
        plt.savefig('results/baseline_comparison.png', dpi=150)
        plt.show()
        
        print("\n✅ Results saved to:")
        print("   - results/baseline_comparison.csv")
        print("   - results/baseline_comparison.png")
    else:
        print("\n❌ No results collected. Check your data loading.")
    
    return results_df if results else None

if __name__ == "__main__":
    run_comparison()