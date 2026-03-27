"""
Fair Comparison: Optimized DBSCAN vs Scikit-learn DBSCAN
"""
import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from load_uber_data import load_uber_2014_data, prepare_gps_data
from src.optimized_dbscan import OptimizedDBSCAN

def run_fair_comparison():
    """Run fair comparison between implementations"""
    
    print("="*70)
    print("FAIR COMPARISON: Optimized DBSCAN vs Scikit-learn DBSCAN")
    print("="*70)
    
    # Test with increasing dataset sizes (start small, increase gradually)
    test_sizes = [10000, 50000, 100000]
    results = []
    
    for n_points in test_sizes:
        print(f"\n{'='*60}")
        print(f"Testing with {n_points:,} points")
        print(f"{'='*60}")
        
        # Load data
        print("Loading data...")
        df = load_uber_2014_data(sample_size=n_points, filter_nyc=True)
        
        if df is None:
            print(f"⚠️ Could not load {n_points} points, skipping...")
            continue
        
        X, timestamps, _ = prepare_gps_data(df)
        
        # Common parameters
        eps_km = 0.5  # 500 meters
        eps_degrees = eps_km / 111.0  # Convert to degrees
        min_samples = 30
        
        print(f"\nParameters:")
        print(f"  eps: {eps_km} km ({eps_degrees:.4f} degrees)")
        print(f"  min_samples: {min_samples}")
        
        # ========== TEST 1: SKLEARN DBSCAN ==========
        print("\n[1/2] Running Scikit-learn DBSCAN...")
        sklearn_dbscan = SklearnDBSCAN(
            eps=eps_degrees,
            min_samples=min_samples,
            metric='euclidean'
        )
        
        start_time = time.time()
        sklearn_labels = sklearn_dbscan.fit_predict(X)
        sklearn_time = time.time() - start_time
        
        sklearn_clusters = len(set(sklearn_labels)) - (1 if -1 in sklearn_labels else 0)
        sklearn_noise = np.sum(sklearn_labels == -1)
        
        print(f"  Time: {sklearn_time:.2f} seconds")
        print(f"  Clusters: {sklearn_clusters}")
        print(f"  Noise: {sklearn_noise} ({sklearn_noise/len(X)*100:.1f}%)")
        
        # ========== TEST 2: OPTIMIZED DBSCAN ==========
        print("\n[2/2] Running Optimized DBSCAN...")
        opt_dbscan = OptimizedDBSCAN(
            eps=eps_km,
            min_samples=min_samples,
            algorithm='kd_tree'
        )
        
        start_time = time.time()
        opt_labels = opt_dbscan.fit_predict(X)
        opt_time = time.time() - start_time
        
        opt_clusters = opt_dbscan.n_clusters_
        opt_noise = np.sum(opt_labels == -1)
        
        print(f"  Time: {opt_time:.2f} seconds")
        print(f"  Clusters: {opt_clusters}")
        print(f"  Noise: {opt_noise} ({opt_noise/len(X)*100:.1f}%)")
        
        # Calculate speedup
        speedup = sklearn_time / opt_time if opt_time > 0 else 0
        
        print(f"\n🚀 SPEEDUP: {speedup:.2f}x faster!")
        
        # Store results
        results.append({
            'n_points': len(X),
            'sklearn_time': sklearn_time,
            'optimized_time': opt_time,
            'speedup': speedup,
            'sklearn_clusters': sklearn_clusters,
            'optimized_clusters': opt_clusters,
            'sklearn_noise_pct': sklearn_noise/len(X)*100,
            'optimized_noise_pct': opt_noise/len(X)*100
        })
    
    # Create results dataframe
    if results:
        results_df = pd.DataFrame(results)
        
        # Save results
        os.makedirs('results', exist_ok=True)
        results_df.to_csv('results/fair_comparison.csv', index=False)
        
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print(results_df[['n_points', 'sklearn_time', 'optimized_time', 'speedup', 
                          'sklearn_clusters', 'optimized_clusters']].to_string(index=False))
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Time Comparison
        axes[0, 0].plot(results_df['n_points'], results_df['sklearn_time'], 
                        'o-', label='Sklearn DBSCAN', linewidth=2, markersize=8, color='red')
        axes[0, 0].plot(results_df['n_points'], results_df['optimized_time'], 
                        's-', label='Optimized DBSCAN', linewidth=2, markersize=8, color='green')
        axes[0, 0].set_xlabel('Number of Points')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].set_title('Execution Time Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Speedup Factor
        axes[0, 1].plot(results_df['n_points'], results_df['speedup'], 
                        'd-', color='blue', linewidth=2, markersize=8)
        axes[0, 1].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Baseline (1x)')
        axes[0, 1].set_xlabel('Number of Points')
        axes[0, 1].set_ylabel('Speedup Factor')
        axes[0, 1].set_title('Speedup vs Scikit-learn')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Clusters Found
        axes[1, 0].plot(results_df['n_points'], results_df['sklearn_clusters'], 
                        'o-', label='Sklearn DBSCAN', linewidth=2, markersize=8, color='red')
        axes[1, 0].plot(results_df['n_points'], results_df['optimized_clusters'], 
                        's-', label='Optimized DBSCAN', linewidth=2, markersize=8, color='green')
        axes[1, 0].set_xlabel('Number of Points')
        axes[1, 0].set_ylabel('Number of Clusters')
        axes[1, 0].set_title('Clusters Found Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Noise Percentage
        axes[1, 1].plot(results_df['n_points'], results_df['sklearn_noise_pct'], 
                        'o-', label='Sklearn DBSCAN', linewidth=2, markersize=8, color='red')
        axes[1, 1].plot(results_df['n_points'], results_df['optimized_noise_pct'], 
                        's-', label='Optimized DBSCAN', linewidth=2, markersize=8, color='green')
        axes[1, 1].set_xlabel('Number of Points')
        axes[1, 1].set_ylabel('Noise Percentage (%)')
        axes[1, 1].set_title('Noise Ratio Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/fair_comparison.png', dpi=150)
        plt.show()
        
        print("\n✅ Results saved to:")
        print("   - results/fair_comparison.csv")
        print("   - results/fair_comparison.png")
        
        return results_df
    else:
        print("\n❌ No results collected.")
        return None

if __name__ == "__main__":
    run_fair_comparison()