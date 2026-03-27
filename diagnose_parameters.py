"""
Diagnostic script to find optimal DBSCAN parameters
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from load_uber_data import load_uber_2014_data, prepare_gps_data

def find_optimal_eps(X, k=5):
    """
    Find optimal eps using k-distance graph
    """
    # Compute distances to k-nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(X)
    distances, indices = neighbors.kneighbors(X)
    
    # Get the k-th nearest neighbor distances
    k_distances = distances[:, -1]
    k_distances_sorted = np.sort(k_distances)
    
    # Create elbow plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_distances_sorted)
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{k}-th Nearest Neighbor Distance')
    plt.title(f'k-Distance Graph (k={k})')
    plt.grid(True, alpha=0.3)
    
    # Find the elbow point
    # Look for where the curve starts increasing rapidly
    diff = np.diff(k_distances_sorted)
    elbow_idx = np.argmax(diff[:len(diff)//2])  # First major increase
    suggested_eps = k_distances_sorted[elbow_idx]
    
    plt.axhline(y=suggested_eps, color='r', linestyle='--', 
                label=f'Suggested eps = {suggested_eps:.4f}')
    plt.legend()
    
    # Plot histogram of distances
    plt.subplot(1, 2, 2)
    plt.hist(k_distances, bins=50, edgecolor='black')
    plt.xlabel(f'{k}-th Nearest Neighbor Distance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Nearest Neighbor Distances')
    plt.axvline(x=suggested_eps, color='r', linestyle='--', 
                label=f'eps = {suggested_eps:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/optimal_eps_analysis.png', dpi=150)
    plt.show()
    
    print(f"\n📊 Suggested eps (k={k}): {suggested_eps:.4f} degrees")
    print(f"   ≈ {suggested_eps * 111:.1f} km (1 degree ≈ 111 km)")
    
    return suggested_eps

def analyze_data_density(X):
    """Analyze data density"""
    print("\n" + "="*60)
    print("DATA DENSITY ANALYSIS")
    print("="*60)
    
    # Calculate overall area
    lat_range = X[:, 0].max() - X[:, 0].min()
    lon_range = X[:, 1].max() - X[:, 1].min()
    area_deg2 = lat_range * lon_range
    area_km2 = area_deg2 * (111 * 111)  # 1 deg ≈ 111 km
    
    density = len(X) / area_km2
    
    print(f"Number of points: {len(X):,}")
    print(f"Spatial extent:")
    print(f"  Latitude: {X[:, 0].min():.4f} to {X[:, 0].max():.4f} ({lat_range:.4f} deg)")
    print(f"  Longitude: {X[:, 1].min():.4f} to {X[:, 1].max():.4f} ({lon_range:.4f} deg)")
    print(f"  Area: {area_km2:.1f} km²")
    print(f"  Density: {density:.1f} points/km²")
    
    # Calculate average distance between points
    from scipy.spatial.distance import pdist
    sample_size = min(10000, len(X))
    sample_indices = np.random.choice(len(X), sample_size, replace=False)
    distances = pdist(X[sample_indices])
    avg_distance = np.mean(distances)
    median_distance = np.median(distances)
    
    print(f"\nDistance statistics (based on {sample_size} points):")
    print(f"  Average distance between points: {avg_distance:.6f} deg ({avg_distance*111:.2f} km)")
    print(f"  Median distance: {median_distance:.6f} deg ({median_distance*111:.2f} km)")
    
    return density, avg_distance

def main():
    print("="*60)
    print("DBSCAN PARAMETER DIAGNOSTIC")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df = load_uber_2014_data(sample_size=50000, filter_nyc=True)
    
    if df is None:
        print("❌ Could not load data")
        return
    
    X, timestamps, _ = prepare_gps_data(df)
    
    # Analyze data density
    density, avg_distance = analyze_data_density(X)
    
    # Find optimal eps
    print("\n" + "="*60)
    print("FINDING OPTIMAL EPS")
    print("="*60)
    
    # Try different k values
    for k in [5, 10, 15]:
        print(f"\n--- Testing k={k} ---")
        suggested_eps = find_optimal_eps(X, k=k)
        
        # Test with suggested eps
        print(f"\nTesting with eps={suggested_eps:.4f}, min_samples={k}")
        from src.optimized_dbscan import OptimizedDBSCAN
        
        dbscan = OptimizedDBSCAN(
            eps=suggested_eps,
            min_samples=k,
            algorithm='kd_tree'
        )
        labels = dbscan.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"Result: {n_clusters} clusters found")
        
        if n_clusters > 0:
            print(f"\n✅ Found clusters! Recommended parameters:")
            print(f"   eps = {suggested_eps:.4f}")
            print(f"   min_samples = {k}")
            break
    else:
        print("\n⚠️ Still no clusters. Try increasing sample size.")
        print("   Run with SAMPLE_SIZE = 200000 or more")

if __name__ == "__main__":
    main()