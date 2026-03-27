"""
Main script for Uber hotspot analysis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from load_uber_data import load_uber_2014_data, prepare_gps_data
from src.optimized_dbscan import OptimizedDBSCAN

def main():
    print("="*60)
    print("NYC UBER HOTSPOT ANALYSIS")
    print("="*60)
    
    # Step 1: Load data with NYC filtering
    print("\n[1/4] Loading data...")
    
    # Increase after confirming it works
    SAMPLE_SIZE = 500000
    
    df = load_uber_2014_data(sample_size=SAMPLE_SIZE, filter_nyc=True)
    
    if df is None:
        print("❌ Could not load data. Check that CSV files are in 'data' folder.")
        return
    
    # Step 2: Prepare GPS data
    print("\n[2/4] Preparing GPS data...")
    X, timestamps, df_clean = prepare_gps_data(df)
    
    if len(X) == 0:
        print("❌ No valid GPS points after filtering!")
        return
    
    # Step 3: Run DBSCAN with adjusted parameters
    print("\n[3/4] Running DBSCAN clustering...")
    
    # Adjusted parameters for NYC scale
    # eps = 0.002 ≈ 220 meters (for smaller, detailed hotspots)
    # eps = 0.005 ≈ 550 meters (neighborhood level)
    # eps = 0.008 ≈ 900 meters (larger areas)
    
    dbscan = OptimizedDBSCAN(
        eps=0.005,      # ~550 meters - good for neighborhood hotspots
        min_samples=15,  # Minimum points to form a hotspot
        algorithm='kd_tree'
    )
    
    labels = dbscan.fit_predict(X)
    
    # Step 4: Analyze results
    print("\n[4/4] Analyzing results...")
    
    # Add labels to dataframe
    df_clean['cluster'] = labels
    
    # Count clusters (excluding noise)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    if n_clusters == 0:
        print("\n⚠️ No clusters found! Try adjusting parameters:")
        print("   - Decrease eps (e.g., 0.003) for smaller hotspots")
        print("   - Decrease min_samples (e.g., 10) to find more clusters")
        print("   - Or increase sample size for more data points")
        
        # Try with more sensitive parameters
        print("\nTrying with more sensitive parameters...")
        dbscan2 = OptimizedDBSCAN(
            eps=0.003,      # ~330 meters
            min_samples=10,  # Lower threshold
            algorithm='kd_tree'
        )
        labels = dbscan2.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        dbscan = dbscan2
        
        if n_clusters == 0:
            print("❌ Still no clusters. Please check your data.")
            return
    
    # Top hotspots
    cluster_counts = df_clean[df_clean['cluster'] != -1]['cluster'].value_counts()
    
    print("\n" + "="*60)
    print(f"TOP {min(10, len(cluster_counts))} HOTSPOTS")
    print("="*60)
    
    for i, (cluster_id, count) in enumerate(cluster_counts.head(10).items(), 1):
        cluster_points = X[labels == cluster_id]
        center_lat = np.mean(cluster_points[:, 0])
        center_lon = np.mean(cluster_points[:, 1])
        
        # Get peak hour if timestamps available
        if timestamps is not None:
            cluster_times = timestamps[labels == cluster_id]
            peak_hour = cluster_times.dt.hour.mode()[0] if len(cluster_times) > 0 else 'N/A'
            print(f"{i:2}. Hotspot {cluster_id:2}: {count:6,} points "
                  f"@ ({center_lat:.4f}, {center_lon:.4f}) "
                  f"Peak: {peak_hour:02d}:00")
        else:
            print(f"{i:2}. Hotspot {cluster_id:2}: {count:6,} points "
                  f"@ ({center_lat:.4f}, {center_lon:.4f})")
    
    # Create visualization
    print("\nCreating visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: All clusters
    scatter = ax1.scatter(X[:, 1], X[:, 0], c=labels, cmap='tab20', 
                         s=5, alpha=0.6, rasterized=True)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title(f'NYC Uber Pickup Hotspots - {n_clusters} Clusters Found')
    ax1.grid(True, alpha=0.3)
    
    # Add NYC landmarks
    landmarks = {
        'Times Sq': (40.7580, -73.9855),
        'Central Pk': (40.7829, -73.9654),
        'Empire St': (40.7488, -73.9857),
        'JFK': (40.6413, -73.7781),
        'LGA': (40.7769, -73.8740),
        'Penn Station': (40.7505, -73.9934)
    }
    
    for name, (lat, lon) in landmarks.items():
        ax1.scatter(lon, lat, c='red', s=50, marker='x', linewidths=2)
        ax1.annotate(name, (lon, lat), fontsize=8, alpha=0.7)
    
    # Set NYC bounds
    ax1.set_xlim(-74.05, -73.85)
    ax1.set_ylim(40.6, 40.9)
    
    # Plot 2: Clusters vs Noise
    noise = labels == -1
    ax2.scatter(X[~noise, 1], X[~noise, 0], c='blue', s=5, alpha=0.6, 
                label=f'Hotspots ({np.sum(~noise):,})')
    ax2.scatter(X[noise, 1], X[noise, 0], c='gray', s=2, alpha=0.3, 
                label=f'Noise ({np.sum(noise):,})')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_title('Hotspots vs Scattered Pickups')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-74.05, -73.85)
    ax2.set_ylim(40.6, 40.9)
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'results/hotspots_{timestamp}.png', dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to results/hotspots_{timestamp}.png")
    
    # Save results
    results_file = f'results/cluster_results_{timestamp}.csv'
    df_clean.to_csv(results_file, index=False)
    print(f"✓ Results saved to {results_file}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total points processed: {len(X):,}")
    print(f"Hotspots found: {n_clusters}")
    print(f"Points in hotspots: {np.sum(~noise):,} ({np.sum(~noise)/len(X)*100:.1f}%)")
    print(f"Noise points: {np.sum(noise):,} ({np.sum(noise)/len(X)*100:.1f}%)")
    if n_clusters > 0:
        print(f"Average hotspot size: {np.sum(~noise)/n_clusters:.0f} points")
    
    print("\n✨ Analysis complete!")

if __name__ == "__main__":
    main()