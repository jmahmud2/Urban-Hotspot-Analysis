"""
FAIR COMPARISON: Naive DBSCAN (O(n²)) vs Optimized DBSCAN (O(n log n))
Comprehensive analysis with multiple visualizations
"""
import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime
from load_uber_data import load_uber_2014_data, prepare_gps_data
from src.naive_dbscan import NaiveDBSCAN
from src.optimized_dbscan import OptimizedDBSCAN

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# NYC landmarks with their actual locations for cluster naming
LANDMARKS = {
    'Times Square': (40.7580, -73.9855, 'red'),
    'Penn Station': (40.7505, -73.9934, 'orange'),
    'Grand Central': (40.7527, -73.9772, 'orange'),
    'JFK Airport': (40.6413, -73.7781, 'purple'),
    'LaGuardia Airport': (40.7769, -73.8740, 'purple'),
    'Central Park': (40.7829, -73.9654, 'green'),
    'Wall Street': (40.7070, -74.0110, 'blue'),
    'Brooklyn Bridge': (40.7061, -73.9969, 'blue'),
    'Downtown Brooklyn': (40.6920, -73.9900, 'brown'),
    'Williamsburg': (40.7140, -73.9610, 'pink'),
    'Upper East Side': (40.7710, -73.9570, 'cyan'),
    'Upper West Side': (40.7870, -73.9750, 'cyan'),
    'Chelsea': (40.7460, -74.0010, 'olive'),
    'SoHo': (40.7230, -74.0000, 'gold'),
    'East Village': (40.7260, -73.9840, 'coral'),
    'Financial District': (40.7075, -74.0113, 'darkblue'),
    'Herald Square': (40.7495, -73.9877, 'darkgreen'),
    'Union Square': (40.7356, -73.9905, 'teal'),
    'Columbus Circle': (40.7681, -73.9822, 'darkorange')
}

# Borough boundaries with precise polygons (simplified as bounding boxes)
BOROUGHS = {
    'Manhattan': {'lat_min': 40.7, 'lat_max': 40.88, 'lon_min': -74.02, 'lon_max': -73.93},
    'Brooklyn': {'lat_min': 40.57, 'lat_max': 40.74, 'lon_min': -74.05, 'lon_max': -73.83},
    'Queens': {'lat_min': 40.55, 'lat_max': 40.80, 'lon_min': -73.96, 'lon_max': -73.70},
    'Bronx': {'lat_min': 40.79, 'lat_max': 40.88, 'lon_min': -73.93, 'lon_max': -73.77},
    'Staten Island': {'lat_min': 40.50, 'lat_max': 40.65, 'lon_min': -74.26, 'lon_max': -74.05}
}

def get_borough(lat, lon):
    """Determine borough based on coordinates"""
    for borough, bounds in BOROUGHS.items():
        if (bounds['lat_min'] <= lat <= bounds['lat_max'] and 
            bounds['lon_min'] <= lon <= bounds['lon_max']):
            return borough
    return 'Other/Water'

def get_cluster_name(centroid_lat, centroid_lon):
    """Assign a meaningful name to a cluster based on nearest landmark"""
    min_distance = float('inf')
    cluster_name = 'Unknown Area'
    
    for name, (lat, lon, _) in LANDMARKS.items():
        # Calculate Euclidean distance in degrees (simplified)
        distance = np.sqrt((centroid_lat - lat)**2 + (centroid_lon - lon)**2)
        if distance < min_distance:
            min_distance = distance
            cluster_name = name
    
    return cluster_name

def run_comparison():
    print("="*80)
    print("COMPREHENSIVE ANALYSIS: Naive vs Optimized DBSCAN")
    print("Progressive testing: 1,000 → 10,000 points")
    print("="*80)
    
    test_sizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    results = []
    all_cluster_data = {'naive': [], 'optimized': []}
    
    for n_points in test_sizes:
        print(f"\n{'='*60}")
        print(f"Testing with {n_points:,} points")
        print(f"{'='*60}")
        
        print("Loading data...")
        df = load_uber_2014_data(sample_size=n_points, filter_nyc=True)
        
        if df is None:
            continue
        
        X, timestamps, df_clean = prepare_gps_data(df)
        
        eps_km = 0.5
        min_samples = 30
        
        print(f"\nParameters: eps={eps_km}km, min_samples={min_samples}, Points={len(X):,}")
        
        # Naive DBSCAN
        print("\n[1/2] Running Naive DBSCAN (O(n²))...")
        naive_dbscan = NaiveDBSCAN(eps=eps_km, min_samples=min_samples)
        start = time.time()
        naive_labels = naive_dbscan.fit_predict(X)
        naive_time = time.time() - start
        naive_clusters = naive_dbscan.n_clusters_
        naive_noise = np.sum(naive_labels == -1)
        
        print(f"  Time: {naive_time:.2f}s | Clusters: {naive_clusters} | Noise: {naive_noise/len(X)*100:.1f}%")
        
        # Optimized DBSCAN
        print("\n[2/2] Running Optimized DBSCAN (BallTree + Haversine)...")
        opt_dbscan = OptimizedDBSCAN(eps=eps_km, min_samples=min_samples)
        start = time.time()
        opt_labels = opt_dbscan.fit_predict(X)
        opt_time = time.time() - start
        opt_clusters = opt_dbscan.n_clusters_
        opt_noise = np.sum(opt_labels == -1)
        
        print(f"  Time: {opt_time:.2f}s | Clusters: {opt_clusters} | Noise: {opt_noise/len(X)*100:.1f}%")
        
        speedup = naive_time / opt_time
        print(f"\n🚀 SPEEDUP: {speedup:.1f}x faster!")
        
        results.append({
            'n_points': len(X),
            'naive_time': naive_time,
            'optimized_time': opt_time,
            'speedup': speedup,
            'naive_clusters': naive_clusters,
            'optimized_clusters': opt_clusters,
            'naive_noise_pct': naive_noise/len(X)*100,
            'optimized_noise_pct': opt_noise/len(X)*100,
            'timestamps': timestamps
        })
        
        # Store cluster data for later analysis
        all_cluster_data['naive'].append((len(X), naive_labels, X))
        all_cluster_data['optimized'].append((len(X), opt_labels, X))
    
    if results:
        results_df = pd.DataFrame(results)
        os.makedirs('results', exist_ok=True)
        results_df.to_csv('results/complete_analysis.csv', index=False)
        
        # Generate all visualizations
        generate_performance_plots(results_df)
        generate_cluster_maps(all_cluster_data)
        generate_temporal_analysis(results_df)
        generate_parameter_sensitivity(all_cluster_data)
        generate_hotspot_ranking(all_cluster_data)
        generate_borough_distribution(all_cluster_data)
        generate_comprehensive_dashboard(results_df)
        
        print("\n" + "="*80)
        print("✅ All visualizations saved to results/ folder")
        print("="*80)
        print("\nGenerated files:")
        print("  1. execution_time_comparison.png")
        print("  2. speedup_factor.png")
        print("  3. clusters_comparison.png")
        print("  4. noise_ratio_comparison.png")
        print("  5. scalability_analysis.png")
        print("  6. cluster_maps/ - All cluster maps (10 maps each for naive and optimized)")
        print("  7. temporal_patterns.png")
        print("  8. parameter_sensitivity.png")
        print("  9. hotspot_ranking.png")
        print("  10. borough_distribution.png")
        print("  11. comprehensive_dashboard.png")
        
        return results_df
    return None

def generate_performance_plots(results_df):
    """Generate performance comparison plots"""
    
    # 1. Execution Time
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
    plt.tight_layout()
    plt.savefig('results/execution_time_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Speedup Factor
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
    plt.tight_layout()
    plt.savefig('results/speedup_factor.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Clusters Found
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(results_df['n_points'], results_df['naive_clusters'], 'o-', 
            label='Naive (Euclidean)', linewidth=2, markersize=8, color='#E63946')
    ax.plot(results_df['n_points'], results_df['optimized_clusters'], 's-', 
            label='Optimized (Haversine)', linewidth=2, markersize=8, color='#2E8B57')
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_ylabel('Number of Clusters', fontsize=12)
    ax.set_title('Clusters Found: Euclidean vs Haversine Distance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/clusters_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Noise Ratio
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(results_df['n_points'], results_df['naive_noise_pct'], 'o-', 
            label='Naive', linewidth=2, markersize=8, color='#E63946')
    ax.plot(results_df['n_points'], results_df['optimized_noise_pct'], 's-', 
            label='Optimized', linewidth=2, markersize=8, color='#2E8B57')
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_ylabel('Noise Percentage (%)', fontsize=12)
    ax.set_title('Noise Ratio: Naive vs Optimized', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/noise_ratio_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Scalability (Log-Log)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.loglog(results_df['n_points'], results_df['naive_time'], 'o-', 
              label='Naive O(n²)', linewidth=2, markersize=8, color='#E63946')
    ax.loglog(results_df['n_points'], results_df['optimized_time'], 's-', 
              label='Optimized O(n log n)', linewidth=2, markersize=8, color='#2E8B57')
    
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

def generate_cluster_maps(all_cluster_data):
    """Generate cluster maps for each dataset size for both methods with contrasting colors"""
    
    # Create subfolder for cluster maps
    os.makedirs('results/cluster_maps', exist_ok=True)
    
    # Use a high-contrast colormap
    colors = plt.cm.Set3(np.linspace(0, 1, 30))  # 30 distinct colors
    
    for n_points, labels, X in all_cluster_data['optimized']:
        if n_points in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
            fig, ax = plt.subplots(figsize=(14, 12))
            
            unique_labels = np.unique(labels)
            # Separate noise from clusters
            cluster_labels = unique_labels[unique_labels != -1]
            
            # Plot clusters with distinct colors
            for i, label in enumerate(cluster_labels):
                cluster_points = X[labels == label]
                color_idx = i % len(colors)
                ax.scatter(cluster_points[:, 1], cluster_points[:, 0], 
                          c=[colors[color_idx]], s=15, alpha=0.7, 
                          label=f'Hotspot {label}', edgecolors='white', linewidth=0.3)
            
            # Plot noise in gray
            noise_points = X[labels == -1]
            if len(noise_points) > 0:
                ax.scatter(noise_points[:, 1], noise_points[:, 0], 
                          c='lightgray', s=5, alpha=0.3, label='Noise')
            
            # Add landmarks
            for name, (lat, lon, color) in LANDMARKS.items():
                ax.scatter(lon, lat, c=color, s=120, marker='X', 
                          edgecolors='black', linewidths=1, zorder=5)
                ax.annotate(name, (lon, lat), fontsize=8, alpha=0.8,
                           xytext=(5, 5), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
            
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
            ax.set_title(f'Optimized DBSCAN - NYC Hotspots ({n_points:,} points)\n{len(cluster_labels)} Clusters Found', 
                         fontsize=14, fontweight='bold')
            ax.set_xlim(-74.05, -73.85)
            ax.set_ylim(40.6, 40.9)
            ax.grid(True, alpha=0.2)
            
            # Add legend (limited to top 15 to avoid clutter)
            handles = []
            for i, label in enumerate(cluster_labels[:15]):
                handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                          markerfacecolor=colors[i % len(colors)], 
                                          markersize=8, label=f'Hotspot {label}'))
            handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor='lightgray', markersize=8, label='Noise'))
            ax.legend(handles=handles, loc='upper right', fontsize=8, ncol=2)
            
            plt.tight_layout()
            plt.savefig(f'results/cluster_maps/optimized_{n_points}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved cluster map for optimized at {n_points} points")
    
    # Also generate naive cluster maps for comparison
    for n_points, labels, X in all_cluster_data['naive']:
        if n_points in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
            fig, ax = plt.subplots(figsize=(14, 12))
            
            unique_labels = np.unique(labels)
            cluster_labels = unique_labels[unique_labels != -1]
            
            for i, label in enumerate(cluster_labels):
                cluster_points = X[labels == label]
                color_idx = i % len(colors)
                ax.scatter(cluster_points[:, 1], cluster_points[:, 0], 
                          c=[colors[color_idx]], s=15, alpha=0.7, 
                          label=f'Cluster {label}', edgecolors='white', linewidth=0.3)
            
            noise_points = X[labels == -1]
            if len(noise_points) > 0:
                ax.scatter(noise_points[:, 1], noise_points[:, 0], 
                          c='lightgray', s=5, alpha=0.3, label='Noise')
            
            for name, (lat, lon, color) in LANDMARKS.items():
                ax.scatter(lon, lat, c=color, s=120, marker='X', 
                          edgecolors='black', linewidths=1, zorder=5)
                ax.annotate(name, (lon, lat), fontsize=8, alpha=0.8,
                           xytext=(5, 5), textcoords='offset points')
            
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
            ax.set_title(f'Naive DBSCAN (Euclidean) - NYC Hotspots ({n_points:,} points)\n{len(cluster_labels)} Clusters Found', 
                         fontsize=14, fontweight='bold')
            ax.set_xlim(-74.05, -73.85)
            ax.set_ylim(40.6, 40.9)
            ax.grid(True, alpha=0.2)
            
            handles = []
            for i, label in enumerate(cluster_labels[:15]):
                handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                          markerfacecolor=colors[i % len(colors)], 
                                          markersize=8, label=f'Cluster {label}'))
            handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor='lightgray', markersize=8, label='Noise'))
            ax.legend(handles=handles, loc='upper right', fontsize=8, ncol=2)
            
            plt.tight_layout()
            plt.savefig(f'results/cluster_maps/naive_{n_points}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved cluster map for naive at {n_points} points")

def generate_temporal_analysis(results_df):
    """Generate temporal pattern analysis"""
    
    # Use the last (largest) dataset for temporal analysis
    last_result = results_df.iloc[-1]
    timestamps = last_result.get('timestamps')
    
    if timestamps is not None and len(timestamps) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Hourly distribution
        hours = timestamps.dt.hour
        axes[0].hist(hours, bins=24, color='#2E8B57', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Hour of Day', fontsize=12)
        axes[0].set_ylabel('Number of Pickups', fontsize=12)
        axes[0].set_title('Hourly Pickup Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xticks(range(0, 24, 3))
        axes[0].grid(True, alpha=0.3)
        
        # Weekday distribution
        weekdays = timestamps.dt.dayofweek
        weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[1].hist(weekdays, bins=7, color='#1E88E5', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Day of Week', fontsize=12)
        axes[1].set_ylabel('Number of Pickups', fontsize=12)
        axes[1].set_title('Weekly Pickup Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xticks(range(7))
        axes[1].set_xticklabels(weekday_names)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/temporal_patterns.png', dpi=150, bbox_inches='tight')
        plt.close()

def generate_parameter_sensitivity(all_cluster_data):
    """Generate parameter sensitivity analysis for both methods in single graphs"""
    
    # Use the largest dataset (10,000 points) for parameter analysis
    opt_10k = None
    naive_10k = None
    
    for n_points, labels, X in all_cluster_data['optimized']:
        if n_points == 10000:
            opt_10k = X
            break
    
    for n_points, labels, X in all_cluster_data['naive']:
        if n_points == 10000:
            naive_10k = X
            break
    
    if opt_10k is None or naive_10k is None:
        print("⚠️ Could not find 10,000 point data for parameter sensitivity")
        return
    
    # Test different eps values (in km)
    eps_values = [0.3, 0.5, 0.6, 0.7, 0.8]
    min_samples_fixed = 30
    
    opt_clusters_by_eps = []
    naive_clusters_by_eps = []
    
    print("\n" + "="*60)
    print("Parameter Sensitivity Analysis")
    print("Testing different eps values on 10,000 points")
    print("="*60)
    
    for eps in eps_values:
        # Optimized DBSCAN
        opt_dbscan = OptimizedDBSCAN(eps=eps, min_samples=min_samples_fixed)
        opt_labels = opt_dbscan.fit_predict(opt_10k)
        opt_clusters = opt_dbscan.n_clusters_
        opt_clusters_by_eps.append(opt_clusters)
        
        # Naive DBSCAN
        naive_dbscan = NaiveDBSCAN(eps=eps, min_samples=min_samples_fixed)
        naive_labels = naive_dbscan.fit_predict(naive_10k)
        naive_clusters = naive_dbscan.n_clusters_
        naive_clusters_by_eps.append(naive_clusters)
        
        print(f"eps={eps}km -> Optimized: {opt_clusters} clusters, Naive: {naive_clusters} clusters")
    
    # Test different min_samples values
    min_samples_values = [10, 20, 30, 40, 50, 60, 70, 80]
    eps_fixed = 0.5
    
    opt_clusters_by_min = []
    naive_clusters_by_min = []
    
    print("\nTesting different min_samples values:")
    for min_s in min_samples_values:
        opt_dbscan = OptimizedDBSCAN(eps=eps_fixed, min_samples=min_s)
        opt_labels = opt_dbscan.fit_predict(opt_10k)
        opt_clusters = opt_dbscan.n_clusters_
        opt_clusters_by_min.append(opt_clusters)
        
        naive_dbscan = NaiveDBSCAN(eps=eps_fixed, min_samples=min_s)
        naive_labels = naive_dbscan.fit_predict(naive_10k)
        naive_clusters = naive_dbscan.n_clusters_
        naive_clusters_by_min.append(naive_clusters)
        
        print(f"min_samples={min_s} -> Optimized: {opt_clusters} clusters, Naive: {naive_clusters} clusters")
    
    # Create single figure with two subplots, each showing both methods
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Eps sensitivity - both methods in same graph
    ax1.plot(eps_values, opt_clusters_by_eps, 's-', 
             label='Optimized (Haversine)', linewidth=2, markersize=8, color='#2E8B57')
    ax1.plot(eps_values, naive_clusters_by_eps, 'o-', 
             label='Naive (Euclidean)', linewidth=2, markersize=8, color='#E63946')
    ax1.set_xlabel('Eps (km)', fontsize=12)
    ax1.set_ylabel('Number of Clusters', fontsize=12)
    ax1.set_title(f'Effect of eps on Clusters\n(min_samples={min_samples_fixed})', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (x, y1, y2) in enumerate(zip(eps_values, opt_clusters_by_eps, naive_clusters_by_eps)):
        ax1.annotate(f'{y1}', (x, y1), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, color='#2E8B57')
        ax1.annotate(f'{y2}', (x, y2), textcoords="offset points", xytext=(0, -12), ha='center', fontsize=9, color='#E63946')
    
    # Min samples sensitivity - both methods in same graph
    ax2.plot(min_samples_values, opt_clusters_by_min, 's-', 
             label='Optimized (Haversine)', linewidth=2, markersize=8, color='#2E8B57')
    ax2.plot(min_samples_values, naive_clusters_by_min, 'o-', 
             label='Naive (Euclidean)', linewidth=2, markersize=8, color='#E63946')
    ax2.set_xlabel('Min Samples', fontsize=12)
    ax2.set_ylabel('Number of Clusters', fontsize=12)
    ax2.set_title(f'Effect of min_samples on Clusters\n(eps={eps_fixed}km)', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (x, y1, y2) in enumerate(zip(min_samples_values, opt_clusters_by_min, naive_clusters_by_min)):
        ax2.annotate(f'{y1}', (x, y1), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, color='#2E8B57')
        ax2.annotate(f'{y2}', (x, y2), textcoords="offset points", xytext=(0, -12), ha='center', fontsize=9, color='#E63946')
    
    plt.tight_layout()
    plt.savefig('results/parameter_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Parameter sensitivity analysis saved (both methods in same graphs)")

def generate_hotspot_ranking(all_cluster_data):
    """Generate vertical bar chart hotspot ranking with cluster names based on actual locations"""
    
    # Use the largest dataset (10,000 points) for ranking
    opt_10k_labels = None
    opt_10k_X = None
    naive_10k_labels = None
    naive_10k_X = None
    
    for n_points, labels, X in all_cluster_data['optimized']:
        if n_points == 10000:
            opt_10k_labels = labels
            opt_10k_X = X
            break
    
    for n_points, labels, X in all_cluster_data['naive']:
        if n_points == 10000:
            naive_10k_labels = labels
            naive_10k_X = X
            break
    
    if opt_10k_labels is None or naive_10k_labels is None:
        print("⚠️ Could not find 10,000 point data for hotspot ranking")
        return
    
    # Calculate optimized cluster sizes and assign names
    opt_unique = np.unique(opt_10k_labels)
    opt_sizes = []
    for label in opt_unique:
        if label != -1:
            size = np.sum(opt_10k_labels == label)
            cluster_points = opt_10k_X[opt_10k_labels == label]
            centroid_lat = np.mean(cluster_points[:, 0])
            centroid_lon = np.mean(cluster_points[:, 1])
            cluster_name = get_cluster_name(centroid_lat, centroid_lon)
            opt_sizes.append((label, size, centroid_lat, centroid_lon, cluster_name))
    opt_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate naive cluster sizes and assign names
    naive_unique = np.unique(naive_10k_labels)
    naive_sizes = []
    for label in naive_unique:
        if label != -1:
            size = np.sum(naive_10k_labels == label)
            cluster_points = naive_10k_X[naive_10k_labels == label]
            centroid_lat = np.mean(cluster_points[:, 0])
            centroid_lon = np.mean(cluster_points[:, 1])
            cluster_name = get_cluster_name(centroid_lat, centroid_lon)
            naive_sizes.append((label, size, centroid_lat, centroid_lon, cluster_name))
    naive_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Create side-by-side vertical bar charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Optimized ranking (vertical bars)
    top_opt = opt_sizes[:10]
    opt_names = [c[4] for c in top_opt]  # Use actual cluster names
    opt_counts = [c[1] for c in top_opt]
    
    bars1 = ax1.bar(opt_names, opt_counts, color='#2E8B57', alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Hotspot Location', fontsize=12)
    ax1.set_ylabel('Number of Pickups', fontsize=12)
    ax1.set_title('Optimized DBSCAN (Haversine) - Top 10 Hotspots', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar, count in zip(bars1, opt_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Naive ranking (vertical bars)
    top_naive = naive_sizes[:10]
    naive_names = [c[4] for c in top_naive]  # Use actual cluster names
    naive_counts = [c[1] for c in top_naive]
    
    bars2 = ax2.bar(naive_names, naive_counts, color='#E63946', alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Cluster Location', fontsize=12)
    ax2.set_ylabel('Number of Pickups', fontsize=12)
    ax2.set_title('Naive DBSCAN (Euclidean) - Top 10 Clusters', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar, count in zip(bars2, naive_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/hotspot_ranking.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Hotspot ranking saved (Optimized: {len(top_opt)} hotspots, Naive: {len(top_naive)} clusters)")
    print("\nOptimized Top 10 Hotspots:")
    for i, (label, size, lat, lon, name) in enumerate(top_opt[:10], 1):
        print(f"  {i}. {name}: {size:,} pickups @ ({lat:.4f}, {lon:.4f})")

def generate_borough_distribution(all_cluster_data):
    """Generate borough distribution for both methods with integer counts"""
    
    # Use the largest dataset (10,000 points)
    opt_10k_labels = None
    opt_10k_X = None
    naive_10k_labels = None
    naive_10k_X = None
    
    for n_points, labels, X in all_cluster_data['optimized']:
        if n_points == 10000:
            opt_10k_labels = labels
            opt_10k_X = X
            break
    
    for n_points, labels, X in all_cluster_data['naive']:
        if n_points == 10000:
            naive_10k_labels = labels
            naive_10k_X = X
            break
    
    if opt_10k_labels is None or naive_10k_labels is None:
        print("⚠️ Could not find 10,000 point data for borough distribution")
        return
    
    # Calculate optimized borough distribution (counts are integers)
    opt_borough_counts = {borough: 0 for borough in BOROUGHS.keys()}
    opt_borough_counts['Other/Water'] = 0
    
    opt_unique = np.unique(opt_10k_labels)
    for label in opt_unique:
        if label != -1:
            cluster_points = opt_10k_X[opt_10k_labels == label]
            centroid_lat = np.mean(cluster_points[:, 0])
            centroid_lon = np.mean(cluster_points[:, 1])
            borough = get_borough(centroid_lat, centroid_lon)
            opt_borough_counts[borough] += 1
    
    # Calculate naive borough distribution (counts are integers)
    naive_borough_counts = {borough: 0 for borough in BOROUGHS.keys()}
    naive_borough_counts['Other/Water'] = 0
    
    naive_unique = np.unique(naive_10k_labels)
    for label in naive_unique:
        if label != -1:
            cluster_points = naive_10k_X[naive_10k_labels == label]
            centroid_lat = np.mean(cluster_points[:, 0])
            centroid_lon = np.mean(cluster_points[:, 1])
            borough = get_borough(centroid_lat, centroid_lon)
            naive_borough_counts[borough] += 1
    
    # Create side-by-side vertical bar charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    boroughs = list(BOROUGHS.keys())
    opt_counts = [opt_borough_counts[b] for b in boroughs]
    naive_counts = [naive_borough_counts[b] for b in boroughs]
    
    # Optimized distribution (vertical bars)
    bars1 = ax1.bar(boroughs, opt_counts, color='#2E8B57', edgecolor='black', alpha=0.8)
    ax1.set_xlabel('Borough', fontsize=12)
    ax1.set_ylabel('Number of Hotspots', fontsize=12)
    ax1.set_title('Optimized DBSCAN (Haversine) - Hotspot Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, count in zip(bars1, opt_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Naive distribution (vertical bars)
    bars2 = ax2.bar(boroughs, naive_counts, color='#E63946', edgecolor='black', alpha=0.8)
    ax2.set_xlabel('Borough', fontsize=12)
    ax2.set_ylabel('Number of Clusters', fontsize=12)
    ax2.set_title('Naive DBSCAN (Euclidean) - Cluster Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, count in zip(bars2, naive_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/borough_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Borough distribution saved")
    print(f"  Optimized: {opt_borough_counts}")
    print(f"  Naive: {naive_borough_counts}")

def generate_comprehensive_dashboard(results_df):
    """Generate a comprehensive dashboard with all metrics"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Time Comparison
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(results_df['n_points'], results_df['naive_time'], 'o-', 
             label='Naive', linewidth=2, markersize=6, color='#E63946')
    ax1.plot(results_df['n_points'], results_df['optimized_time'], 's-', 
             label='Optimized', linewidth=2, markersize=6, color='#2E8B57')
    ax1.set_xlabel('Points')
    ax1.set_ylabel('Time (s)')
    ax1.set_title('Execution Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Speedup
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(results_df['n_points'], results_df['speedup'], 'd-', 
             color='#1E88E5', linewidth=2, markersize=6)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Points')
    ax2.set_ylabel('Speedup')
    ax2.set_title(f'Speedup: {results_df["speedup"].max():.0f}x max')
    ax2.grid(True, alpha=0.3)
    
    # 3. Clusters
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(results_df['n_points'], results_df['naive_clusters'], 'o-', 
             label='Naive', linewidth=2, markersize=6, color='#E63946')
    ax3.plot(results_df['n_points'], results_df['optimized_clusters'], 's-', 
             label='Optimized', linewidth=2, markersize=6, color='#2E8B57')
    ax3.set_xlabel('Points')
    ax3.set_ylabel('Clusters')
    ax3.set_title('Clusters Found')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Noise Ratio
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(results_df['n_points'], results_df['naive_noise_pct'], 'o-', 
             label='Naive', linewidth=2, markersize=6, color='#E63946')
    ax4.plot(results_df['n_points'], results_df['optimized_noise_pct'], 's-', 
             label='Optimized', linewidth=2, markersize=6, color='#2E8B57')
    ax4.set_xlabel('Points')
    ax4.set_ylabel('Noise (%)')
    ax4.set_title('Noise Ratio')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Processing Rate
    ax5 = plt.subplot(2, 3, 5)
    naive_rate = results_df['n_points'] / results_df['naive_time']
    opt_rate = results_df['n_points'] / results_df['optimized_time']
    ax5.plot(results_df['n_points'], naive_rate, 'o-', 
             label='Naive', linewidth=2, markersize=6, color='#E63946')
    ax5.plot(results_df['n_points'], opt_rate, 's-', 
             label='Optimized', linewidth=2, markersize=6, color='#2E8B57')
    ax5.set_xlabel('Points')
    ax5.set_ylabel('Points/sec')
    ax5.set_title('Processing Rate')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary Stats
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    stats_text = f"""
    SUMMARY STATISTICS
    
    Best Speedup: {results_df['speedup'].max():.1f}x
    at {results_df.loc[results_df['speedup'].idxmax(), 'n_points']:.0f} points
    
    Optimized Time (10k): {results_df[results_df['n_points']==10000]['optimized_time'].values[0]:.2f}s
    Naive Time (10k): {results_df[results_df['n_points']==10000]['naive_time'].values[0]:.2f}s
    
    Avg Clusters (Optimized): {results_df['optimized_clusters'].mean():.1f}
    Avg Clusters (Naive): {results_df['naive_clusters'].mean():.1f}
    
    Avg Noise (Optimized): {results_df['optimized_noise_pct'].mean():.1f}%
    Avg Noise (Naive): {results_df['naive_noise_pct'].mean():.1f}%
    """
    ax6.text(0.1, 0.5, stats_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8))
    ax6.set_title('Performance Summary', fontsize=12, fontweight='bold')
    
    plt.suptitle('DBSCAN Optimization: Comprehensive Performance Dashboard', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/comprehensive_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_comparison()