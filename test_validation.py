"""
Validation Test Script - Proper unit tests with assertions
"""
import numpy as np
from sklearn.metrics import adjusted_rand_score
from load_uber_data import load_uber_2014_data, prepare_gps_data
from src.naive_dbscan import NaiveDBSCAN
from src.optimized_dbscan import OptimizedDBSCAN
from sklearn.cluster import DBSCAN as SklearnDBSCAN

def test_cluster_consistency():
    """Test that all three implementations produce consistent clusters"""
    print("="*60)
    print("TEST: Cluster Consistency Across Implementations")
    print("="*60)
    
    # Load small dataset for testing
    df = load_uber_2014_data(sample_size=5000, filter_nyc=True)
    X, timestamps, _ = prepare_gps_data(df)
    
    eps_km = 0.5
    eps_degrees = eps_km / 111.0
    min_samples = 30
    
    print(f"\nTesting with {len(X):,} points")
    print(f"Parameters: eps={eps_km}km, min_samples={min_samples}")
    
    # Sklearn baseline
    print("\n1. Running Sklearn DBSCAN (Ground Truth)...")
    sklearn_dbscan = SklearnDBSCAN(eps=eps_degrees, min_samples=min_samples)
    sklearn_labels = sklearn_dbscan.fit_predict(X)
    sklearn_clusters = len(set(sklearn_labels)) - (1 if -1 in sklearn_labels else 0)
    print(f"   Clusters: {sklearn_clusters}")
    
    # Naive DBSCAN
    print("\n2. Running Naive DBSCAN...")
    naive_dbscan = NaiveDBSCAN(eps=eps_km, min_samples=min_samples)
    naive_labels = naive_dbscan.fit_predict(X)
    naive_clusters = naive_dbscan.n_clusters_
    print(f"   Clusters: {naive_clusters}")
    
    # Optimized DBSCAN
    print("\n3. Running Optimized DBSCAN...")
    opt_dbscan = OptimizedDBSCAN(eps=eps_km, min_samples=min_samples)
    opt_labels = opt_dbscan.fit_predict(X)
    opt_clusters = opt_dbscan.n_clusters_
    print(f"   Clusters: {opt_clusters}")
    
    # Assertions
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    
    ari_naive_vs_sklearn = adjusted_rand_score(sklearn_labels, naive_labels)
    ari_opt_vs_sklearn = adjusted_rand_score(sklearn_labels, opt_labels)
    
    print(f"ARI (Naive vs Sklearn): {ari_naive_vs_sklearn:.4f}")
    print(f"ARI (Optimized vs Sklearn): {ari_opt_vs_sklearn:.4f}")
    
    # Assert with threshold
    assert ari_naive_vs_sklearn > 0.95, f"Naive ARI too low: {ari_naive_vs_sklearn}"
    assert ari_opt_vs_sklearn > 0.95, f"Optimized ARI too low: {ari_opt_vs_sklearn}"
    
    print("\n✅ All validation tests passed!")

def test_cluster_counts_match():
    """Test that cluster counts match across implementations"""
    print("\n" + "="*60)
    print("TEST: Cluster Count Matching")
    print("="*60)
    
    df = load_uber_2014_data(sample_size=5000, filter_nyc=True)
    X, timestamps, _ = prepare_gps_data(df)
    
    eps_km = 0.5
    eps_degrees = eps_km / 111.0
    min_samples = 30
    
    sklearn_dbscan = SklearnDBSCAN(eps=eps_degrees, min_samples=min_samples)
    sklearn_labels = sklearn_dbscan.fit_predict(X)
    sklearn_clusters = len(set(sklearn_labels)) - (1 if -1 in sklearn_labels else 0)
    
    naive_dbscan = NaiveDBSCAN(eps=eps_km, min_samples=min_samples)
    naive_labels = naive_dbscan.fit_predict(X)
    naive_clusters = naive_dbscan.n_clusters_
    
    opt_dbscan = OptimizedDBSCAN(eps=eps_km, min_samples=min_samples)
    opt_labels = opt_dbscan.fit_predict(X)
    opt_clusters = opt_dbscan.n_clusters_
    
    assert naive_clusters == sklearn_clusters, f"Naive clusters {naive_clusters} != Sklearn {sklearn_clusters}"
    assert opt_clusters == sklearn_clusters, f"Optimized clusters {opt_clusters} != Sklearn {sklearn_clusters}"
    
    print(f"✅ All implementations found {sklearn_clusters} clusters")

if __name__ == "__main__":
    test_cluster_consistency()
    test_cluster_counts_match()
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED")
    print("="*60)