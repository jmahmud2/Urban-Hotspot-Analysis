"""
DEBUG: Find parameters that make your implementation match sklearn
"""
import numpy as np
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.metrics import adjusted_rand_score
from load_uber_data import load_uber_2014_data, prepare_gps_data
from src.optimized_dbscan import OptimizedDBSCAN

print("="*60)
print("DEBUG: Finding parameters to match sklearn")
print("="*60)

# Load data
df = load_uber_2014_data(sample_size=5000, filter_nyc=True)
X, timestamps, _ = prepare_gps_data(df)

# Sklearn baseline
eps_km = 0.5
eps_degrees = eps_km / 111.0
min_samples = 30

sklearn_dbscan = SklearnDBSCAN(eps=eps_degrees, min_samples=min_samples)
sklearn_labels = sklearn_dbscan.fit_predict(X)
sklearn_clusters = len(set(sklearn_labels)) - (1 if -1 in sklearn_labels else 0)
print(f"\nSklearn baseline: {sklearn_clusters} clusters at eps={eps_km}km")

# Test different eps values for your implementation
test_eps = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

print("\nTesting different eps values to match sklearn:")
print("-" * 50)

best_ari = 0
best_eps = None

for eps in test_eps:
    opt_dbscan = OptimizedDBSCAN(eps=eps, min_samples=min_samples)
    opt_labels = opt_dbscan.fit_predict(X)
    opt_clusters = opt_dbscan.n_clusters_
    ari = adjusted_rand_score(sklearn_labels, opt_labels)
    
    match = "✓ MATCH" if opt_clusters == sklearn_clusters else ""
    print(f"  eps={eps}km -> clusters={opt_clusters}, ARI={ari:.4f} {match}")
    
    if ari > best_ari:
        best_ari = ari
        best_eps = eps

print("\n" + "="*60)
print("RECOMMENDATION:")
print(f"  Best match: eps={best_eps}km gives ARI={best_ari:.4f}")
print(f"  Sklearn clusters: {sklearn_clusters}")
print(f"  Your clusters at that eps: will show above")
print("="*60)