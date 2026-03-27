"""
Simple test to verify clustering works on both implementations
"""
import numpy as np
from load_uber_data import load_uber_2014_data, prepare_gps_data
from src.naive_dbscan import NaiveDBSCAN
from src.optimized_dbscan import OptimizedDBSCAN

print("="*60)
print("QUICK TEST - 5,000 points")
print("="*60)

# Load data
df = load_uber_2014_data(sample_size=5000, filter_nyc=True)
X, timestamps, _ = prepare_gps_data(df)

eps_km = 0.5
min_samples = 30

print(f"\nParameters: eps={eps_km}km, min_samples={min_samples}")

# Naive DBSCAN
print("\n1. Naive DBSCAN...")
naive_dbscan = NaiveDBSCAN(eps=eps_km, min_samples=min_samples)
naive_labels = naive_dbscan.fit_predict(X)
print(f"   Found {naive_dbscan.n_clusters_} clusters")

# Optimized DBSCAN
print("\n2. Optimized DBSCAN...")
opt_dbscan = OptimizedDBSCAN(eps=eps_km, min_samples=min_samples)
opt_labels = opt_dbscan.fit_predict(X)
print(f"   Found {opt_dbscan.n_clusters_} clusters")

# Compare
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"Naive clusters: {naive_dbscan.n_clusters_}")
print(f"Optimized clusters: {opt_dbscan.n_clusters_}")

if naive_dbscan.n_clusters_ == opt_dbscan.n_clusters_:
    print("\n✅ SUCCESS! Both found the same number of clusters!")
else:
    print(f"\n⚠️ Mismatch. Naive: {naive_dbscan.n_clusters_}, Optimized: {opt_dbscan.n_clusters_}")