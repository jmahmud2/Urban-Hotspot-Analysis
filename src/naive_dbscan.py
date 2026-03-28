"""
Naive DBSCAN - Vectorized baseline for fair comparison
"""
import numpy as np
import time

class NaiveDBSCAN:
    def __init__(self, eps=0.5, min_samples=30):
        """
        Parameters:
        -----------
        eps : float
            Maximum distance between points in KILOMETERS
        min_samples : int
            Minimum points to form a cluster
        """
        self.eps_meters = eps * 1000
        self.min_samples = min_samples
        self.labels_ = None
        self.n_clusters_ = None
        
        # NYC projection to local coordinates (meters)
        try:
            from pyproj import Transformer
            self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:2263", always_xy=True)
            self.use_projection = True
        except ImportError:
            print("⚠️ pyproj not installed. Using approximate projection.")
            self.use_projection = False
            # Approximate conversion for NYC (1° ≈ 111km lat, 88km lon)
            self.lat_m_per_deg = 111000
            self.lon_m_per_deg = 88000
            self.ref_lat = 40.7580
            self.ref_lon = -73.9855
    
    def _project(self, X):
        """Convert lat/lon to local coordinates in meters"""
        n = len(X)
        X_proj = np.zeros((n, 2), dtype=np.float32)
        
        if self.use_projection:
            for i, (lat, lon) in enumerate(X):
                x, y = self.transformer.transform(lon, lat)
                X_proj[i, 0] = x
                X_proj[i, 1] = y
        else:
            for i, (lat, lon) in enumerate(X):
                X_proj[i, 0] = (lon - self.ref_lon) * self.lon_m_per_deg
                X_proj[i, 1] = (lat - self.ref_lat) * self.lat_m_per_deg
        
        return X_proj
    
    def fit_predict(self, X):
        """Run naive DBSCAN with vectorized queries"""
        print("="*50)
        print("NAIVE DBSCAN (Vectorized Baseline)")
        print("="*50)
        
        print("\nProjecting coordinates to local system...")
        start = time.time()
        X_proj = self._project(X)
        print(f"  Projection took {time.time() - start:.2f}s")
        
        n_samples = X_proj.shape[0]
        self.labels_ = np.full(n_samples, -1, dtype=np.int32)
        cluster_id = 0
        
        print(f"Points: {n_samples:,}")
        print(f"Eps: {self.eps_meters/1000:.1f} km ({self.eps_meters:.0f} meters)")
        print(f"Min samples: {self.min_samples}")
        
        visited = np.zeros(n_samples, dtype=bool)
        processed = 0
        batch_size = max(1000, n_samples // 20)
        
        print("\nClustering points...")
        start_time = time.time()
        
        for point_idx in range(n_samples):
            if visited[point_idx]:
                continue
            
            visited[point_idx] = True
            
            # Vectorized distance calculation
            point = X_proj[point_idx]
            distances = np.sqrt(np.sum((X_proj - point) ** 2, axis=1))
            neighbors = np.where(distances <= self.eps_meters)[0].tolist()
            
            if len(neighbors) < self.min_samples:
                self.labels_[point_idx] = -1
            else:
                self.labels_[point_idx] = cluster_id
                seeds = list(neighbors)
                while seeds:
                    current_idx = seeds.pop(0)
                    if not visited[current_idx]:
                        visited[current_idx] = True
                        current_point = X_proj[current_idx]
                        current_distances = np.sqrt(np.sum((X_proj - current_point) ** 2, axis=1))
                        current_neighbors = np.where(current_distances <= self.eps_meters)[0].tolist()
                        if len(current_neighbors) >= self.min_samples:
                            seeds.extend(current_neighbors)
                    if self.labels_[current_idx] == -1:
                        self.labels_[current_idx] = cluster_id
                cluster_id += 1
            
            processed += 1
            if processed % batch_size == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed
                print(f"  Processed {processed:,}/{n_samples:,} points ({rate:.0f} pts/sec)")
        
        elapsed_time = time.time() - start_time
        self.n_clusters_ = cluster_id
        n_noise = np.sum(self.labels_ == -1)
        
        print(f"\n{'='*50}")
        print(f"Results")
        print(f"{'='*50}")
        print(f"Time: {elapsed_time:.2f} seconds")
        print(f"Clusters: {self.n_clusters_}")
        print(f"Noise: {n_noise:,} ({n_noise/n_samples*100:.1f}%)")
        
        return self.labels_