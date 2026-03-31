"""
Naive DBSCAN - Vectorized baseline for fair comparison
- Added duplicate prevention (in_queue array)
- Added phase timing methods
"""
import numpy as np
import time
from collections import deque

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
        self.eps_km = eps
        self.eps_meters = eps * 1000
        self.min_samples = min_samples
        self.labels_ = None
        self.n_clusters_ = None
        
        # NYC projection to local coordinates
        try:
            from pyproj import Transformer
            self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:2263", always_xy=True)
            self.use_projection = True
            self.feet_to_meters = 0.3048006096012192
        except ImportError:
            print("⚠️ pyproj not installed. Using approximate projection.")
            self.use_projection = False
            self.lat_m_per_deg = 111000
            self.lon_m_per_deg = 88000
            self.ref_lat = 40.7580
            self.ref_lon = -73.9855
    
    def _project(self, X):
        """Convert lat/lon to local coordinates in METERS"""
        n = len(X)
        X_proj = np.zeros((n, 2), dtype=np.float64)
        
        if self.use_projection:
            for i, (lat, lon) in enumerate(X):
                x_feet, y_feet = self.transformer.transform(lon, lat)
                X_proj[i, 0] = x_feet * self.feet_to_meters
                X_proj[i, 1] = y_feet * self.feet_to_meters
        else:
            for i, (lat, lon) in enumerate(X):
                X_proj[i, 0] = (lon - self.ref_lon) * self.lon_m_per_deg
                X_proj[i, 1] = (lat - self.ref_lat) * self.lat_m_per_deg
        
        return X_proj
    
    def _region_query_vectorized(self, X_proj, point_idx):
        """Vectorized neighbor search using NumPy"""
        point = X_proj[point_idx]
        distances = np.sqrt(np.sum((X_proj - point) ** 2, axis=1))
        return np.where(distances <= self.eps_meters)[0].tolist()
    
    def fit_predict(self, X):
        """Run naive DBSCAN with vectorized queries"""
        X_proj = self._project(X)
        return self.fit_predict_phased(X, X_proj)
    
    def fit_predict_phased(self, X, X_proj=None):
        """
        Run naive DBSCAN with phase timing support.
        If X_proj is provided, uses it directly (for phase timing).
        """
        if X_proj is None:
            X_proj = self._project(X)
        
        n_samples = X_proj.shape[0]
        self.labels_ = np.full(n_samples, -1, dtype=np.int32)
        cluster_id = 0
        
        visited = np.zeros(n_samples, dtype=bool)
        in_queue = np.zeros(n_samples, dtype=bool)  # Duplicate prevention
        processed = 0
        batch_size = max(1000, n_samples // 20)
        
        for point_idx in range(n_samples):
            if visited[point_idx]:
                continue
            
            visited[point_idx] = True
            neighbors = self._region_query_vectorized(X_proj, point_idx)
            
            if len(neighbors) < self.min_samples:
                self.labels_[point_idx] = -1
            else:
                self.labels_[point_idx] = cluster_id
                seeds = deque(neighbors)
                for seed in neighbors:
                    in_queue[seed] = True
                
                while seeds:
                    current_idx = seeds.popleft()
                    in_queue[current_idx] = False
                    
                    if not visited[current_idx]:
                        visited[current_idx] = True
                        current_neighbors = self._region_query_vectorized(X_proj, current_idx)
                        if len(current_neighbors) >= self.min_samples:
                            for nb in current_neighbors:
                                if not visited[nb] and not in_queue[nb]:
                                    seeds.append(nb)
                                    in_queue[nb] = True
                    if self.labels_[current_idx] == -1:
                        self.labels_[current_idx] = cluster_id
                cluster_id += 1
            
            processed += 1
            if processed % batch_size == 0:
                pass  # Progress handled by caller
        
        self.n_clusters_ = cluster_id
        return self.labels_