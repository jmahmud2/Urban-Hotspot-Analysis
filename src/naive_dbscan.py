"""
Naive DBSCAN implementation WITHOUT spatial indexing
O(n²) complexity - intentionally slow to demonstrate the value of optimization
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
        self.eps_km = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.n_clusters_ = None
        # Convert km to degrees for Euclidean distance (approximate)
        self.eps_degrees = eps / 111.0
        
    def _distance(self, p1, p2):
        """Euclidean distance between two points in degrees"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _region_query(self, X, point_idx):
        """Find neighbors by checking EVERY point (O(n) per query)"""
        neighbors = []
        for i in range(len(X)):
            if self._distance(X[point_idx], X[i]) <= self.eps_degrees:
                neighbors.append(i)
        return neighbors
    
    def fit_predict(self, X):
        """Run naive DBSCAN - intentionally slow O(n²) implementation"""
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1, dtype=np.int32)
        cluster_id = 0
        
        print(f"\n{'='*50}")
        print(f"NAIVE DBSCAN (O(n²) - No Spatial Index)")
        print(f"{'='*50}")
        print(f"Points: {n_samples:,}")
        print(f"Eps: {self.eps_km} km ({self.eps_degrees:.6f} degrees)")
        print(f"Min samples: {self.min_samples}")
        print(f"⚠️ This is intentionally slow to demonstrate the need for spatial indexing")
        
        visited = np.zeros(n_samples, dtype=bool)
        processed = 0
        batch_size = max(1000, n_samples // 20)
        
        print("\nClustering points...")
        start_time = time.time()
        
        for point_idx in range(n_samples):
            if visited[point_idx]:
                continue
            
            visited[point_idx] = True
            neighbors = self._region_query(X, point_idx)
            
            if len(neighbors) < self.min_samples:
                self.labels_[point_idx] = -1
            else:
                self.labels_[point_idx] = cluster_id
                seeds = list(neighbors)
                while seeds:
                    current_idx = seeds.pop(0)
                    if not visited[current_idx]:
                        visited[current_idx] = True
                        current_neighbors = self._region_query(X, current_idx)
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