"""
Optimized DBSCAN with spatial indexing for GPS data
"""
import numpy as np
from scipy.spatial import cKDTree
import time
from collections import deque

class OptimizedDBSCAN:
    """Fast DBSCAN using spatial indexing for GPS coordinates"""
    
    def __init__(self, eps=0.5, min_samples=30, algorithm='kd_tree', leaf_size=100):
        """
        Parameters:
        -----------
        eps : float
            Maximum distance between points in KILOMETERS
        min_samples : int
            Minimum points to form a cluster
        algorithm : str
            'kd_tree' (faster) - uses Euclidean distance
        leaf_size : int
            Leaf size for spatial index
        """
        self.eps = eps
        self.min_samples = min_samples
        self.leaf_size = leaf_size
        self.labels_ = None
        self.n_clusters_ = None
        
    def _build_spatial_index(self, X):
        """Build spatial index for fast neighbor queries"""
        # Convert km to degrees for Euclidean distance (approximate)
        # 1 degree latitude ≈ 111 km
        # For NYC, 1 degree longitude ≈ 88 km
        # Use average: 1 degree ≈ 100 km for simplicity
        eps_degrees = self.eps / 100.0
        
        # Use cKDTree with scaled coordinates for Euclidean distance
        return cKDTree(X, leafsize=self.leaf_size)
    
    def _region_query(self, tree, X, point_idx):
        """Find neighbors within eps distance"""
        point = X[point_idx].reshape(1, -1)
        
        # Convert eps from km to degrees
        eps_degrees = self.eps / 100.0
        
        indices = tree.query_ball_point(point, eps_degrees)
        return indices
    
    def fit_predict(self, X):
        """Run DBSCAN clustering"""
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1, dtype=np.int32)
        cluster_id = 0
        
        print(f"\n{'='*50}")
        print(f"Optimized DBSCAN Clustering")
        print(f"{'='*50}")
        print(f"Points: {n_samples:,}")
        print(f"Eps: {self.eps} km")
        print(f"Min samples: {self.min_samples}")
        
        # Build spatial index
        print("\nBuilding spatial index...")
        start_time = time.time()
        tree = self._build_spatial_index(X)
        build_time = time.time() - start_time
        print(f"Index built in {build_time:.2f} seconds")
        
        # Process points
        visited = np.zeros(n_samples, dtype=bool)
        processed = 0
        batch_size = max(1000, n_samples // 20)
        
        print("\nClustering points...")
        start_time = time.time()
        
        for point_idx in range(n_samples):
            if visited[point_idx]:
                continue
            
            visited[point_idx] = True
            neighbors = self._region_query(tree, X, point_idx)
            
            if len(neighbors) < self.min_samples:
                self.labels_[point_idx] = -1  # Noise
            else:
                # Start new cluster
                self.labels_[point_idx] = cluster_id
                self._expand_cluster(tree, X, neighbors, cluster_id, visited)
                cluster_id += 1
            
            processed += 1
            if processed % batch_size == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed
                print(f"  Processed {processed:,}/{n_samples:,} points "
                      f"({rate:.0f} pts/sec)")
        
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
    
    def _expand_cluster(self, tree, X, neighbors, cluster_id, visited):
        """Expand cluster using BFS"""
        queue = deque(neighbors)
        
        while queue:
            current_idx = queue.popleft()
            
            if not visited[current_idx]:
                visited[current_idx] = True
                current_neighbors = self._region_query(tree, X, current_idx)
                
                if len(current_neighbors) >= self.min_samples:
                    queue.extend(current_neighbors)
            
            if self.labels_[current_idx] == -1:
                self.labels_[current_idx] = cluster_id