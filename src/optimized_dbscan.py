"""
Optimized DBSCAN with custom Grid Index - Implemented from scratch
"""
import numpy as np
import time
from collections import defaultdict, deque
import math

class GridIndex:
    """Grid-based spatial index implemented from scratch"""
    
    def __init__(self, X, cell_size_meters=250):
        """
        Build grid index from points
        
        Parameters:
        -----------
        X : numpy array of shape (n, 2)
            Projected coordinates in meters
        cell_size_meters : float
            Grid cell size in meters (smaller = better filtering, more memory)
        """
        self.cell_size = cell_size_meters
        self.points = X.astype(np.float32)
        self.n_points = len(X)
        
        # Build grid using dictionary
        self.grid = defaultdict(list)
        
        # Vectorized cell computation
        cell_x = (self.points[:, 0] / self.cell_size).astype(np.int32)
        cell_y = (self.points[:, 1] / self.cell_size).astype(np.int32)
        
        for i in range(self.n_points):
            self.grid[(cell_x[i], cell_y[i])].append(i)
        
        # Simple cache for neighbor cells (repeated queries benefit)
        self._neighbor_cache = {}
    
    def _get_cell(self, point):
        """Get cell coordinates for a point"""
        return (int(point[0] / self.cell_size), int(point[1] / self.cell_size))
    
    def _get_neighbor_cells(self, cell, radius):
        """Get all cells within radius of given cell with caching"""
        key = (cell[0], cell[1], radius)
        if key in self._neighbor_cache:
            return self._neighbor_cache[key]
        
        cells = []
        cx, cy = cell
        radius_cells = int(math.ceil(radius / self.cell_size)) + 1
        
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                cells.append((cx + dx, cy + dy))
        
        self._neighbor_cache[key] = cells
        return cells
    
    def query_radius(self, point, radius):
        """
        Find all points within radius of query point
        Uses vectorized distance calculation
        """
        cell = self._get_cell(point)
        neighbor_cells = self._get_neighbor_cells(cell, radius)
        
        # Collect candidate points from neighbor cells
        candidates = []
        for c in neighbor_cells:
            if c in self.grid:
                candidates.extend(self.grid[c])
        
        if not candidates:
            return []
        
        # Vectorized distance calculation (use squared to avoid sqrt)
        candidates = np.array(candidates, dtype=np.int32)
        points = self.points[candidates]
        
        dx = points[:, 0] - point[0]
        dy = points[:, 1] - point[1]
        dist_sq = dx*dx + dy*dy
        radius_sq = radius * radius
        
        return candidates[dist_sq <= radius_sq].tolist()


class OptimizedDBSCAN:
    def __init__(self, eps=0.5, min_samples=30, cell_size_km=0.25):
        """
        Parameters:
        -----------
        eps : float
            Maximum distance between points in KILOMETERS
        min_samples : int
            Minimum points to form a cluster
        cell_size_km : float
            Grid cell size in kilometers (tunable parameter)
        """
        self.eps_meters = eps * 1000
        self.min_samples = min_samples
        self.cell_size_meters = cell_size_km * 1000
        self.labels_ = None
        self.n_clusters_ = None
        self.grid_index = None
        
        # NYC projection to local coordinates (meters)
        try:
            from pyproj import Transformer
            self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:2263", always_xy=True)
            self.use_projection = True
        except ImportError:
            print("⚠️ pyproj not installed. Using approximate projection.")
            self.use_projection = False
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
        """Run DBSCAN with custom grid index"""
        print("="*50)
        print("OPTIMIZED DBSCAN (Custom Grid Index)")
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
        print(f"Grid cell size: {self.cell_size_meters:.0f} meters")
        
        print("\nBuilding grid index...")
        start = time.time()
        self.grid_index = GridIndex(X_proj, cell_size_meters=self.cell_size_meters)
        print(f"  Index built in {time.time() - start:.2f}s")
        
        visited = np.zeros(n_samples, dtype=bool)
        processed = 0
        batch_size = max(1000, n_samples // 20)
        
        print("\nClustering points...")
        start_time = time.time()
        
        for point_idx in range(n_samples):
            if visited[point_idx]:
                continue
            
            visited[point_idx] = True
            neighbors = self.grid_index.query_radius(X_proj[point_idx], self.eps_meters)
            
            if len(neighbors) < self.min_samples:
                self.labels_[point_idx] = -1
            else:
                self.labels_[point_idx] = cluster_id
                seeds = deque(neighbors)
                
                while seeds:
                    current_idx = seeds.popleft()
                    if not visited[current_idx]:
                        visited[current_idx] = True
                        current_neighbors = self.grid_index.query_radius(
                            X_proj[current_idx], self.eps_meters
                        )
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