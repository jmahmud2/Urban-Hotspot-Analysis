# Urban Hotspot Analysis - Optimized DBSCAN

## Overview

This project implements an optimized DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm using a **custom grid-based spatial index** for detecting urban hotspots from NYC Uber pickup data. The implementation is rigorously benchmarked against a vectorized naive baseline with **5 runs per test size**, reporting **mean ± standard deviation** for statistical significance.

## Key Results

| Points | Naive Time (s) | Optimized Time (s) | Speedup | ARI | Clusters |
|--------|----------------|--------------------|---------|-----|----------|
| 10,000 | 1.69 ± 0.00 | 0.66 ± 0.00 | 2.6x | 1.000 | 6 |
| 20,000 | 10.93 ± 0.00 | 3.62 ± 0.00 | 3.0x | 1.000 | 11 |
| 30,000 | 23.24 ± 0.00 | 6.44 ± 2.45 | 3.6x | 1.000 | 14 |
| 40,000 | 46.41 ± 4.12 | 14.30 ± 4.33 | 3.2x | 1.000 | 12 |
| 50,000 | 67.88 ± 22.67 | 21.06 ± 7.53 | 3.2x | 1.000 | 12 |
| 60,000 | 133.70 ± 20.08 | 44.04 ± 2.53 | 3.0x | 1.000 | 12 |
| 70,000 | 238.52 ± 9.73 | 54.43 ± 6.13 | **4.4x** | 1.000 | 13 |
| 80,000 | 283.87 ± 22.72 | 75.66 ± 3.20 | 3.8x | 1.000 | 14 |
| 90,000 | 379.61 ± 31.52 | 82.23 ± 22.77 | 4.6x | 1.000 | 18 |
| 100,000 | 381.97 ± 78.32 | 73.72 ± 27.45 | **5.2x** | 1.000 | 17 |

- **Best Speedup**: 5.2x ± 2.2x at 100,000 points
- **Noise Reduction**: 78% (from 12.8% at 10k to 2.9% at 100k)
- **Correctness**: Perfect cluster agreement (ARI = 1.000)
- **Scaling Exponent**: Naive: 2.41 | Optimized: 2.17 (theoretical O(n²) = 2.0)

## Features

- ✅ **Custom grid-based spatial index** (implemented from scratch, not sklearn)
- ✅ **EPSG:2263 projection** with correct feet-to-meters conversion
- ✅ **Vectorized naive baseline** for fair comparison
- ✅ **5 runs per test size** with mean ± standard deviation
- ✅ **Error bars** on all performance plots
- ✅ **ARI validation** proving both implementations produce identical clusters
- ✅ **Phase-wise timing breakdown** (projection, index build, clustering)
- ✅ **12 cluster maps** showing actual geographic clustering at 10k, 50k, 100k points
- ✅ **Statistically rigorous benchmarking**

## Project Structure

Urban-Hotspot-Analysis/
├── data/ # Place CSV files here
├── results/ # Generated output
│ ├── execution_time_comparison.png # Time with error bars
│ ├── speedup_factor.png # Speedup with error bars
│ ├── ari_comparison.png # Correctness verification
│ ├── phase_timing_breakdown.png # Phase-wise timing
│ ├── processing_rate.png # Points/sec comparison
│ ├── cluster_comparison.png # Cluster count comparison
│ ├── noise_comparison.png # Noise reduction
│ ├── cluster_quality.png # Separation ratio
│ ├── performance_table.png # Summary table
│ ├── fair_comparison_results.csv # Raw data
│ └── cluster_maps/ # 6 cluster maps
│ ├── naive_10000.png
│ ├── naive_50000.png
│ ├── naive_100000.png
│ ├── optimized_10000.png
│ ├── optimized_50000.png
│ └── optimized_100000.png
├── src/
│ ├── naive_dbscan.py # Vectorized baseline (with in_queue)
│ └── optimized_dbscan.py # Custom grid index + DBSCAN
├── load_uber_data.py # Data loader
├── fair_comparison.py # Main benchmark script
└── requirements.txt # Dependencies


## Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Steps

1. **Clone the repository**
`
git clone https://github.com/jmahmud2/Urban-Hotspot-Analysis.git
cd Urban-Hotspot-Analysis

2. **Create virtual envrionment**

python -m venv venv

3. **Activate virtual environment**

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

4. **Install dependencies**

pip install -r requirements.txt

**Data Setup**

Download the NYC Uber pickup data from FiveThirtyEight's Uber FOIL Response and place the CSV files in the data/ folder:

data/
├── uber-raw-data-apr14.csv
├── uber-raw-data-may14.csv
├── uber-raw-data-jun14.csv
├── uber-raw-data-jul14.csv
├── uber-raw-data-aug14.csv
└── uber-raw-data-sep14.csv

**Usage**

Run Full Benchmark (5 runs per test size, 10k to 100k points): 

python fair_comparison.py

**Custom Test Sizes**

Edit the test_sizes list in fair_comparison.py: 

test_sizes = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]

**Change Number of Runs**

Edit the N_RUNS variable at the top of fair_comparison.py:

N_RUNS = 5  # Change to 3 for faster runs

**Implementation Details**

Naive DBSCAN (Baseline):

Complexity: O(n²)
Distance: Euclidean on projected meters
Optimizations: Vectorized NumPy operations, deque for O(1) queue, duplicate prevention
Purpose: Fair baseline for comparison

Optimized DBSCAN (Our Contribution):
Complexity: O(n log n) in practice
Spatial Index: Custom grid-based index (implemented from scratch)
Distance: Euclidean on projected meters (EPSG:2263 with feet-to-meters conversion)
Optimizations: Deque for O(1) queue operations, vectorized candidate filtering, duplicate prevention

**Dependencies**

numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
scipy==1.10.1
pyproj==3.6.0
scikit-learn==1.3.0

**Acknowledgements**

NYC Taxi & Limousine Commission for data release
