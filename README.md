# Urban Hotspot Analysis - Optimized DBSCAN

## Overview

This project implements an optimized DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm using a custom grid-based spatial index for detecting urban hotspots from NYC Uber pickup data. The implementation is compared against a vectorized naive baseline to demonstrate the effectiveness of spatial indexing.

## Key Results

| Points | Naive Time | Optimized Time | Speedup | Clusters |
|--------|------------|----------------|---------|----------|
| 1,000 | 0.03s | 0.03s | 1.0x | 2 |
| 2,000 | 0.10s | 0.11s | 0.9x | 3 |
| 5,000 | 0.52s | 0.24s | **2.1x** | 6 |
| 10,000 | 1.88s | 0.68s | **2.8x** | 6 |
| 20,000 | 7.37s | 2.23s | **3.3x** | 11 |
| 50,000 | 44.99s | 11.85s | **3.8x** | 12 |

- **Best Speedup**: 3.8x at 50,000 points
- **Noise Reduction**: 92% (from 52.9% to 4.4%)
- **Scaling Exponent**: Naive: 1.87, Optimized: 1.45 (theoretical O(n²) = 2.0)

## Features

- ✅ Custom grid-based spatial index (implemented from scratch)
- ✅ EPSG:2263 projection with correct feet-to-meters conversion
- ✅ Vectorized naive baseline for fair comparison
- ✅ 17 performance visualizations
- ✅ 12 cluster maps showing actual geographic clustering
- ✅ Size-independent observations (scaling exponents, crossover point, noise reduction)

## Project Structure

Urban-Hotspot-Analysis/
├── data/ # Place CSV files here
├── results/ # Generated output
│ ├── execution_time_comparison.png
│ ├── speedup_factor.png
│ ├── cluster_comparison.png
│ ├── noise_comparison.png
│ ├── cluster_quality.png
│ ├── cluster_size_comparison.png
│ ├── processing_rate.png
│ ├── scalability_analysis.png
│ ├── scaling_exponent_analysis.png
│ ├── speedup_trend.png
│ ├── performance_heatmap.png
│ ├── efficiency_convergence.png
│ ├── overhead_benefit_analysis.png
│ ├── cluster_quality_dashboard.png
│ ├── scaling_comparison.png
│ ├── comprehensive_dashboard.png
│ ├── performance_table.png
│ ├── fair_comparison_results.csv
│ └── cluster_maps/ # 12 cluster maps
│ ├── naive_1000.png ... naive_50000.png
│ └── optimized_1000.png ... optimized_50000.png
├── src/
│ ├── naive_dbscan.py # Vectorized baseline
│ └── optimized_dbscan.py # Custom grid index + DBSCAN
├── load_uber_data.py # Data loader
├── fair_comparison.py # Main comparison script
└── requirements.txt # Dependencies


## Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Steps

1. **Clone the repository**

git clone https://github.com/jmahmud2/Urban-Hotspot-Analysis.git
cd Urban-Hotspot-Analysis

2. **Create virtual environment**

python -m venv venv

3. **Activate virtual environment**

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

4. **Install dependencies**

pip install -r requirements.txt

Data Setup:

Download the NYC Uber pickup data from FiveThirtyEight's Uber FOIL Response and place the CSV files in the data/ folder:

data/
├── uber-raw-data-apr14.csv
├── uber-raw-data-may14.csv
├── uber-raw-data-jun14.csv
├── uber-raw-data-jul14.csv
├── uber-raw-data-aug14.csv
└── uber-raw-data-sep14.csv

Usage:

Quick Test (5,000 points)
python test_clusters.py


Full Comparison:

python fair_comparison.py

This runs tests on progressive dataset sizes (10,000 to 100,000 points) and generates all visualizations.


Custom Test Sizes
Edit the test_sizes list in fair_comparison.py:

test_sizes = [1000, 2000, 5000, 10000, 20000, 50000]  # Change as needed

Implementation Details:

Naive DBSCAN

Complexity: O(n²)
Distance: Euclidean on projected meters
Optimizations: Vectorized NumPy operations
Purpose: Fair baseline for comparison

Optimized DBSCAN

Complexity: O(n log n) in practice
Spatial Index: Custom grid-based index (implemented from scratch)
Distance: Euclidean on projected meters (EPSG:2263 with feet-to-meters conversion)
Optimizations: Deque for O(1) queue operations, vectorized candidate filtering

Acknowledgments:

NYC Taxi & Limousine Commission for data release

FiveThirtyEight for FOIL request and data publication
