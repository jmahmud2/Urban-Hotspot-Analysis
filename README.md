# Urban Hotspot Analysis - Optimized DBSCAN

## Overview
Optimized DBSCAN with BallTree spatial indexing for NYC Uber hotspot detection. Achieves **76x speedup** over naive implementation at 10,000 points.

## Key Results
- **76x faster** at 10,000 points (103.6s → 1.36s)
- **6-7 hotspots** identified (Times Square, Penn Station, JFK, etc.)
- **O(n log n)** vs O(n²) scaling

## Installation

```bash
git clone https://github.com/jmahmud2/Urban-Hotspot-Analysis.git
cd Urban-Hotspot-Analysis
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Data Setup
Place Uber CSV files in data/ folder. Download from: https://github.com/fivethirtyeight/uber-tlc-foil-response

data/
├── uber-raw-data-apr14.csv
├── uber-raw-data-may14.csv
├── uber-raw-data-jun14.csv
├── uber-raw-data-jul14.csv
├── uber-raw-data-aug14.csv
└── uber-raw-data-sep14.csv

Run Analysis

python test_clusters.py        # Quick test (5,000 points)
python fair_comparison.py      # Full comparison (1,000-10,000 points)
Output
Results saved to results/ folder:

File	Description
execution_time_comparison.png	Time comparison: Naive vs Optimized
speedup_factor.png	Speedup factor graph
clusters_comparison.png	Cluster count comparison
noise_ratio_comparison.png	Noise percentage comparison
scalability_analysis.png	Log-log scaling analysis
cluster_maps/	20 cluster maps (10 naive + 10 optimized)
temporal_patterns.png	Hourly and weekly pickup patterns
parameter_sensitivity.png	Effect of eps and min_samples on clusters
hotspot_ranking.png	Top 10 hotspots with location names
borough_distribution.png	Hotspot distribution across NYC boroughs
comprehensive_dashboard.png	All metrics combined dashboard

Project Structure
Urban-Hotspot-Analysis/
├── data/                          # CSV files here
├── results/                       # Output visualizations
├── src/
│   ├── naive_dbscan.py            # Baseline O(n²)
│   └── optimized_dbscan.py        # Optimized O(n log n)
├── load_uber_data.py              # Data loader
├── fair_comparison.py             # Main script
├── test_clusters.py               # Quick test
└── requirements.txt               # Dependencies
Dependencies

numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
scikit-learn==1.3.0
scipy==1.10.1