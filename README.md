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