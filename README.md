# Urban-Hotspot-Analysis
Group Project for SDSC3002: Data Mining
# Urban Hotspot Analysis - Optimized DBSCAN

## Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/jmahmud2/Urban-Hotspot-Analysis.git
cd Urban-Hotspot-Analysis

2. Create Virtual Environment
bash
python -m venv venv
3. Activate Virtual Environment
Windows:

bash
venv\Scripts\activate
Mac/Linux:

bash
source venv/bin/activate
4. Install Dependencies
bash
pip install -r requirements.txt
5. Download Data
Download NYC Uber data from FiveThirtyEight's Uber FOIL Response and place CSV files in the data/ folder.

Required files:

uber-raw-data-apr14.csv

uber-raw-data-may14.csv

uber-raw-data-jun14.csv

uber-raw-data-jul14.csv

uber-raw-data-aug14.csv

uber-raw-data-sep14.csv

6. Run Quick Test (5,000 points)
bash
python test_clusters.py
7. Run Full Comparison
bash
python fair_comparison.py
This runs tests from 1,000 to 10,000 points and generates results in the results/ folder.