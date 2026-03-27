"""
Test your setup before running the analysis
"""
import os
import sys
from pathlib import Path

def test_setup():
    print("="*60)
    print("Testing Urban Hotspot Analysis Setup")
    print("="*60)
    
    # Test 1: Check folder structure
    print("\n[1/4] Checking folder structure:")
    folders = ['data', 'src', 'benchmarks', 'results']
    for folder in folders:
        if os.path.exists(folder):
            print(f"  ✓ {folder}/ exists")
        else:
            print(f"  ✗ {folder}/ missing")
            os.makedirs(folder, exist_ok=True)
            print(f"  ✓ Created {folder}/")
    
    # Test 2: Check data files
    print("\n[2/4] Checking data files:")
    data_folder = Path('data')
    if data_folder.exists():
        csv_files = list(data_folder.glob('uber-raw-data-*.csv'))
        if csv_files:
            print(f"  ✓ Found {len(csv_files)} CSV files:")
            for csv_file in csv_files[:3]:  # Show first 3
                size = csv_file.stat().st_size / (1024 * 1024)
                print(f"    - {csv_file.name} ({size:.1f} MB)")
            if len(csv_files) > 3:
                print(f"    ... and {len(csv_files)-3} more")
        else:
            print("  ⚠️ No Uber CSV files found in data/ folder")
            print("     Expected files: uber-raw-data-apr14.csv, etc.")
    else:
        print("  ✗ data/ folder not found")
    
    # Test 3: Check imports
    print("\n[3/4] Testing Python imports:")
    try:
        import numpy as np
        print("  ✓ NumPy")
        import pandas as pd
        print("  ✓ Pandas")
        import matplotlib
        print("  ✓ Matplotlib")
        from sklearn.cluster import DBSCAN
        print("  ✓ Scikit-learn")
        from scipy.spatial import cKDTree
        print("  ✓ SciPy")
        print("\n  ✅ All imports successful!")
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        print("\n  Run: pip install -r requirements.txt")
        return False
    
    # Test 4: Check our module
    print("\n[4/4] Testing custom module:")
    try:
        from src.optimized_dbscan import OptimizedDBSCAN
        print("  ✓ OptimizedDBSCAN imported successfully")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("\nNext steps:")
    print("1. Make sure CSV files are in the 'data' folder")
    print("2. Run: python run_hotspot_analysis.py")
    
    return True

if __name__ == "__main__":
    test_setup()