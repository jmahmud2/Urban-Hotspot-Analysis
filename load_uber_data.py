"""
Load Uber data from 'data' folder with NYC filtering
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Set the data folder path
DATA_FOLDER = Path(__file__).parent / 'data'

# NYC boundaries (approximate)
NYC_LAT_MIN = 40.5
NYC_LAT_MAX = 40.9
NYC_LON_MIN = -74.3
NYC_LON_MAX = -73.7

def load_uber_2014_data(sample_size=None, filter_nyc=True):
    """
    Load all 2014 Uber data from data folder
    
    Parameters:
    -----------
    sample_size : int or None
        Number of samples to take (None for all data)
    filter_nyc : bool
        Whether to filter points to NYC area only
    """
    months = ['apr', 'may', 'jun', 'jul', 'aug', 'sep']
    all_data = []
    
    print("Loading 2014 Uber data from 'data' folder...")
    print("-" * 50)
    
    # Check if data folder exists
    if not DATA_FOLDER.exists():
        print(f"❌ Data folder not found: {DATA_FOLDER}")
        return None
    
    total_original = 0
    total_filtered = 0
    
    for month in months:
        filename = f'uber-raw-data-{month}14.csv'
        filepath = DATA_FOLDER / filename
        
        if filepath.exists():
            file_size = filepath.stat().st_size / (1024 * 1024)
            print(f"✓ Loading {filename} ({file_size:.1f} MB)...")
            
            try:
                df = pd.read_csv(filepath)
                original_count = len(df)
                total_original += original_count
                
                if len(all_data) == 0:
                    print(f"  Columns: {list(df.columns)}")
                
                # Standardize column names
                df.columns = [col.strip() for col in df.columns]
                
                # Rename columns
                column_mapping = {}
                for col in df.columns:
                    if 'Date' in col or 'Time' in col:
                        column_mapping[col] = 'pickup_time'
                    elif col.lower() == 'lat':
                        column_mapping[col] = 'lat'
                    elif col.lower() == 'lon':
                        column_mapping[col] = 'lon'
                    elif col.lower() == 'base':
                        column_mapping[col] = 'base_code'
                
                df.rename(columns=column_mapping, inplace=True)
                
                # Filter to NYC area if requested
                if filter_nyc and 'lat' in df.columns and 'lon' in df.columns:
                    df = df[
                        (df['lat'].between(NYC_LAT_MIN, NYC_LAT_MAX)) &
                        (df['lon'].between(NYC_LON_MIN, NYC_LON_MAX))
                    ]
                    filtered_count = len(df)
                    total_filtered += filtered_count
                    print(f"  Records: {original_count:,} -> {filtered_count:,} (NYC filtered)")
                else:
                    total_filtered += original_count
                    print(f"  Records: {original_count:,}")
                
                df['year'] = 2014
                df['month'] = month
                
                all_data.append(df)
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
        else:
            print(f"✗ {filename} not found")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        
        print(f"\n📊 Summary:")
        print(f"  Total original records: {total_original:,}")
        print(f"  After NYC filter: {total_filtered:,}")
        if total_original > 0:
            print(f"  Removed: {total_original - total_filtered:,} points ({(1-total_filtered/total_original)*100:.1f}%)")
        
        if sample_size and sample_size < len(combined):
            combined = combined.sample(n=sample_size, random_state=42)
            print(f"  Sampled: {sample_size:,} records for analysis")
        
        print(f"\n✓ Total records for analysis: {len(combined):,}")
        return combined
    
    return None

def prepare_gps_data(df):
    """Prepare GPS data for DBSCAN"""
    # Remove missing coordinates
    df_clean = df.dropna(subset=['lat', 'lon'])
    
    # Convert to numpy array (use float32 to save memory)
    X = df_clean[['lat', 'lon']].values.astype(np.float32)
    
    # Get timestamps if available
    timestamps = None
    if 'pickup_time' in df_clean.columns:
        timestamps = pd.to_datetime(df_clean['pickup_time'])
    
    print(f"\nPrepared GPS data:")
    print(f"  Valid points: {len(X):,}")
    if len(X) > 0:
        print(f"  Lat range: [{X[:, 0].min():.4f}, {X[:, 0].max():.4f}]")
        print(f"  Lon range: [{X[:, 1].min():.4f}, {X[:, 1].max():.4f}]")
    print(f"  Memory: {X.nbytes / (1024*1024):.1f} MB")
    
    return X, timestamps, df_clean

def quick_check():
    """Check available files"""
    print("Checking for Uber CSV files in 'data' folder...")
    print("="*50)
    
    if not DATA_FOLDER.exists():
        print(f"\n❌ Data folder not found!")
        return
    
    print(f"\nData folder: {DATA_FOLDER.absolute()}")
    print("\nFiles found:")
    
    months = ['apr', 'may', 'jun', 'jul', 'aug', 'sep']
    found = 0
    
    for month in months:
        filename = f'uber-raw-data-{month}14.csv'
        filepath = DATA_FOLDER / filename
        if filepath.exists():
            size = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {filename} ({size:.1f} MB)")
            found += 1
    
    if found == 0:
        print("\n⚠️ No 2014 Uber data files found!")
    else:
        print(f"\n✓ Found {found}/6 files")
        print(f"\nNYC Filter Bounds:")
        print(f"  Latitude: {NYC_LAT_MIN} - {NYC_LAT_MAX}")
        print(f"  Longitude: {NYC_LON_MIN} - {NYC_LON_MAX}")

if __name__ == "__main__":
    quick_check()