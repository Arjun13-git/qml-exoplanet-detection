import pandas as pd

def append_false_positives():
    print("📡 Fetching False Positives from NASA Exoplanet Archive API...")
    
    # Query the cumulative Kepler table specifically for FALSE POSITIVE dispositions
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+kepid,koi_period,koi_time0bk+from+cumulative+where+koi_disposition='FALSE+POSITIVE'+and+koi_period>0+and+koi_time0bk>0&format=csv"
    
    df_fp = pd.read_csv(url)
    
    # Grab 600 samples (we fetch slightly more than 493 to account for potential lightkurve download failures)
    df_fp = df_fp.sample(n=600, random_state=42)
    
    # Format to match your existing catalog exactly
    new_data = pd.DataFrame({
        'target_name': 'KIC ' + df_fp['kepid'].astype(str),
        'label': 0,  # Negative class
        'period': df_fp['koi_period'],
        'epoch': df_fp['koi_time0bk'] + 2454833.0  # Convert Kepler BKJD to standard Julian Date
    })
    
    catalog_path = "data/raw/full_stellar_catalog.csv"
    
    # Append these new rows to the bottom of your existing CSV
    new_data.to_csv(catalog_path, mode='a', header=False, index=False)
    print(f"✅ Appended {len(new_data)} negative samples (label 0) to {catalog_path}")

if __name__ == "__main__":
    append_false_positives()