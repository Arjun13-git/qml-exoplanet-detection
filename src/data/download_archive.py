# src/data/download_archive.py
import lightkurve as lk
import os

def download_tess_lightcurve(tic_id, output_dir="data/raw"):
    print(f"🚀 Querying MAST Archive for target: {tic_id}...")
    
    # Search for high-cadence (2-minute) data processed by the SPOC pipeline
    search_result = lk.search_lightcurve(tic_id, author='SPOC', mission='TESS', cadence='fast')
    
    if len(search_result) == 0:
        print(f"❌ No high-cadence SPOC data found for {tic_id}.")
        return None
        
    print(f"📦 Found {len(search_result)} data sectors. Downloading now...")
    lc_collection = search_result.download_all()
    
    # Stitch all observation sectors into one unified time-series
    print("🧵 Stitching sectors into a single continuous timeline...")
    lc_stitched = lc_collection.stitch()
    
    # Ensure directory exists and save to CSV
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{tic_id}_raw.csv")
    lc_stitched.to_csv(file_path)
    
    print(f"✅ Success! Saved {len(lc_stitched)} data points to: {file_path}")
    return file_path

if __name__ == "__main__":
    target_star = "TIC 260128333"  # WASP-121
    download_tess_lightcurve(target_star)