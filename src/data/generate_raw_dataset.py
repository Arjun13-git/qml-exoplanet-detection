# src/data/generate_raw_dataset.py
import lightkurve as lk
import numpy as np
import pandas as pd
import os
from stellar_catalog import STELLAR_CATALOG

def extract_transit_features(tic_id, period, epoch):
    """Downloads, cleans, and returns the pure flux array for a star."""
    print(f"🔄 Processing {tic_id}...")
    try:
        # 1. Download SPOC data (Phase 1)
        search_result = lk.search_lightcurve(tic_id, author='SPOC', mission='TESS', cadence='fast')
        if len(search_result) == 0:
            return None
            
        lc = search_result[:2].download_all().stitch()
        
        # 2. Savitzky-Golay Flattening (Phase 1)
        lc_flat = lc.flatten(window_length=901)
        
        # 3. Phase Folding & Binning (Phase 1)
        lc_folded = lc_flat.fold(period=period, epoch_time=epoch)
        lc_binned = lc_folded.bin(time_bin_size=0.005)
        
        # We drop NaN values that occur during binning gaps
        flux_array = lc_binned.flux.value
        flux_array = flux_array[~np.isnan(flux_array)]
        
        # Ensure uniform vector size (e.g., exactly 200 data points per star)
        # Interpolation ensures every star's array is identical in length for the Neural Network
        target_length = 200
        standardized_flux = np.interp(
            np.linspace(0, 1, target_length), 
            np.linspace(0, 1, len(flux_array)), 
            flux_array
        )
        return standardized_flux
        
    except Exception as e:
        print(f"❌ Failed to process {tic_id}: {e}")
        return None

def build_master_dataset():
    print("🚀 Initializing Master Dataset Construction...")
    dataset_rows = []
    
    for key, data in STELLAR_CATALOG.items():
        flux_vector = extract_transit_features(data['tic_id'], data['period'], data['epoch'])
        
        if flux_vector is not None:
            # Create a row with the label first, followed by the 200 flux data points
            row = [data['tic_id'], data['label']] + flux_vector.tolist()
            dataset_rows.append(row)
            print(f"✅ Successfully extracted features for {data['tic_id']}")
            
    # Define column names: 'ID', 'Label', 'F_1', 'F_2', ..., 'F_200'
    columns = ['tic_id', 'label'] + [f'flux_{i}' for i in range(200)]
    df = pd.DataFrame(dataset_rows, columns=columns)
    
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/master_lightcurves.csv"
    df.to_csv(output_path, index=False)
    print(f"💾 Master dataset saved to {output_path} with shape {df.shape}")

if __name__ == "__main__":
    build_master_dataset()