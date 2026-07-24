# # src/data/generate_raw_dataset.py
# import lightkurve as lk
# import numpy as np
# import pandas as pd
# import os
# import time

# def extract_transit_features(target_name, period, epoch):
#     try:
#         search_result = lk.search_lightcurve(target_name, author='SPOC', mission='TESS', cadence='fast')
#         if len(search_result) == 0:
#             search_result = lk.search_lightcurve(target_name, author='Kepler', cadence='short')
            
#         if len(search_result) == 0:
#             return None
            
#         # FIX: Just grab the first available high-quality sector. No stitching required.
#         lc = search_result[0].download()
#         if lc is None:
#             return None
        
#         # Time Calibration Fix
#         if epoch > 2400000:
#             if lc.time.format == 'btjd': 
#                 epoch = epoch - 2457000
#             elif lc.time.format == 'bkjd': 
#                 epoch = epoch - 2454833
        
#         lc_flat = lc.flatten(window_length=901)
#         lc_folded = lc_flat.fold(period=period, epoch_time=epoch)
#         lc_binned = lc_folded.bin(time_bin_size=0.005)
        
#         flux_array = lc_binned.flux.value
#         flux_array = flux_array[~np.isnan(flux_array)]
        
#         # Strict quality control: reject highly corrupted/sparse light curves
#         if len(flux_array) < 50:
#             return None
            
#         standardized_flux = np.interp(
#             np.linspace(0, 1, 200), 
#             np.linspace(0, 1, len(flux_array)), 
#             flux_array
#         )
#         return standardized_flux
        
#     except Exception as e:
#         print(f"❌ Error on {target_name}: {str(e)}")
#         return None

# def build_master_dataset():
#     catalog_path = "data/raw/full_stellar_catalog.csv"
#     output_path = "data/processed/master_lightcurves.csv"
    
#     if not os.path.exists(catalog_path):
#         raise FileNotFoundError("Run parse_nasa_archive.py first!")
        
#     catalog_df = pd.read_csv(catalog_path)
    
#     processed_targets = set()
#     if os.path.exists(output_path):
#         existing_df = pd.read_csv(output_path)
#         processed_targets = set(existing_df['target_name'].values)
#         print(f"🔄 Found existing dataset with {len(processed_targets)} processed stars. Resuming...")
#     else:
#         print("🚀 Starting fresh bulk download...")
#         columns = ['target_name', 'label'] + [f'flux_{i}' for i in range(200)]
#         pd.DataFrame(columns=columns).to_csv(output_path, index=False)
    
#     success_count = 0
#     for _, row in catalog_df.iterrows():
#         target_name = row['target_name']
        
#         # Skip if we already downloaded this star (handles multi-planet systems)
#         if target_name in processed_targets:
#             continue 
            
#         print(f"⏳ Fetching {target_name}...")
#         flux_vector = extract_transit_features(target_name, row['period'], row['epoch'])
        
#         # FIX: Add the star to the set immediately so we don't fetch it again
#         processed_targets.add(target_name)
        
#         if flux_vector is not None:
#             data_row = [target_name, row['label']] + flux_vector.tolist()
#             pd.DataFrame([data_row]).to_csv(output_path, mode='a', header=False, index=False)
#             success_count += 1
#             print(f"   ↳ ✅ Success. Total new: {success_count}")
            
#         time.sleep(0.5)

# if __name__ == "__main__":
#     build_master_dataset()

# src/data/generate_raw_dataset.py
import lightkurve as lk
import numpy as np
import pandas as pd
import os
import time

def extract_transit_features(target_name, period, epoch):
    try:
        # Handle NaN values commonly found in false positive entries
        if pd.isna(period) or float(period) <= 0:
            period = 1.0  # Safe dummy fallback for phase folding non-transiting false positives
        else:
            period = float(period)

        if pd.isna(epoch):
            epoch = 0.0
        else:
            epoch = float(epoch)

        search_result = lk.search_lightcurve(target_name, author='SPOC', mission='TESS')
        if len(search_result) == 0:
            search_result = lk.search_lightcurve(target_name, author='Kepler')
            
        if len(search_result) == 0:
            return None
            
        lc = search_result[0].download()
        if lc is None:
            return None
        
        # Adjust JD offsets dynamically based on Lightkurve time format
        if epoch > 2400000:
            if lc.time.format == 'btjd': 
                epoch = epoch - 2457000
            elif lc.time.format == 'bkjd': 
                epoch = epoch - 2454833
        
        lc_flat = lc.flatten(window_length=901)
        lc_folded = lc_flat.fold(period=period, epoch_time=epoch)
        lc_binned = lc_folded.bin(time_bin_size=0.005)
        
        flux_array = lc_binned.flux.value
        flux_array = flux_array[~np.isnan(flux_array)]
        
        if len(flux_array) < 50:
            return None
            
        standardized_flux = np.interp(
            np.linspace(0, 1, 200), 
            np.linspace(0, 1, len(flux_array)), 
            flux_array
        )
        return standardized_flux
        
    except Exception as e:
        print(f"❌ Error on {target_name}: {str(e)}")
        return None

def build_master_dataset():
    catalog_path = "data/raw/full_stellar_catalog.csv"
    output_path = "data/processed/master_lightcurves.csv"
    failed_path = "data/processed/failed_targets.txt"
    
    if not os.path.exists(catalog_path):
        raise FileNotFoundError("Run parse_nasa_archive.py first!")
        
    catalog_df = pd.read_csv(catalog_path)
    processed_targets = set()
    
    # 1. Load Successful Stars
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        processed_targets.update(existing_df['target_name'].values)
        print(f"🔄 Found {len(processed_targets)} successful stars. Resuming...")
        
    # 2. Load Failed Stars (The Blacklist)
    if os.path.exists(failed_path):
        with open(failed_path, 'r') as f:
            failed_stars = set(f.read().splitlines())
            processed_targets.update(failed_stars)
            print(f"🚫 Skipping {len(failed_stars)} previously failed stars.")
    else:
        print("🚀 Starting fresh bulk download...")
        if not os.path.exists(output_path):
            columns = ['target_name', 'label'] + [f'flux_{i}' for i in range(200)]
            pd.DataFrame(columns=columns).to_csv(output_path, index=False)
    
    success_count = 0
    for _, row in catalog_df.iterrows():
        target_name = row['target_name']
        
        if target_name in processed_targets:
            continue 
            
        print(f"⏳ Fetching {target_name}...")
        flux_vector = extract_transit_features(target_name, row['period'], row['epoch'])
        
        processed_targets.add(target_name)
        
        if flux_vector is not None:    
            data_row = [target_name, row['label']] + flux_vector.tolist()
            pd.DataFrame([data_row]).to_csv(output_path, mode='a', header=False, index=False)
            success_count += 1
            print(f"   ↳ ✅ Success. Total new: {success_count}")
        else:
            # Add to Blacklist so we don't try standard retry again in this run
            with open(failed_path, 'a') as f:
                f.write(str(target_name) + '\n')
            
        time.sleep(0.5)

if __name__ == "__main__":
    build_master_dataset()