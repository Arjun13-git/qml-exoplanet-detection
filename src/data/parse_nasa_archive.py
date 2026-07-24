# src/data/parse_nasa_archive.py
import pandas as pd
import os

def build_real_catalog():
    print("🔭 Parsing Official NASA Exoplanet Archive...")
    
    nasa_csv_path = "data/raw/nasa_planetary_systems.csv"
    if not os.path.exists(nasa_csv_path):
        raise FileNotFoundError(f"❌ Please download the Planetary Systems CSV from NASA and place it at {nasa_csv_path}")
        
    df = pd.read_csv(nasa_csv_path, comment='#', low_memory=False)
    
    if 'tran_flag' in df.columns:
        transiting = df[df['tran_flag'] == 1].copy()
    elif 'discoverymethod' in df.columns:
        transiting = df[df['discoverymethod'] == 'Transit'].copy()
    else:
        raise KeyError("Could not find a column to identify transiting planets.")
    
    transiting = transiting.dropna(subset=['pl_orbper', 'pl_tranmid', 'hostname'])
    
    # THE FIX: Drop duplicate stars so we get UNIQUE targets
    unique_stars = transiting.drop_duplicates(subset=['hostname'])
    
    catalog = []
    
    # Now grab exactly 1500 UNIQUE stars (or more, if you want!)
    for _, row in unique_stars.head(1500).iterrows(): 
        catalog.append({
            "target_name": str(row['hostname']),
            "period": row['pl_orbper'],
            "epoch": row['pl_tranmid'],
            "label": 1 
        })
        
    final_df = pd.DataFrame(catalog)
    
    os.makedirs("data/raw", exist_ok=True)
    output_path = "data/raw/full_stellar_catalog.csv"
    final_df.to_csv(output_path, index=False)
    
    print(f"✅ Extracted {len(final_df)} UNIQUE transiting exoplanet systems.")
    print(f"💾 Saved publication-ready catalog to {output_path}")

if __name__ == "__main__":
    build_real_catalog()