# src/data/stellar_catalog.py

# Class Labels:
# 1 = Confirmed Exoplanet
# 2 = Eclipsing Binary (False Positive)
# 0 = No Transit (Quiet Star)

STELLAR_CATALOG = {
    # --- CONFIRMED PLANETS (Class 1) ---
    "TIC_260128333": {"tic_id": "TIC 260128333", "period": 1.274925, "epoch": 1492.25, "label": 1}, # WASP-121b
    "TIC_281541555": {"tic_id": "TIC 281541555", "period": 3.288800, "epoch": 1327.44, "label": 1}, # WASP-126b
    
    # --- ECLIPSING BINARIES (Class 2) ---
    # Eclipsing binaries also have periods! We must fold them to teach the model the V-shape difference.
    "TIC_55652896": {"tic_id": "TIC 55652896", "period": 3.693, "epoch": 1326.1, "label": 2}, 
    
    # --- QUIET STARS / NO TRANSIT (Class 0) ---
    # We assign a dummy period to these just so the pipeline processes them in the exact same dimension.
    "TIC_149603524": {"tic_id": "TIC 149603524", "period": 2.5, "epoch": 1330.0, "label": 0} 
}