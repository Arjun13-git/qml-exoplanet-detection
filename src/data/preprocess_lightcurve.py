# src/data/preprocess_lightcurve.py
import lightkurve as lk
import matplotlib.pyplot as plt
import os

def preprocess_and_plot(tic_id, period, t0):
    print(f"🧹 Fetching and cleaning data for {tic_id}...")
    
    # We can pull directly from the MAST archive again (lightkurve caches it locally so it's instant)
    search_result = lk.search_lightcurve(tic_id, author='SPOC', mission='TESS', cadence='fast')
    lc_collection = search_result[:2].download_all() # Using 2 sectors for a clean, fast fold
    lc = lc_collection.stitch()
    
    print("📉 Flattening the light curve (Removing stellar noise)...")
    # window_length determines the flexibility of the Savitzky-Golay filter
    lc_flat = lc.flatten(window_length=901)
    
    print(f"⏱️ Phase folding the data (Period: {period} days)...")
    # Fold the data based on the known planetary orbital period and reference time (epoch)
    lc_folded = lc_flat.fold(period=period, epoch_time=t0)
    
    # Bin the data to compress it for the Quantum model
    print("🗜️ Binning data to reduce dimensionality...")
    lc_binned = lc_folded.bin(time_bin_size=0.005)
    
    print("📊 Generating preprocessed plot...")
    plt.figure(figsize=(10, 6))
    
    # Plot both folded and binned data to see the improvement
    plt.plot(lc_folded.time.value, lc_folded.flux.value, color='gray', marker='.', 
             linestyle='none', markersize=1, alpha=0.2, label='Folded (Unbinned)')
    plt.plot(lc_binned.time.value, lc_binned.flux.value, color='blue', marker='o', 
             linestyle='none', markersize=3, alpha=0.8, label='Binned (Quantum-Ready)')
    
    plt.title(f"Processed TESS Light Curve - {tic_id} (WASP-121b)", fontsize=14)
    plt.xlabel("Phase", fontsize=12)
    plt.ylabel("Normalized Flux", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs("data/processed", exist_ok=True)
    output_plot = "data/processed/TIC_260128333_processed_plot.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"✅ Preprocessed plot successfully saved to: {output_plot}")
    plt.show()

if __name__ == "__main__":
    target_star = "TIC 260128333"
    # Known parameters for WASP-121b
    orbital_period = 1.274925 # in days
    epoch = 1492.25 # Approximate TESS reference epoch for this transit
    
    preprocess_and_plot(target_star, orbital_period, epoch)