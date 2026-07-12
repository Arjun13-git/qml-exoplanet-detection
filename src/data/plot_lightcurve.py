# src/data/plot_lightcurve.py
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_raw_data(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Error: The file {file_path} does not exist.")
        return

    print("📖 Reading raw light curve data...")
    # Load the data, skipping the metadata header lines that lightkurve creates
    df = pd.read_csv(file_path, comment='#')
    
    # Standard TESS light curves use 'time' and 'flux'
    if 'time' not in df.columns or 'flux' not in df.columns:
        print("❌ Error: CSV format unexpected. Missing 'time' or 'flux' columns.")
        return

    print("📊 Generating diagnostic plot...")
    plt.figure(figsize=(12, 6))
    plt.plot(df['time'], df['flux'], color='black', marker='.', markersize=0.5, linestyle='none', alpha=0.3)
    
    plt.title("Raw TESS Light Curve - Target: TIC 260128333 (WASP-121b)", fontsize=14)
    plt.xlabel("Time (BJD - 2457000)", fontsize=12)
    plt.ylabel("Normalized Flux", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Ensure the directory exists and save the image
    os.makedirs("data/raw", exist_ok=True)
    output_plot = "data/raw/TIC_260128333_raw_plot.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"✅ Diagnostic plot successfully saved to: {output_plot}")
    plt.show()

if __name__ == "__main__":
    raw_csv_path = "data/raw/TIC 260128333_raw.csv"
    plot_raw_data(raw_csv_path)