# Phase 1: Astrophysical Data Ingestion & Preprocessing
**Date:** July 12, 2026
**Target System:** TIC 260128333 (WASP-121b)
**Pipeline Stage:** Data Acquisition and Signal Cleaning

## 1. Primary Objectives
To programmatically retrieve, clean, and compress space telescope time-series data for exoplanet transit detection, ensuring physical signals are amplified while instrumental and stellar noise are mathematically suppressed.

## 2. Methodology & Execution

* **Programmatic Download:** Utilized the official NASA `lightkurve` API to fetch high-cadence (2-minute) observation sectors directly from the Mikulski Archive for Space Telescopes (MAST). Limited the fetch to 2 sectors to ensure network stability and avoid data bloat during the testing phase.
* **Presearch Data Conditioning (PDC):** Specified the `SPOC` (Science Processing Operations Center) author during API retrieval. This inherently applies NASA's supercomputing pipeline to the raw data, scrubbing spacecraft thruster noise, thermal anomalies, and cosmic ray interference (extracting the `PDCSAP_FLUX` array).
* **Savitzky-Golay Filtering:** Applied the `flatten(window_length=901)` function to the stitched light curve. This polynomial filter successfully modeled and removed low-frequency stellar activity (e.g., starspot rotation) without eroding the high-frequency planetary transit signal, normalizing the baseline flux to 1.0.
* **Phase Folding & Binning:** Folded the flattened time-series over the known orbital period of WASP-121b (1.274925 days) at the reference epoch (1492.25). This stacked all transit events to maximize the Signal-to-Noise Ratio (SNR). Finally, applied a binning function (`time_bin_size=0.005`) to compress the dimensionality of the time-series array, preparing it for the memory constraints of future Quantum Machine Learning (QML) embeddings.

## 3. Outputs & Verification
* **Raw Data Extract:** `data/raw/TIC 260128333_raw.csv` 
* **Raw Diagnostic Plot:** `data/raw/TIC_260128333_raw_plot.png`
* **Processed/Folded Diagnostic Plot:** `data/processed/TIC_260128333_processed_plot.png`

## 4. Conclusion
Phase 1 successfully isolated the U-shaped transit signature of WASP-121b. The data is now properly formatted, dimensionally reduced, and mathematically clean enough to act as a baseline input for both classical Deep Learning and Quantum Neural Network architectures.