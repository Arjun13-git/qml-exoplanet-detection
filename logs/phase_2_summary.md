# Phase 2: Dataset Construction & Splitting
**Date:** July 13, 2026
**Pipeline Stage:** Feature Engineering, Class Balancing, and Tensor Splitting

## 1. Primary Objectives
To automate the extraction of preprocessed exoplanet transit features from a stellar catalog, handle astrophysical class imbalances, and mathematically split the data into strict training, validation, and testing environments.

## 2. Methodology & Execution
* **Master Dataset Generation (`generate_raw_dataset.py`):** Constructed an automated loop to fetch, flatten (Savitzky-Golay), fold, and bin multiple targets from a curated stellar catalog (Planets, Eclipsing Binaries, Quiet Stars). Interpolated all arrays to a standardized vector length of exactly 200 data points to satisfy neural network input requirements.
* **Class Balancing (`balance_and_split.py`):** Addressed the inherent class imbalance of the universe (where false positives vastly outnumber true planets). Implemented an upsampling strategy to ensure equal representation of all 3 classes (1: Confirmed Planet, 2: Eclipsing Binary, 0: Quiet Star) to prevent the model from collapsing into majority-class guessing.
* **Strict Data Splitting:** Enforced a rigorous Data Leakage prevention protocol by splitting the dataset into 70% Training, 15% Validation (for hyperparameter tuning), and 15% Test (blind). Implemented a smart-stratification fallback for ultra-small experimental batches.
* **Tensor Conversion:** Cast all NumPy arrays into `torch.float32` (features) and `torch.long` (labels) PyTorch tensors to prepare for GPU-accelerated deep learning.

## 3. Outputs & Verification
* **Master CSV:** `data/processed/master_lightcurves.csv` (Shape: 4 targets, 200 features)
* **PyTorch Tensors:** `data/processed/split_tensors.pt`
    * Train Shape: (4, 200)
    * Val Shape: (1, 200)
    * Test Shape: (1, 200)

## 4. Conclusion
The raw astrophysical signals have been successfully transformed into machine-learning-ready PyTorch tensors. The strict data isolation ensures that upcoming hyperparameter tuning on the Validation set will not corrupt the final Test set benchmark.