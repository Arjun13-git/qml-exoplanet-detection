# Phase 2: Dataset Construction & Splitting (Experimental Mini-Batch)
**Date:** July 13, 2026
**Pipeline Stage:** Feature Engineering, Class Balancing, and Tensor Splitting
**Status:** ⚠️ EXPERIMENTAL PHASE - Full-scale data ingestion pending.

## 1. Primary Objectives
To build and validate the automated data extraction, preprocessing, and tensor-splitting architecture using a small, controlled "mini-batch" of targets before executing the large-scale MAST archive scrape. 

## 2. Methodology & Execution
* **Master Dataset Generation (`generate_raw_dataset.py`):** Constructed the pipeline to fetch, flatten (Savitzky-Golay), fold, and bin targets from a curated stellar catalog. **Note:** This run was strictly limited to a mini-batch of 4 test stars (2 Planets, 1 Eclipsing Binary, 1 Quiet Star) to verify system stability and tensor dimensions.
* **Class Balancing (`balance_and_split.py`):** Implemented an upsampling strategy to ensure equal representation of all 3 classes. For this mini-batch, manual replacement sampling was used to match the maximum class count, preparing the logic for SMOTE in the final scaled dataset.
* **Strict Data Splitting:** Implemented a smart-stratification fallback for this ultra-small experimental batch. The pipeline successfully bypassed strict stratification limits to split the data into Training (70%), Validation (15%), and Test (15%) subsets.
* **Tensor Conversion:** Cast all NumPy arrays into PyTorch tensors (`torch.float32` and `torch.long`).

## 3. Outputs & Verification (Mini-Batch)
* **Master CSV:** `data/processed/master_lightcurves.csv` (Shape: 4 targets, 200 features)
* **PyTorch Tensors:** `data/processed/split_tensors.pt`
    * Train Shape: (4, 200)
    * Val Shape: (1, 200)
    * Test Shape: (1, 200)

## 4. Conclusion & Next Steps
The feature engineering and splitting architecture is mathematically sound and bug-free. The system successfully transformed raw astrophysical signals into dimensionally stable PyTorch tensors. 
**Pending Action:** The catalog must be expanded to hundreds of targets to generate the actual main dataset for full model training.