# src/features/balance_and_split.py
import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def load_and_split_dataset(csv_path="data/processed/master_lightcurves.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Could not find {csv_path}. Please run generate_raw_dataset.py first!")

    print("📖 Loading processed master dataset...")
    df = pd.read_csv(csv_path)
    
    if 'label' not in df.columns:
        raise KeyError("❌ 'label' column is missing from master_lightcurves.csv!")

    # Drop rows with missing labels and ensure integer dtype
    df = df.dropna(subset=['label'])
    y = df['label'].astype(int).values

    # --- CLEAN & EXTRACT FEATURE MATRIX ---
    # Identify non-feature metadata columns to drop
    id_candidates = ['tic_id', 'kepid', 'id', 'target_id', 'disposition', 'source', 'object_id']
    drop_cols = [col for col in id_candidates + ['label'] if col in df.columns]
    
    feature_df = df.drop(columns=drop_cols)

    # Force all feature columns to numeric, replace unparseable strings with 0.0, and cast to float32
    feature_df = feature_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    X = feature_df.values.astype(np.float32)
    
    print(f"📊 Original class distribution: {np.bincount(y)}")

    # --- HANDLING CLASS IMBALANCE ---
    max_class_count = max(np.bincount(y))
    balanced_X_list, balanced_y_list = [], []
    
    for class_label in np.unique(y):
        X_c = X[y == class_label]
        y_c = y[y == class_label]
        
        if len(X_c) < max_class_count:
            X_c_resampled, y_c_resampled = resample(
                X_c, y_c, 
                replace=True, 
                n_samples=max_class_count, 
                random_state=42
            )
            balanced_X_list.append(X_c_resampled)
            balanced_y_list.append(y_c_resampled)
        else:
            balanced_X_list.append(X_c)
            balanced_y_list.append(y_c)
            
    X_balanced = np.vstack(balanced_X_list).astype(np.float32)
    y_balanced = np.concatenate(balanced_y_list).astype(int)
    print(f"⚖️ Balanced class distribution: {np.bincount(y_balanced)}")

    # --- STRICT DATA SPLIT: 70% Train / 15% Val / 15% Test ---
    total_samples = len(y_balanced)
    num_classes = len(np.unique(y_balanced))
    
    temp_size = int(np.ceil(total_samples * 0.30))
    should_stratify = y_balanced if temp_size >= num_classes else None
    
    if should_stratify is None:
        print("⚠️ Mini-batch too small for safe stratification. Proceeding with standard split.")

    # Step 1: Separate Train (70%) from Temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_balanced, y_balanced, test_size=0.30, random_state=42, stratify=should_stratify
    )
    
    # Step 2: Split remaining 30% into Val (15%) and Test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=None
    )
    
    print(f"🔒 Data splits verified:")
    print(f"   ↳ Train shape: {X_train.shape} (70%)")
    print(f"   ↳ Val shape:   {X_val.shape} (15%)")
    print(f"   ↳ Test shape:  {X_test.shape} (15%)")

    # --- CONVERSION TO PYTORCH TENSORS ---
    tensors = {
        "X_train": torch.from_numpy(X_train),
        "y_train": torch.from_numpy(y_train).long(),
        "X_val": torch.from_numpy(X_val),
        "y_val": torch.from_numpy(y_val).long(),
        "X_test": torch.from_numpy(X_test),
        "y_test": torch.from_numpy(y_test).long()
    }
    
    os.makedirs("data/processed", exist_ok=True)
    torch.save(tensors, "data/processed/split_tensors.pt")
    print("💾 PyTorch tensors successfully saved to data/processed/split_tensors.pt")

if __name__ == "__main__":
    load_and_split_dataset()