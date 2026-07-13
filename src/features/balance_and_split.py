# src/features/balance_and_split.py
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import os

def load_and_split_dataset(csv_path="data/processed/master_lightcurves.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Could not find {csv_path}. Please run generate_raw_dataset.py first!")

    print("📖 Loading processed master dataset...")
    df = pd.read_csv(csv_path)
    
    # Extract features (flux arrays) and targets (labels)
    X = df.drop(columns=['tic_id', 'label']).values
    y = df['label'].values
    
    print(f"📊 Original class distribution: {np.bincount(y)}")

    # --- HANDLING CLASS IMBALANCE (Experimental Strategy) ---
    # For our mini-batch experiment, we will upsample minority classes manually
    # to match the maximum class count, ensuring perfect balance.
    max_class_count = max(np.bincount(y))
    balanced_X_list, balanced_y_list = [], []
    
    for class_label in np.unique(y):
        X_c = X[y == class_label]
        y_c = y[y == class_label]
        
        if len(X_c) < max_class_count:
            # Upsample with replacement to match the highest count class
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
            
    X_balanced = np.vstack(balanced_X_list)
    y_balanced = np.concatenate(balanced_y_list)
    print(f"⚖️ Balanced class distribution: {np.bincount(y_balanced)}")

    # --- STRICT DATA SPLIT: 70% Train / 15% Val / 15% Test ---
    # Smart handling for ultra-small experimental mini-batches
    total_samples = len(y_balanced)
    num_classes = len(np.unique(y_balanced))
    
    # Calculate initial temp size (30% of total)
    temp_size = int(np.ceil(total_samples * 0.30))
    
    # Safe stratification rule: only stratify if test allocation can physically hold all classes
    should_stratify = y_balanced if temp_size >= num_classes else None
    if should_stratify is None:
        print("⚠️ Mini-batch too small for safe stratification. Proceeding with standard split.")

    # Step 1: Separate Train (70%) from the rest (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_balanced, y_balanced, test_size=0.30, random_state=42, stratify=should_stratify
    )
    
    # Step 2: Split the remaining 30% equally into Val (15%) and Test (15%)
    # For a mini-batch of 6, X_temp will have 2 samples. We split 50/50 -> 1 Val, 1 Test.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=None
    )
    
    print(f"🔒 Data splits verified:")
    print(f"   ↳ Train shape: {X_train.shape} (70%)")
    print(f"   ↳ Val shape:   {X_val.shape} (15%)")
    print(f"   ↳ Test shape:  {X_test.shape} (15%)")

    # --- CONVERSION TO PYTORCH TENSORS ---
    # PyTorch expects floats for inputs and long/int64 for classification targets
    tensors = {
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.long),
        "X_val": torch.tensor(X_val, dtype=torch.float32),
        "y_val": torch.tensor(y_val, dtype=torch.long),
        "X_test": torch.tensor(X_test, dtype=torch.float32),
        "y_test": torch.tensor(y_test, dtype=torch.long)
    }
    
    # Save the tensors inside the processed folder as a ready-to-load dictionary
    os.makedirs("data/processed", exist_ok=True)
    torch.save(tensors, "data/processed/split_tensors.pt")
    print("💾 PyTorch tensors successfully saved to data/processed/split_tensors.pt")

if __name__ == "__main__":
    load_and_split_dataset()