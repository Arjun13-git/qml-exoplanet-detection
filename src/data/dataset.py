import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class ExoplanetLightcurveDataset(Dataset):
    def __init__(self, features, labels, augment=False):
        """
        Args:
            features (np.ndarray): Scaled flux features [N, 200]
            labels (np.ndarray): Binary labels [N]
            augment (bool): Apply data augmentation (Train only!)
        """
        self.features = features
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        flux = self.features[idx].copy().astype(np.float32)
        label = int(self.labels[idx])

        if self.augment:
            # 1. Gaussian Noise
            noise = np.random.normal(0, 0.001, flux.shape)
            flux += noise
            
            # 2. Random Roll Shift (±10 points)
            shift = np.random.randint(-10, 11)
            flux = np.roll(flux, shift)

        return torch.tensor(flux, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def prepare_dataloaders(
    csv_path='data/processed/master_lightcurves.csv',
    batch_size=32,
    feature_range=(0, np.pi)  # [0, pi] for Quantum Angle Encoding
):
    df = pd.read_csv(csv_path)
    
    # Col 1: label, Col 2..201: flux_0 to flux_199
    labels = df.iloc[:, 1].values
    features = df.iloc[:, 2:].values

    # 1. Stratified Train / Test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.20, random_state=42, stratify=labels
    )
    
    # 2. Stratified Train / Val (80/10/10 final ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, random_state=42, stratify=y_train
    )

    # 3. Fit Scaler ONLY on Training Set
    scaler = MinMaxScaler(feature_range=feature_range)
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 4. Create Datasets (Augment TRAIN only)
    train_ds = ExoplanetLightcurveDataset(X_train, y_train, augment=True)
    val_ds   = ExoplanetLightcurveDataset(X_val, y_val, augment=False)
    test_ds  = ExoplanetLightcurveDataset(X_test, y_test, augment=False)

    # 5. Build DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Calculate class weight for BCE loss: count(neg) / count(pos)
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32)

    print(f" Dataset Split Complete:")
    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"   Calculated pos_weight for BCE Loss: {pos_weight.item():.4f}")

    return train_loader, val_loader, test_loader, scaler, pos_weight