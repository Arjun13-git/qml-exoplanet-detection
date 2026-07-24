import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class ExoplanetLightcurveDataset(Dataset):
    def __init__(self, csv_file, augment=False):
        """
        Args:
            csv_file (str): Path to the processed master_lightcurves.csv
            augment (bool): Whether to apply data augmentation
        """
        self.data = pd.read_csv(csv_file)
        
        # Based on your extraction script: 
        # Col 0: target_name, Col 1: label, Col 2 to 201: flux values
        self.labels = self.data.iloc[:, 1].values
        self.features = self.data.iloc[:, 2:].values
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract flux array as float32
        flux = self.features[idx].astype(np.float32)
        label = int(self.labels[idx])

        if self.augment:
            # 1. Gaussian Noise (std dev of 0.001 - adjust based on flux scaling)
            noise = np.random.normal(0, 0.001, flux.shape)
            flux = flux + noise
            
            # 2. Random Roll (Shift the transit event by up to 10 points left/right)
            shift = np.random.randint(-10, 11)
            flux = np.roll(flux, shift)

        # Convert to PyTorch tensors
        flux_tensor = torch.tensor(flux, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return flux_tensor, label_tensor

# --- Verification Block ---
if __name__ == "__main__":
    csv_path = 'data/processed/master_lightcurves.csv'
    
    print("🚀 Initializing Dataset with Augmentation...")
    dataset = ExoplanetLightcurveDataset(csv_file=csv_path, augment=True)
    
    print(f"Total samples loaded: {len(dataset)}")
    
    # Initialize DataLoader
    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Fetch one batch
    features, labels = next(iter(dataloader))
    
    print("\n✅ Verification Successful:")
    print(f"Batch Features shape: {features.shape}")  # Expected: [16, 200]
    print(f"Batch Labels shape:   {labels.shape}")    # Expected: [16]
    print(f"First 5 labels:       {labels[:5].tolist()}")