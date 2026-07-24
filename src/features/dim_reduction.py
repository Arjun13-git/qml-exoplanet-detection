import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split


# --- 1. PyTorch 1D Convolutional Autoencoder Model ---
class ConvAutoencoder1D(nn.Module):
    def __init__(self, input_len=1000, latent_dim=8):
        super().__init__()
        self.input_len = input_len
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        
        # Calculate flattened dimension dynamically
        dummy_x = torch.zeros(1, 1, input_len)
        conv_out = self.encoder_conv(dummy_x)
        self.flatten_dim = conv_out.numel()
        self.spatial_len = conv_out.shape[-1]

        self.fc_encoder = nn.Linear(self.flatten_dim, latent_dim)
        
        # Decoder
        self.fc_decoder = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
        )

    def encode(self, x):
        features = self.encoder_conv(x)
        features = features.view(features.size(0), -1)
        latent = self.fc_encoder(features)
        return latent

    def decode(self, latent):
        x = self.fc_decoder(latent)
        x = x.view(x.size(0), 64, self.spatial_len)
        reconstruction = self.decoder_conv(x)
        # Handle small shape mismatches if input_len isn't divisible by 8
        if reconstruction.shape[-1] != self.input_len:
            reconstruction = nn.functional.interpolate(reconstruction, size=self.input_len)
        return reconstruction

    def forward(self, x):
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return latent, reconstruction


# --- 2. Training & Probe Evaluation Helpers ---
def train_autoencoder(model, train_loader, val_loader, epochs=35, lr=1e-3, device="cpu"):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            _, recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)

    return model


def evaluate_probe(X_train, y_train, X_test, y_test):
    """Linear probe test to evaluate signal separability on compressed features."""
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) > 1 else preds
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5
    return acc, auc


# --- 3. Main Pipeline ---
def main():
    data_path = "data/processed/master_lightcurves.csv"
    output_dir = "data/processed/latent"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print(f"Loading master dataset from {data_path}...")
    df = pd.read_csv(data_path)

    # Separate label and metadata
    target_col = 'label'
    metadata_cols = ['target_name', target_col]
    
    # Extract targets and convert to integer
    y = df[target_col].values.astype(int)

    # Extract time-series feature matrix X (flux values)
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    X = df[feature_cols].values

    print(f"Dataset successfully loaded.")
    print(f"Feature matrix shape: {X.shape} | Targets: {y.shape}")
    print(f"Class counts: {pd.Series(y).value_counts().to_dict()}\n")

    # Train / Test Split (80 / 20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize inputs
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    input_len = X_train_scaled.shape[1]
    target_dimensions = [4, 8, 16]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running compression experiments on device: {device}\n")

    results = []

    for d in target_dimensions:
        print(f"--- Compressing to {d} Qubits / Latent Features ---")

        # 1. PCA Reduction
        pca = PCA(n_components=d)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        X_test_pca_recon = pca.inverse_transform(X_test_pca)
        pca_mse = mean_squared_error(X_test_scaled, X_test_pca_recon)
        pca_acc, pca_auc = evaluate_probe(X_train_pca, y_train, X_test_pca, y_test)

        print(f"[PCA]   Dim: {d:2d} | Recon MSE: {pca_mse:.6f} | Probe Acc: {pca_acc*100:.2f}% | Probe AUC: {pca_auc:.4f}")

        # Save PCA latent arrays
        np.save(os.path.join(output_dir, f"X_train_pca_{d}.npy"), X_train_pca)
        np.save(os.path.join(output_dir, f"X_test_pca_{d}.npy"), X_test_pca)

        # 2. 1D ConvAutoencoder Reduction
        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)
        X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)

        train_loader = DataLoader(TensorDataset(X_train_t), batch_size=64, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test_t), batch_size=64, shuffle=False)

        cae = ConvAutoencoder1D(input_len=input_len, latent_dim=d)
        cae = train_autoencoder(cae, train_loader, test_loader, epochs=35, lr=1e-3, device=device)

        cae.eval()
        with torch.no_grad():
            X_train_cae = cae.encode(X_train_t.to(device)).cpu().numpy()
            X_test_cae = cae.encode(X_test_t.to(device)).cpu().numpy()
            _, X_test_cae_recon = cae(X_test_t.to(device))
            cae_mse = mean_squared_error(X_test_scaled, X_test_cae_recon.squeeze(1).cpu().numpy())

        cae_acc, cae_auc = evaluate_probe(X_train_cae, y_train, X_test_cae, y_test)

        print(f"[1D-CAE] Dim: {d:2d} | Recon MSE: {cae_mse:.6f} | Probe Acc: {cae_acc*100:.2f}% | Probe AUC: {cae_auc:.4f}\n")

        # Save CAE latent arrays
        np.save(os.path.join(output_dir, f"X_train_cae_{d}.npy"), X_train_cae)
        np.save(os.path.join(output_dir, f"X_test_cae_{d}.npy"), X_test_cae)

        results.extend([
            {"method": "PCA", "dim": d, "recon_mse": pca_mse, "probe_acc": pca_acc, "probe_auc": pca_auc},
            {"method": "1D-CAE", "dim": d, "recon_mse": cae_mse, "probe_acc": cae_acc, "probe_auc": cae_auc}
        ])

    # Save target arrays for downstream QML ingestion
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    # Save comparison log
    summary_df = pd.DataFrame(results)
    summary_df.to_csv("logs/dim_reduction_comparison.csv", index=False)
    print("✅ Dimensionality reduction complete. Latent features saved in `data/processed/latent/`.")

if __name__ == "__main__":
    main()