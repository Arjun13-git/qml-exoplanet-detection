import os
import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# =====================================================================
# 1. ARCHITECTURE DEFINITION
# =====================================================================

class DilatedResidualBlock1D(nn.Module):
    """
    Synthesizes 1D ResNet, 1D CNN, and TCN Dilated Convolutions.
    Provides local spatial inductive bias to eliminate majority class collapse.
    """
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=5, 
            padding=2 * dilation, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=5, 
            padding=2 * dilation, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.gelu(out + residual)
        return out


class SpectralPeriodicityModule(nn.Module):
    """
    Inspired by TimesNet: Uses 1D Real FFT to extract periodic orbital frequencies.
    """
    def __init__(self, channels):
        super().__init__()
        self.freq_conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        fft_res = torch.fft.rfft(x, dim=-1)
        magnitude = torch.abs(fft_res)
        freq_feat = F.gelu(self.bn(self.freq_conv(magnitude)))
        complex_feat = torch.complex(freq_feat, torch.zeros_like(freq_feat))
        time_reconstructed = torch.fft.irfft(complex_feat, n=x.shape[-1], dim=-1)
        return x + time_reconstructed


class GatedSSMBlock(nn.Module):
    """
    Inspired by Selective-SSM (Mamba-1/2), NiMamba-2, xLSTM, and BiLSTM:
    Gated State-Space Sequence Block for long-range context.
    """
    def __init__(self, channels, hidden_dim=64):
        super().__init__()
        self.in_proj = nn.Conv1d(channels, hidden_dim * 2, kernel_size=1)
        self.gru = nn.GRU(
            input_size=hidden_dim, 
            hidden_size=hidden_dim // 2, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        self.out_proj = nn.Conv1d(hidden_dim, channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        x_proj = self.in_proj(x)
        x_signal, x_gate = x_proj.chunk(2, dim=1)
        x_seq = x_signal.transpose(1, 2)
        gru_out, _ = self.gru(x_seq)
        x_ssm = gru_out.transpose(1, 2)
        gated_out = x_ssm * torch.sigmoid(x_gate)
        out = self.norm(self.out_proj(gated_out))
        return F.gelu(out + residual)


class InvertedChannelAttention(nn.Module):
    """
    Inspired by iTransformer & PatchTST:
    Applies Multi-Head Attention across feature channels.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x_trans = x.transpose(1, 2)
        attn_out, _ = self.mha(x_trans, x_trans, x_trans)
        out = self.norm(x_trans + attn_out)
        return out.transpose(1, 2)


class XenoPulseNet(nn.Module):
    """
    XenoPulse-Net: Novel Custom Architecture synthesizing all 11 baseline paradigms.
    """
    def __init__(self, in_channels=1, seq_len=201, num_classes=1):
        super().__init__()
        
        self.stem_in = nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3)
        self.res_block1 = DilatedResidualBlock1D(32, 64, dilation=1)
        self.res_block2 = DilatedResidualBlock1D(64, 128, dilation=2)
        self.res_block3 = DilatedResidualBlock1D(128, 128, dilation=4)
        
        self.spectral_module = SpectralPeriodicityModule(128)
        
        self.ssm_layer1 = GatedSSMBlock(128, hidden_dim=128)
        self.ssm_layer2 = GatedSSMBlock(128, hidden_dim=128)
        
        self.attention_head = InvertedChannelAttention(channels=128, num_heads=4)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = F.gelu(self.stem_in(x))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        x = self.spectral_module(x)
        
        x = self.ssm_layer1(x)
        x = self.ssm_layer2(x)
        
        x = self.attention_head(x)
        
        avg_p = self.global_avg_pool(x).squeeze(-1)
        max_p = self.global_max_pool(x).squeeze(-1)
        pooled = torch.cat([avg_p, max_p], dim=1)
        
        logits = self.classifier(pooled)
        return logits


# =====================================================================
# 2. DATA LOADING & EPOCH-BY-EPOCH TRAINING
# =====================================================================

DATASET_PATH = "/home/arjunshenoy13/qml-exoplanet-detection/data/processed/master_lightcurves.csv"
LOG_TXT_PATH = "logs/xenopulse_net.txt"
LOG_CSV_PATH = "logs/classical_benchmark_results.csv"

def log_message(msg):
    print(msg)
    os.makedirs("logs", exist_ok=True)
    with open(LOG_TXT_PATH, "a") as f:
        f.write(msg + "\n")

def load_master_dataset(path):
    log_message(f"📂 Reading master CSV: {path}")
    df = pd.read_csv(path)
    
    possible_targets = ['label', 'LABEL', 'target', 'Target', 'y', 'class', 'Class']
    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break
            
    if target_col is None:
        target_col = df.columns[0]
        log_message(f"⚠️ Target column not explicitly named. Defaulting to first column: '{target_col}'")
    else:
        log_message(f"🎯 Target column detected: '{target_col}'")

    ignore_cols = [target_col, 'id', 'ID', 'kepid', 'tic_id', 'source_id', 'Unnamed: 0', 'name', 'Name']
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    
    X_df = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    X = X_df.values.astype(np.float32)
    
    # z-score normalization per sample to stabilize gradients
    means = np.mean(X, axis=1, keepdims=True)
    stds = np.std(X, axis=1, keepdims=True) + 1e-8
    X = (X - means) / stds

    y_raw = pd.to_numeric(df[target_col], errors='coerce').fillna(0).values
    unique_labels = np.unique(y_raw)
    if set(unique_labels) == {1, 2}:
        y_raw = y_raw - 1
        
    y = y_raw.astype(np.float32)

    log_message(f"📊 Dataset Loaded: {X.shape[0]} samples, sequence length = {X.shape[1]}")
    log_message(f"⚖️ Class distribution: Positives (1) = {int(np.sum(y == 1))}, Negatives (0) = {int(np.sum(y == 0))}")
    return X, y

def train_xenopulse_on_master():
    os.makedirs("logs", exist_ok=True)
    with open(LOG_TXT_PATH, "w") as f:
        f.write("=== XenoPulse-Net Master Dataset Benchmark ===\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"🚀 Execution Device: {device}")

    X, y = load_master_dataset(DATASET_PATH)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=32, shuffle=False)

    seq_len = X_train.shape[1]
    model = XenoPulseNet(in_channels=1, seq_len=seq_len, num_classes=1).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_message(f"🧠 Total Model Parameters: {total_params:,}")

    epochs = 30
    log_message(f"🏋️ Starting Training for {epochs} Epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x).squeeze(-1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)

        scheduler.step()
        train_loss /= len(train_loader.dataset)

        # Log EVERY single epoch explicitly
        log_message(f"Epoch [{epoch:02d}/{epochs}] - Loss: {train_loss:.6f}")

    total_training_time = time.time() - start_time
    log_message(f"⏱️ Training Completed in {total_training_time:.2f} seconds.")

    # Evaluation
    model.eval()
    all_preds = []
    all_targets = []

    inference_start = time.time()
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x).squeeze(-1)
            probs = torch.sigmoid(logits)

            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(batch_y.numpy())

    total_inference_time = time.time() - inference_start
    avg_inference_time_ms = (total_inference_time / len(y_val)) * 1000

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    binary_preds = (all_preds >= 0.5).astype(int)

    acc = accuracy_score(all_targets, binary_preds)
    roc_auc = roc_auc_score(all_targets, all_preds)
    f1 = f1_score(all_targets, binary_preds, zero_division=0)
    precision = precision_score(all_targets, binary_preds, zero_division=0)
    recall = recall_score(all_targets, binary_preds, zero_division=0)

    log_message("\n==========================================")
    log_message("📊 Real Dataset Evaluation Metrics")
    log_message("==========================================")
    log_message(f"Accuracy:        {acc:.4f}")
    log_message(f"ROC-AUC:         {roc_auc:.4f}")
    log_message(f"F1-Score:        {f1:.4f}")
    log_message(f"Precision:       {precision:.4f}")
    log_message(f"Recall:          {recall:.4f}")
    log_message(f"Inference Speed: {avg_inference_time_ms:.2f} ms/sample")
    log_message("==========================================")

    file_exists = os.path.exists(LOG_CSV_PATH)
    with open(LOG_CSV_PATH, mode='a', newline='') as csv_file:
        fieldnames = ['model', 'accuracy', 'roc_auc', 'f1_score', 'precision', 'recall', 'inference_time_ms', 'parameters']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'model': 'XenoPulse-Net',
            'accuracy': f"{acc:.4f}",
            'roc_auc': f"{roc_auc:.4f}",
            'f1_score': f"{f1:.4f}",
            'precision': f"{precision:.4f}",
            'recall': f"{recall:.4f}",
            'inference_time_ms': f"{avg_inference_time_ms:.2f}",
            'parameters': total_params
        })

    log_message(f"✅ Results successfully appended to {LOG_CSV_PATH} and {LOG_TXT_PATH}")

if __name__ == "__main__":
    train_xenopulse_on_master()