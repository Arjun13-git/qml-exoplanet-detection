import os
import time
import logging
import csv
from datetime import datetime
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# Ensure root directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.dataset import prepare_dataloaders

# ---------------------------------------------------------------------------
# 1. SETUP FILE PATHS & DUAL LOGGING
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
os.makedirs("models/saved", exist_ok=True)

run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_txt_log = f"logs/xenopulse_net_train_{run_timestamp}.txt"
master_txt_summary = "logs/classical_benchmark_summary.txt"
master_csv_summary = "logs/classical_benchmark_results.csv"
model_save_path = "models/saved/xenopulse_net_weights.pth"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(run_txt_log),
        logging.StreamHandler(sys.stdout)
    ]
)

# ---------------------------------------------------------------------------
# 2. ARCHITECTURE DEFINITION
# ---------------------------------------------------------------------------

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
    XenoPulse-Net: Synthesizes all classical baseline paradigms.
    """
    def __init__(self, in_channels=1, seq_len=200, num_classes=1):
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

# ---------------------------------------------------------------------------
# 3. BENCHMARK LOGGING UTILITY
# ---------------------------------------------------------------------------

def save_benchmark_logs(model_name, metrics):
    """Appends XenoPulse-Net metrics to master summary TXT and CSV files."""
    
    # A. Master TXT Summary
    with open(master_txt_summary, "a") as f:
        f.write("=" * 65 + "\n")
        f.write(f"MODEL: {model_name} | TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 65 + "\n")
        f.write(f"Test Accuracy:  {metrics['acc']*100:.2f}%\n")
        f.write(f"Precision:      {metrics['precision']:.4f}\n")
        f.write(f"Recall:         {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:       {metrics['f1']:.4f}\n")
        f.write(f"ROC-AUC:        {metrics['auc']:.4f}\n")
        f.write(f"Train Loss:     {metrics['train_loss']:.4f}\n")
        f.write(f"Val Loss:       {metrics['val_loss']:.4f}\n")
        f.write(f"Test Loss:      {metrics['test_loss']:.4f}\n")
        f.write(f"Train Time:     {metrics['train_time']:.2f} seconds\n")
        f.write(f"Weights Saved:  {model_save_path}\n")
        f.write("=" * 65 + "\n\n")

    # B. Master CSV Matrix
    file_exists = os.path.exists(master_csv_summary)
    headers = [
        "timestamp", "model_name", "test_acc", "precision", 
        "recall", "f1_score", "roc_auc", "train_loss", "val_loss", "test_loss", "train_time_sec"
    ]
    
    with open(master_csv_summary, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_name,
            f"{metrics['acc']:.4f}",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['f1']:.4f}",
            f"{metrics['auc']:.4f}",
            f"{metrics['train_loss']:.4f}",
            f"{metrics['val_loss']:.4f}",
            f"{metrics['test_loss']:.4f}",
            f"{metrics['train_time']:.2f}"
        ])
        
    logging.info(f"📝 Master TXT log updated : {master_txt_summary}")
    logging.info(f"📊 Master CSV log updated : {master_csv_summary}")
    logging.info(f"💾 Checkpoint Saved To    : {model_save_path}")

# ---------------------------------------------------------------------------
# 4. TRAINING & EVALUATION PIPELINE
# ---------------------------------------------------------------------------

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    csv_path = '/home/arjunshenoy13/qml-exoplanet-detection/data/processed/master_lightcurves.csv'
    train_loader, val_loader, test_loader, _, _ = prepare_dataloaders(csv_path=csv_path)

    model = XenoPulseNet(in_channels=1, seq_len=200, num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

    epochs = 30
    best_val_loss = float('inf')
    start_time = time.time()

    logging.info("🚀 Training XenoPulse-Net Model...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device).float()
            optimizer.zero_grad()
            out = model(x_b).squeeze(1)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_b)
            
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device).float()
                out = model(x_b).squeeze(1)
                loss = criterion(out, y_b)
                val_loss += loss.item() * len(y_b)
        val_loss /= len(val_loader.dataset)
        
        scheduler.step()

        # Checkpoint Saving: Save model state when validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} 🔥 (Saved Checkpoint)")
        else:
            logging.info(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    total_train_time = time.time() - start_time

    # Load best checkpoint for evaluation
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        logging.info(f"Loaded best weights from {model_save_path} for evaluation.")

    # Testing Evaluation
    model.eval()
    test_loss = 0.0
    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b, y_b = x_b.to(device), y_b.to(device).float()
            out = model(x_b).squeeze(1)
            loss = criterion(out, y_b)
            test_loss += loss.item() * len(y_b)
            probs = torch.sigmoid(out)
            preds = (probs >= 0.5).long()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_b.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary')
    auc = roc_auc_score(all_targets, all_probs)

    logging.info("==== 🏆 FINAL XenoPulse-Net TEST RESULTS ====")
    logging.info(f"Accuracy:  {acc*100:.2f}%")
    logging.info(f"Precision: {prec:.4f}")
    logging.info(f"Recall:    {rec:.4f}")
    logging.info(f"F1-Score:  {f1:.4f}")
    logging.info(f"ROC-AUC:   {auc:.4f}")

    save_benchmark_logs("XenoPulse-Net", {
        "acc": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
        "train_loss": train_loss, "val_loss": val_loss, "test_loss": test_loss,
        "train_time": total_train_time
    })

if __name__ == "__main__":
    train()