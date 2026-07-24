import os
import time
import logging
import csv
from datetime import datetime
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# Ensure root directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.dataset import prepare_dataloaders

# 1. Setup File Paths
os.makedirs("logs", exist_ok=True)
run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_txt_log = f"logs/tcn_train_{run_timestamp}.txt"
master_txt_summary = "logs/classical_benchmark_summary.txt"
master_csv_summary = "logs/classical_benchmark_results.csv"

# 2. Setup Dual Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(run_txt_log),
        logging.StreamHandler(sys.stdout)
    ]
)

# 3. TCN Components
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )

    def forward(self, x):
        out = self.conv(x)
        if self.padding != 0:
            out = out[:, :, :-self.padding]
        return out

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, num_inputs=1, num_channels=[32, 64, 128], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, dilation=dilation_size, dropout=dropout)
            )

        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out = self.tcn(x)
        out = self.pool(out).squeeze(-1)
        return self.fc(out)

def save_benchmark_logs(model_name, metrics):
    """Appends TCN metrics to master summary TXT and CSV files."""
    
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
        
    logging.info(f"📝 Master TXT log updated: {master_txt_summary}")
    logging.info(f"📊 Master CSV log updated: {master_csv_summary}")

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    csv_path = '/home/arjunshenoy13/qml-exoplanet-detection/data/processed/master_lightcurves.csv'
    train_loader, val_loader, test_loader, _, _ = prepare_dataloaders(csv_path=csv_path)

    model = TCNModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    epochs = 30
    start_time = time.time()

    logging.info("🚀 Training Temporal Convolutional Network (TCN)...")

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
        scheduler.step(val_loss)

        logging.info(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    total_train_time = time.time() - start_time

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

    logging.info("==== 🏆 FINAL TCN TEST RESULTS ====")
    logging.info(f"Accuracy:  {acc*100:.2f}%")
    logging.info(f"Precision: {prec:.4f}")
    logging.info(f"Recall:    {rec:.4f}")
    logging.info(f"F1-Score:  {f1:.4f}")
    logging.info(f"ROC-AUC:   {auc:.4f}")

    save_benchmark_logs("TCN", {
        "acc": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
        "train_loss": train_loss, "val_loss": val_loss, "test_loss": test_loss,
        "train_time": total_train_time
    })

if __name__ == "__main__":
    train()