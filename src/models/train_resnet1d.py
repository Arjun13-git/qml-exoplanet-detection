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
run_txt_log = f"logs/resnet1d_train_{run_timestamp}.txt"
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

# 3. 1D ResNet Architecture
class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        return self.relu(out)

class ResNet1D(nn.Module):
    def __init__(self, input_len=200):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.layer1 = ResBlock1D(32, 32, stride=1)
        self.layer2 = ResBlock1D(32, 64, stride=2)
        self.layer3 = ResBlock1D(64, 128, stride=2)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out = self.prep(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        return self.fc(out)

def save_benchmark_logs(model_name, metrics):
    """Appends ResNet1D metrics to master summary TXT and CSV files."""
    
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

    model = ResNet1D().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    epochs = 30
    start_time = time.time()

    logging.info("🚀 Training 1D ResNet Baseline Model...")

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

    logging.info("==== 🏆 FINAL 1D RESNET TEST RESULTS ====")
    logging.info(f"Accuracy:  {acc*100:.2f}%")
    logging.info(f"Precision: {prec:.4f}")
    logging.info(f"Recall:    {rec:.4f}")
    logging.info(f"F1-Score:  {f1:.4f}")
    logging.info(f"ROC-AUC:   {auc:.4f}")

    save_benchmark_logs("ResNet1D", {
        "acc": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
        "train_loss": train_loss, "val_loss": val_loss, "test_loss": test_loss,
        "train_time": total_train_time
    })

if __name__ == "__main__":
    train()