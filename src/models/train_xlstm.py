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

# 1. Setup File Paths
os.makedirs("logs", exist_ok=True)
run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_txt_log = f"logs/xlstm_train_{run_timestamp}.txt"
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

# 3. sLSTM Cell with Exponential Gating & Memory Stabiliser
class sLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Linear projections for Input (i), Forget (f), Cell (z), Output (o)
        self.w_i = nn.Linear(input_size, hidden_size)
        self.u_i = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.w_f = nn.Linear(input_size, hidden_size)
        self.u_f = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.w_z = nn.Linear(input_size, hidden_size)
        self.u_z = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.w_o = nn.Linear(input_size, hidden_size)
        self.u_o = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x_t, states):
        h_prev, c_prev, n_prev, m_prev = states
        
        # Compute raw gate inputs
        i_tilde = self.w_i(x_t) + self.u_i(h_prev)
        f_tilde = self.w_f(x_t) + self.u_f(h_prev)
        z_t = torch.tanh(self.w_z(x_t) + self.u_z(h_prev))
        o_tilde = self.w_o(x_t) + self.u_o(h_prev)
        
        # Exponential Stabiliser state: m_t = max(f_tilde + m_prev, i_tilde)
        m_t = torch.maximum(f_tilde + m_prev, i_tilde)
        
        # Stabilized Exponential Gates
        i_t = torch.exp(i_tilde - m_t)
        f_t = torch.exp(f_tilde + m_prev - m_t)
        o_t = torch.sigmoid(o_tilde)
        
        # Memory & Normalizer Updates
        c_t = f_t * c_prev + i_t * z_t
        n_t = f_t * n_prev + i_t
        
        # Stabilized Hidden State Calculation
        h_t = o_t * (c_t / (n_t + 1e-6))
        
        return h_t, (h_t, c_t, n_t, m_t)

class xLSTMBlock(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.layer1 = sLSTMCell(input_size, hidden_size)
        self.layer2 = sLSTMCell(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # x shape: (B, T) or (B, T, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            
        B, T, _ = x.shape
        device = x.device
        
        # Init states for Layer 1
        h1 = torch.zeros(B, self.hidden_size, device=device)
        c1 = torch.zeros(B, self.hidden_size, device=device)
        n1 = torch.zeros(B, self.hidden_size, device=device)
        m1 = torch.zeros(B, self.hidden_size, device=device)
        state1 = (h1, c1, n1, m1)
        
        # Init states for Layer 2
        h2 = torch.zeros(B, self.hidden_size, device=device)
        c2 = torch.zeros(B, self.hidden_size, device=device)
        n2 = torch.zeros(B, self.hidden_size, device=device)
        m2 = torch.zeros(B, self.hidden_size, device=device)
        state2 = (h2, c2, n2, m2)
        
        outputs = []
        for t in range(T):
            x_t = x[:, t, :]
            out1, state1 = self.layer1(x_t, state1)
            out1 = self.dropout(out1)
            out2, state2 = self.layer2(out1, state2)
            outputs.append(out2)
            
        seq_out = torch.stack(outputs, dim=1) # (B, T, hidden_size)
        return self.norm(seq_out)

class xLSTMClassifier(nn.Module):
    def __init__(self, seq_len=200, hidden_size=64, dropout=0.2):
        super().__init__()
        self.xlstm = xLSTMBlock(input_size=1, hidden_size=hidden_size, num_layers=2, dropout=dropout)
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_size, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        features = self.xlstm(x) # (B, T, hidden_size)
        features = features.transpose(1, 2) # (B, hidden_size, T)
        return self.head(features)

def save_benchmark_logs(model_name, metrics):
    """Appends xLSTM metrics to master summary TXT and CSV files."""
    
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

    model = xLSTMClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    epochs = 30
    start_time = time.time()

    logging.info("🚀 Training Extended LSTM (xLSTM) Baseline Model...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device).float()
            optimizer.zero_grad()
            out = model(x_b).squeeze(1)
            loss = criterion(out, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

    logging.info("==== 🏆 FINAL xLSTM TEST RESULTS ====")
    logging.info(f"Accuracy:  {acc*100:.2f}%")
    logging.info(f"Precision: {prec:.4f}")
    logging.info(f"Recall:    {rec:.4f}")
    logging.info(f"F1-Score:  {f1:.4f}")
    logging.info(f"ROC-AUC:   {auc:.4f}")

    save_benchmark_logs("xLSTM", {
        "acc": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
        "train_loss": train_loss, "val_loss": val_loss, "test_loss": test_loss,
        "train_time": total_train_time
    })

if __name__ == "__main__":
    train()