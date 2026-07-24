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
run_txt_log = f"logs/nimamba2_train_{run_timestamp}.txt"
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

# 3. Neural Implicit Fourier Feature Kernel
class ImplicitFourierKernel(nn.Module):
    def __init__(self, in_features=1, hidden_dim=64, num_frequencies=8):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(num_frequencies) * 2.0)
        self.mlp = nn.Sequential(
            nn.Linear(num_frequencies * 2 + in_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x):
        B, T, C = x.shape
        t_coords = torch.linspace(0, 1, T, device=x.device).view(1, T, 1).expand(B, -1, -1)
        
        args = t_coords * self.freqs.unsqueeze(0).unsqueeze(0) * 2 * 3.141592653589793
        fourier_feat = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        feat = torch.cat([x, fourier_feat], dim=-1)
        return self.mlp(feat)

# 4. NiMamba-2 Block (State Space Duality + Neural Implicit Feature Kernel)
class NiMamba2Block(nn.Module):
    def __init__(self, d_in=1, d_model=64, d_state=16, headdim=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.headdim = headdim
        self.nheads = d_model // headdim
        
        self.implicit_encoder = ImplicitFourierKernel(in_features=d_in, hidden_dim=d_model)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        
        self.in_proj = nn.Linear(d_model, d_model * 2 + self.nheads * d_state)
        self.dt_proj = nn.Linear(d_model, self.nheads)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.nheads + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.nheads, self.headdim))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            
        B, T, _ = x.shape
        x_imp = self.implicit_encoder(x)
        
        x_conv = self.conv1d(x_imp.transpose(1, 2)).transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        proj = self.in_proj(x_conv)
        u, gate, B_ssm = torch.split(proj, [self.d_model, self.d_model, self.nheads * self.d_state], dim=-1)
        
        u = u.view(B, T, self.nheads, self.headdim)
        B_ssm = B_ssm.view(B, T, self.nheads, self.d_state)
        
        dt = F.softplus(self.dt_proj(x_conv))
        A = -torch.exp(self.A_log)
        
        states = torch.zeros(B, self.nheads, self.headdim, self.d_state, device=x.device)
        ys = []
        
        for t in range(T):
            dt_t = dt[:, t, :].unsqueeze(-1).unsqueeze(-1)
            u_t = u[:, t, :, :]
            B_t = B_ssm[:, t, :, :]
            
            dA = torch.exp(dt_t * A.view(1, self.nheads, 1, 1))
            dB_u = torch.einsum('bhk,bhn->bhkn', u_t, B_t)
            states = states * dA + dt_t * dB_u
            
            y_t = states.sum(dim=-1) + self.D.unsqueeze(0) * u_t
            ys.append(y_t)
            
        y = torch.stack(ys, dim=1).view(B, T, self.d_model)
        out = self.out_proj(y * F.silu(gate))
        return out

class NiMamba2Classifier(nn.Module):
    def __init__(self, seq_len=200, d_model=64, d_state=16):
        super().__init__()
        self.block1 = NiMamba2Block(d_in=1, d_model=d_model, d_state=d_state)
        self.block2 = NiMamba2Block(d_in=d_model, d_model=d_model, d_state=d_state)
        self.norm = nn.LayerNorm(d_model)
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out = self.norm(out1 + out2)
        out = out.transpose(1, 2)
        return self.head(out)

def save_benchmark_logs(model_name, metrics):
    """Appends NiMamba-2 metrics to master summary TXT and CSV files."""
    
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

    model = NiMamba2Classifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    epochs = 30
    start_time = time.time()

    logging.info("🚀 Training Neural Implicit Mamba-2 (NiMamba-2) Baseline Model...")

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

    logging.info("==== 🏆 FINAL NiMamba-2 TEST RESULTS ====")
    logging.info(f"Accuracy:  {acc*100:.2f}%")
    logging.info(f"Precision: {prec:.4f}")
    logging.info(f"Recall:    {rec:.4f}")
    logging.info(f"F1-Score:  {f1:.4f}")
    logging.info(f"ROC-AUC:   {auc:.4f}")

    save_benchmark_logs("NiMamba-2", {
        "acc": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
        "train_loss": train_loss, "val_loss": val_loss, "test_loss": test_loss,
        "train_time": total_train_time
    })

if __name__ == "__main__":
    train()