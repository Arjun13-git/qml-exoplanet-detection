import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score
)


# --- 1. Dual Logger Helper (Console + Text File) ---
LOG_FILE = "logs/qnn_benchmark.txt"

def log_msg(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


# --- 2. PennyLane Quantum Circuit Architectures ---
def create_qnn_circuit(n_qubits, n_layers, architecture="data_reuploading"):
    dev = qml.device("default.qubit", wires=n_qubits)

    if architecture == "data_reuploading":
        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            for l in range(n_layers):
                # Data Re-uploading step
                for i in range(n_qubits):
                    qml.RY(inputs[..., i], wires=i)
                    qml.RZ(inputs[..., i], wires=i)
                
                # Parameterized rotation layers
                for i in range(n_qubits):
                    qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)
                
                # Entangling layer (Ring topology)
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
                    
            return qml.expval(qml.PauliZ(0))

    elif architecture == "vqc":
        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            # Angle Encoding (Single load)
            for i in range(n_qubits):
                qml.RY(inputs[..., i], wires=i)

            # Strongly Entangling Layers
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])

            return qml.expval(qml.PauliZ(0))

    return circuit


# --- 3. PyTorch Wrapper for PennyLane QNode ---
class QuantumClassifier(nn.Module):
    def __init__(self, n_qubits, n_layers=3, architecture="data_reuploading"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.qcircuit = create_qnn_circuit(n_qubits, n_layers, architecture)

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qnode_layer = qml.qnn.TorchLayer(self.qcircuit, weight_shapes)
        self.head = nn.Linear(1, 1)

    def forward(self, x):
        x_scaled = torch.tanh(x) * np.pi
        q_out = self.qnode_layer(x_scaled)
        
        if q_out.ndim == 1:
            q_out = q_out.unsqueeze(-1)
            
        logits = self.head(q_out)
        return logits.squeeze(-1)


# --- 4. Training & Evaluation Helper ---
def train_eval_qnn(n_qubits, arch, latent_type, epochs=30, lr=0.01, batch_size=32):
    data_dir = "data/processed/latent"
    
    X_train = np.load(os.path.join(data_dir, f"X_train_{latent_type}_{n_qubits}.npy"))
    X_test = np.load(os.path.join(data_dir, f"X_test_{latent_type}_{n_qubits}.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = QuantumClassifier(n_qubits=n_qubits, n_layers=3, architecture=arch)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    log_msg(f"\n==================================================")
    log_msg(f"Training {arch.upper()} | Qubits: {n_qubits} | Latent: {latent_type.upper()}")
    log_msg(f"==================================================")
    
    start_time = time.time()

    # Train for exactly 30 epochs, printing every single epoch
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * bx.size(0)

        epoch_loss = running_loss / len(train_ds)
        log_msg(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {epoch_loss:.6f}")

    elapsed = time.time() - start_time

    # Evaluation
    model.eval()
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for bx, _ in test_loader:
            logits = model(bx)
            probs = torch.sigmoid(logits).numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)

    acc = accuracy_score(y_test, all_preds)
    auc = roc_auc_score(y_test, all_probs) if len(np.unique(y_test)) > 1 else 0.5
    prec = precision_score(y_test, all_preds, zero_division=0)
    rec = recall_score(y_test, all_preds, zero_division=0)
    f1 = f1_score(y_test, all_preds, zero_division=0)

    log_msg(f"\n--------------------------------------------------")
    log_msg(f"Results for {arch.upper()} ({n_qubits} Qubits, Latent: {latent_type.upper()}):")
    log_msg(f"  Execution Time: {elapsed:.2f}s")
    log_msg(f"  Accuracy:       {acc*100:.2f}%")
    log_msg(f"  ROC-AUC:        {auc:.4f}")
    log_msg(f"  Precision:      {prec:.4f}")
    log_msg(f"  Recall:         {rec:.4f}")
    log_msg(f"  F1-Score:       {f1:.4f}")
    log_msg(f"--------------------------------------------------\n")

    return {
        "architecture": arch,
        "n_qubits": n_qubits,
        "latent_type": latent_type,
        "accuracy": acc,
        "roc_auc": auc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "training_time_sec": elapsed
    }


# --- 5. Main Benchmark Suite ---
def main():
    os.makedirs("logs", exist_ok=True)
    
    # Initialize / overwrite previous text log header
    with open(LOG_FILE, "w") as f:
        f.write("=== QNN BENCHMARK RUN LOG ===\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    results = []

    # Benchmark Configurations (4 & 8 qubits, CAE & PCA latent features)
    qubit_configs = [4, 8]
    architectures = ["data_reuploading", "vqc"]
    latent_sources = ["cae", "pca"]

    for qubits in qubit_configs:
        for source in latent_sources:
            for arch in architectures:
                res = train_eval_qnn(
                    n_qubits=qubits, 
                    arch=arch, 
                    latent_type=source, 
                    epochs=30, 
                    lr=0.01
                )
                results.append(res)

    # Save CSV benchmark summary
    summary_df = pd.DataFrame(results)
    csv_path = "logs/qnn_benchmark_results.csv"
    summary_df.to_csv(csv_path, index=False)

    log_msg("="*60)
    log_msg("✅ ALL QNN BENCHMARKS COMPLETED!")
    log_msg(f"Detailed epoch text logs saved to: {LOG_FILE}")
    log_msg(f"Summary metrics CSV saved to:      {csv_path}")
    log_msg("="*60)
    log_msg("\n" + summary_df.to_string(index=False))


if __name__ == "__main__":
    main()