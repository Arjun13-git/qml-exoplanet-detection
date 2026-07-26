import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
import pennylane as qml

# ----------------------------------------------------------------------
# Dual stdout/file logging wrapper for .txt logs
# ----------------------------------------------------------------------
class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log_file = open(filepath, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


# ----------------------------------------------------------------------
# 1. GPU / Execution Device Selection
# ----------------------------------------------------------------------
def get_execution_hardware(n_qubits):
    """
    Sets up both PyTorch CUDA device and PennyLane lightning.gpu device.
    """
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        dev = qml.device("lightning.gpu", wires=n_qubits)
        diff_method = "adjoint"
        qml_device_str = "PennyLane lightning.gpu"
    except Exception as e:
        qml_device_str = f"default.qubit (Fallback: {e})"
        dev = qml.device("default.qubit", wires=n_qubits)
        diff_method = "backprop"

    print(f"PyTorch Execution Device:   {torch_device}")
    print(f"PennyLane Simulation Device: {qml_device_str}")
    
    return torch_device, dev, diff_method


# ----------------------------------------------------------------------
# 2. QCNN Unitary Blocks
# ----------------------------------------------------------------------
def conv_block(weights, w1, w2):
    """2-Qubit Quantum Convolutional Unitary Gate."""
    qml.Rot(*weights[0:3], wires=w1)
    qml.Rot(*weights[3:6], wires=w2)
    qml.CNOT(wires=[w1, w2])
    qml.Rot(*weights[6:9], wires=w1)
    qml.Rot(*weights[9:12], wires=w2)


def pool_block(weights, source, target):
    """2-Qubit Quantum Pooling Unitary (maps state feature from source -> target)."""
    qml.Rot(*weights[0:3], wires=source)
    qml.Rot(*weights[3:6], wires=target)
    qml.CRZ(weights[6], wires=[source, target])
    qml.CRY(weights[7], wires=[source, target])
    qml.Rot(*weights[8:11], wires=target)


# ----------------------------------------------------------------------
# 3. PyTorch QCNN Module
# ----------------------------------------------------------------------
class QCNNModel(nn.Module):
    def __init__(self, n_qubits, dev, diff_method):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_levels = int(np.log2(n_qubits))

        # Shared parameter kernels across translationally invariant channels
        self.conv_weights = nn.Parameter(torch.randn(self.n_levels, 12) * 0.1)
        self.pool_weights = nn.Parameter(torch.randn(self.n_levels, 11) * 0.1)

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def qcnn_circuit(inputs, conv_w, pool_w):
            # Input Feature Encoding
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))

            active_wires = list(range(self.n_qubits))

            # Hierarchical Tapering (Conv -> Pool)
            for lvl in range(self.n_levels):
                # 1. Convolution (Even Pairs)
                for i in range(0, len(active_wires), 2):
                    conv_block(conv_w[lvl], active_wires[i], active_wires[i + 1])

                # Convolution (Odd/Shifted Pairs for Ring Topology)
                if len(active_wires) > 2:
                    for i in range(1, len(active_wires) - 1, 2):
                        conv_block(conv_w[lvl], active_wires[i], active_wires[i + 1])
                    conv_block(conv_w[lvl], active_wires[-1], active_wires[0])

                # 2. Pooling (Reduce active qubits by half)
                sources = active_wires[0::2]
                targets = active_wires[1::2]
                for s, t in zip(sources, targets):
                    pool_block(pool_w[lvl], s, t)

                active_wires = targets

            # Measure final output expectation value on remaining qubit
            return qml.expval(qml.PauliZ(active_wires[0]))

        self.qcircuit = qcnn_circuit

    def forward(self, x):
        outputs = [self.qcircuit(x[i], self.conv_weights, self.pool_weights) for i in range(x.shape[0])]
        return torch.stack(outputs)


# ----------------------------------------------------------------------
# 4. Training & Benchmark Runner
# ----------------------------------------------------------------------
def run_qcnn_experiment(n_qubits, latent_type, epochs=30, batch_size=32, lr=0.01):
    print(f"\n{'='*50}")
    print(f"Training {n_qubits}-QUBIT QCNN | Latent: {latent_type.upper()}")
    print(f"{'='*50}")

    DATA_DIR = "data/processed/latent"

    X_train = np.load(os.path.join(DATA_DIR, f"X_train_{latent_type}_{n_qubits}.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, f"X_test_{latent_type}_{n_qubits}.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    torch_device, dev, diff_method = get_execution_hardware(n_qubits)
    
    # Explicitly place model onto GPU
    model = QCNNModel(n_qubits, dev, diff_method).to(torch_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            # Move tensors to GPU explicitly
            batch_x = batch_x.to(torch_device)
            batch_y = batch_y.to(torch_device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        epoch_loss = total_loss / len(train_ds)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {epoch_loss:.6f}")

    training_time = time.time() - start_time

    # Evaluation
    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(torch_device)
            logits = model(batch_x)
            
            # Transfer back to CPU for numpy metric calculations
            all_logits.extend(logits.cpu().numpy())
            all_targets.extend(batch_y.numpy())

    probs = 1 / (1 + np.exp(-np.array(all_logits)))
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(all_targets, preds)
    roc_auc = roc_auc_score(all_targets, probs)
    precision = precision_score(all_targets, preds, zero_division=0)
    recall = recall_score(all_targets, preds, zero_division=0)
    f1 = f1_score(all_targets, preds, zero_division=0)

    print(f"\n{'-'*50}")
    print(f"Results for {n_qubits}-QUBIT QCNN (Latent: {latent_type.upper()}):")
    print(f"  Execution Time: {training_time:.2f}s")
    print(f"  Accuracy:       {acc*100:.2f}%")
    print(f"  ROC-AUC:        {roc_auc:.4f}")
    print(f"  Precision:      {precision:.4f}")
    print(f"  Recall:         {recall:.4f}")
    print(f"  F1-Score:       {f1:.4f}")
    print(f"{'-'*50}")

    return {
        "architecture": "qcnn",
        "n_qubits": n_qubits,
        "latent_type": latent_type,
        "accuracy": acc,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "training_time_sec": training_time,
    }


# ----------------------------------------------------------------------
# 5. Main Suite Execution & File Logging
# ----------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)

    txt_log_path = "logs/qcnn_benchmark.txt"
    sys.stdout = DualLogger(txt_log_path)

    results = []

    qubit_configs = [8, 16]
    latent_sources = ["cae", "pca"]

    for n_q in qubit_configs:
        for latent in latent_sources:
            res = run_qcnn_experiment(n_qubits=n_q, latent_type=latent, epochs=30, batch_size=32)
            results.append(res)

    csv_log_path = "logs/qcnn_benchmark_results.csv"
    df_results = pd.DataFrame(results)
    df_results.to_csv(csv_log_path, index=False)

    print("\n" + "=" * 60)
    print("✅ ALL QCNN BENCHMARKS COMPLETED!")
    print(f"Detailed Epoch Logs saved to: {txt_log_path}")
    print(f"Summary Metrics CSV saved to: {csv_log_path}")
    print("=" * 60)
    print("\n", df_results)