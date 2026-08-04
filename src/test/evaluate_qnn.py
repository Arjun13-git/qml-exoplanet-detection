import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve
)

# =====================================================================
# 1. QUANTUM CIRCUIT ARCHITECTURES & MODEL DEFINITION
# =====================================================================

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


class QuantumClassifier(nn.Module):
    def __init__(self, n_qubits=4, n_layers=3, architecture="data_reuploading"):
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


# =====================================================================
# 2. DATA LOADER FOR LATENT TEST SET (442 SAMPLES)
# =====================================================================

def load_qnn_test_dataloader(latent_type="cae", n_qubits=4, data_dir="data/processed/latent", batch_size=32):
    x_path = os.path.join(data_dir, f"X_test_{latent_type}_{n_qubits}.npy")
    y_path = os.path.join(data_dir, "y_test.npy")

    if not os.path.exists(x_path):
        raise FileNotFoundError(f"❌ Test features file not found: {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"❌ Test labels file not found: {y_path}")

    X_test = np.load(x_path)
    y_test = np.load(y_path)

    print(f"📖 Loaded Latent Test Data: {x_path} ({X_test.shape[0]} samples, {X_test.shape[1]} features)")

    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)


# =====================================================================
# 3. MODEL EVALUATION & DIAGNOSTICS PIPELINE
# =====================================================================

def evaluate_qnn_config(arch, latent_type, n_qubits=4, weights_dir="models/saved", fig_dir="reports/figures"):
    os.makedirs(fig_dir, exist_ok=True)
    model_label = f"QNN-{arch.upper()}-{latent_type.upper()}-{n_qubits}Q"
    weights_path = os.path.join(weights_dir, f"qnn_{arch}_{latent_type}_{n_qubits}q_weights.pth")

    if not os.path.exists(weights_path):
        print(f"⚠️ Skipping {model_label}: Weights file missing at {weights_path}")
        return None

    print(f"\n==================================================================")
    print(f"🚀 EVALUATING: {model_label}")
    print(f"📦 Checkpoint : {weights_path}")
    print(f"==================================================================")

    # 1. Load Data
    test_loader = load_qnn_test_dataloader(latent_type=latent_type, n_qubits=n_qubits)

    # 2. Instantiate Model & Load State Dict
    model = QuantumClassifier(n_qubits=n_qubits, n_layers=3, architecture=arch)
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    y_true, y_probs = [], []

    # 3. Perform Inference
    inference_start = time.time()
    with torch.no_grad():
        for bx, by in test_loader:
            logits = model(bx)
            probs = torch.sigmoid(logits).numpy()

            y_probs.extend(np.atleast_1d(probs))
            y_true.extend(by.numpy())

    infer_time = time.time() - inference_start
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    # 4. Compute Metrics & Youden's J Optimal Threshold
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    test_auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5

    y_preds_default = (y_probs >= 0.50).astype(int)
    y_preds_optimal = (y_probs >= best_threshold).astype(int)

    # 5. Output Diagnostics
    print(f"\n--- PROBABILITY DIAGNOSTICS ---")
    print(f"Probabilities Range : Min = {y_probs.min():.4f} | Max = {y_probs.max():.4f}")
    print(f"Probabilities Dist. : Mean = {y_probs.mean():.4f} | Median = {np.median(y_probs):.4f}")
    print(f"Optimal Threshold   : {best_threshold:.4f} (Youden's J Index)")
    print(f"Inference Time      : {infer_time:.2f}s ({infer_time / len(y_true) * 1000:.2f} ms/sample)")

    print(f"\n--- EVALUATION @ DEFAULT THRESHOLD (0.50) ---")
    print(classification_report(y_true, y_preds_default, target_names=["Non-Transit", "Exoplanet Transit"], zero_division=0))

    print(f"\n--- EVALUATION @ OPTIMAL THRESHOLD ({best_threshold:.4f}) ---")
    print(classification_report(y_true, y_preds_optimal, target_names=["Non-Transit", "Exoplanet Transit"], zero_division=0))
    print(f"ROC-AUC Score: {test_auc:.4f}\n" + "-"*65)

    # 6. Save Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_preds_optimal)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=["Non-Transit", "Transit"],
                yticklabels=["Non-Transit", "Transit"])
    plt.title(f"{model_label} Test CM (Thresh={best_threshold:.4f})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_path = os.path.join(fig_dir, f"test_cm_{arch}_{latent_type}_{n_qubits}q.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 7. Save ROC Curve Plot
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {test_auc:.4f}", color="purple", lw=2)
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.scatter(fpr[best_idx], tpr[best_idx], color="red", marker="o", label=f"Opt Thresh = {best_threshold:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_label} Test ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(fig_dir, f"test_roc_{arch}_{latent_type}_{n_qubits}q.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    print(f"📊 Figures saved:\n - {cm_path}\n - {roc_path}\n")
    return {
        "model": model_label,
        "auc": test_auc,
        "best_thresh": best_threshold,
        "cm_path": cm_path,
        "roc_path": roc_path
    }


# =====================================================================
# 4. MAIN SUITE FOR 4-QUBIT EVALUATION
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate 4-qubit QNN models on 442 holdout test samples.")
    parser.add_argument("--arch", type=str, choices=["all", "data_reuploading", "vqc"], default="all")
    parser.add_argument("--latent", type=str, choices=["all", "cae", "pca"], default="all")
    parser.add_argument("--qubits", type=int, default=4)
    args = parser.parse_args()

    architectures = ["data_reuploading", "vqc"] if args.arch == "all" else [args.arch]
    latents = ["cae", "pca"] if args.latent == "all" else [args.latent]

    print(f"🧪 Running Evaluation Suite for {args.qubits}-Qubit QNN Models")
    print(f"Architectures : {architectures}")
    print(f"Latent Types  : {latents}")

    summary = []
    for arch in architectures:
        for latent in latents:
            res = evaluate_qnn_config(arch=arch, latent_type=latent, n_qubits=args.qubits)
            if res:
                summary.append(res)

    if summary:
        print("\n" + "="*60)
        print(f"🎉 {args.qubits}-QUBIT QNN EVALUATION SUITE COMPLETE")
        print("="*60)
        for s in summary:
            print(f"• {s['model']:<25} | AUC: {s['auc']:.4f} | Optimal Thresh: {s['best_thresh']:.4f}")


if __name__ == "__main__":
    main()