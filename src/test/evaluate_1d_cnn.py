import os
import sys
import argparse
import inspect
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Ensure repository root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.dataset import prepare_dataloaders


def load_test_dataloader(
    csv_path="data/processed/master_lightcurves.csv", 
    tensor_path="data/processed/split_tensors.pt", 
    batch_size=32
):
    """
    Loads test dataset via prepare_dataloaders to enforce exact z-score normalization.
    """
    if os.path.exists(csv_path):
        print(f"📖 Loading test dataset via prepare_dataloaders from {csv_path}...")
        _, _, test_loader, _, _ = prepare_dataloaders(csv_path=csv_path, batch_size=batch_size)
        return test_loader
    elif os.path.exists(tensor_path):
        print(f"⚠️ {csv_path} not found. Falling back to {tensor_path}...")
        tensors = torch.load(tensor_path)
        return DataLoader(TensorDataset(tensors["X_test"], tensors["y_test"]), batch_size=batch_size, shuffle=False)
    else:
        raise FileNotFoundError(f"❌ Could not find {csv_path} or {tensor_path}.")


def load_1d_cnn_model(weights_path, device="cpu"):
    """
    Imports and instantiates Conv1DBaseline dynamically.
    """
    model_cls = None
    module_candidates = [
        ("src.models.train_1d_cnn", ["Conv1DBaseline", "Conv1DCNN", "CNN1DBaseline"]),
        ("src.models.conv1d", ["Conv1DBaseline", "Conv1DCNN", "CNN1DBaseline"])
    ]

    for mod_path, class_names in module_candidates:
        try:
            mod = __import__(mod_path, fromlist=["*"])
            for cls_name in class_names:
                if hasattr(mod, cls_name):
                    model_cls = getattr(mod, cls_name)
                    print(f"🤖 Loaded '{cls_name}' from {mod_path}")
                    break
            if model_cls:
                break
        except ImportError:
            continue

    if model_cls is None:
        for mod_name in ["src.models.train_1d_cnn", "src.models.conv1d"]:
            try:
                mod = __import__(mod_name, fromlist=["*"])
                classes = [
                    cls for name, cls in inspect.getmembers(mod, inspect.isclass)
                    if issubclass(cls, nn.Module) and cls.__module__ == mod.__name__
                ]
                for cls in classes:
                    if "Conv" in cls.__name__ or "CNN" in cls.__name__:
                        model_cls = cls
                        print(f"🤖 Auto-detected class '{model_cls.__name__}' from {mod_name}")
                        break
                if model_cls:
                    break
            except ImportError:
                continue

    if model_cls is None:
        raise AttributeError("❌ No PyTorch nn.Module class found for 1D CNN.")

    model = model_cls()

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"❌ Weights file not found at: {weights_path}")

    print(f"📦 Loading weights from {weights_path}...")
    checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def evaluate_1d_cnn(
    model, 
    test_loader, 
    device="cuda" if torch.cuda.is_available() else "cpu",
    output_fig_dir="reports/figures"
):
    os.makedirs(output_fig_dir, exist_ok=True)

    y_true, y_probs = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)

            outputs = model(inputs)

            if outputs.ndim > 1 and outputs.shape[-1] == 1:
                outputs = outputs.squeeze(-1)

            if outputs.ndim == 1:
                probs = torch.sigmoid(outputs).cpu().numpy()
            else:
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

            y_probs.extend(np.atleast_1d(probs))
            y_true.extend(targets.cpu().numpy() if isinstance(targets, torch.Tensor) else targets)

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    # Calculate optimal decision threshold using Youden's J statistic
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    test_auc = roc_auc_score(y_true, y_probs)

    y_preds_default = (y_probs >= 0.50).astype(int)
    y_preds_optimal = (y_probs >= best_threshold).astype(int)

    # Output Diagnostics
    print(f"\n==================== 1D CNN - PROBABILITY DIAGNOSTICS ====================")
    print(f"Probabilities Range   : Min = {y_probs.min():.4f} | Max = {y_probs.max():.4f}")
    print(f"Probabilities Dist.   : Mean = {y_probs.mean():.4f} | Median = {np.median(y_probs):.4f}")
    print(f"Optimal Threshold     : {best_threshold:.4f} (Youden's J Index)")

    # Classification Metrics
    print(f"\n==================== EVALUATION @ DEFAULT THRESHOLD (0.50) ====================")
    print(classification_report(y_true, y_preds_default, target_names=["Non-Transit", "Exoplanet Transit"], zero_division=0))

    print(f"\n==================== EVALUATION @ OPTIMAL THRESHOLD ({best_threshold:.4f}) ====================")
    print(classification_report(y_true, y_preds_optimal, target_names=["Non-Transit", "Exoplanet Transit"], zero_division=0))
    print(f"Test Set ROC-AUC Score: {test_auc:.4f}\n" + "="*70)

    # Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_preds_optimal)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Transit", "Transit"],
                yticklabels=["Non-Transit", "Transit"])
    plt.title(f"1D CNN CM (Thresh={best_threshold:.4f})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_path = os.path.join(output_fig_dir, "test_cm_1d_cnn.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # ROC Curve Plot
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {test_auc:.4f}", color="darkorange", lw=2)
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.scatter(fpr[best_idx], tpr[best_idx], color="red", marker="o", label=f"Opt Thresh = {best_threshold:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("1D CNN Test ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(output_fig_dir, "test_roc_1d_cnn.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    print(f"📊 Evaluation plots saved to:\n - {cm_path}\n - {roc_path}")
    return test_auc


def main():
    parser = argparse.ArgumentParser(description="Evaluate 1D CNN model on holdout test set.")
    parser.add_argument("--weights", type=str, default="models/saved/1d_cnn_weights.pth")
    parser.add_argument("--data", type=str, default="data/processed/master_lightcurves.csv")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Running evaluation on: {device}")

    try:
        test_loader = load_test_dataloader(csv_path=args.data)
        print(f"✅ Successfully loaded test dataset with {len(test_loader.dataset)} samples.")
        
        model = load_1d_cnn_model(weights_path=args.weights, device=device)
        evaluate_1d_cnn(
            model=model, 
            test_loader=test_loader, 
            device=device
        )
    except Exception as e:
        print(f"\n❌ 1D CNN Evaluation failed: {e}")


if __name__ == "__main__":
    main()