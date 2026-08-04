import os
import sys
import argparse
import inspect
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Ensure root directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.dataset import prepare_dataloaders


def load_test_dataloader(
    csv_path="data/processed/master_lightcurves.csv", 
    tensor_path="data/processed/split_tensors.pt", 
    batch_size=32
):
    """
    Loads test data using prepare_dataloaders to ensure identical z-score normalization.
    Falls back to split_tensors.pt if CSV is unavailable.
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


def instantiate_model_from_module(module, module_label):
    classes = [
        cls for name, cls in inspect.getmembers(module, inspect.isclass)
        if issubclass(cls, torch.nn.Module) and cls.__module__ == module.__name__
    ]
    if not classes:
        raise AttributeError(f"No PyTorch nn.Module class found in src/models/{module_label}.py")
    
    main_classes = [c for c in classes if not c.__name__.endswith(("Block", "Layer", "Embedding", "Head"))]
    target_cls = main_classes[0] if main_classes else classes[0]
    
    print(f"🤖 Auto-detected class '{target_cls.__name__}' from src/models/{module_label}.py")
    return target_cls()


def load_model_from_checkpoint(model_name, weights_path, device="cpu"):
    model_name_lower = model_name.lower().replace("-", "_").replace(" ", "_")
    
    try:
        if "qnn" in model_name_lower or "res_qnn" in model_name_lower:
            from src.models import hybrid_qnn
            model = instantiate_model_from_module(hybrid_qnn, "hybrid_qnn")
        elif "astronet" in model_name_lower:
            from src.models import astronet
            model = instantiate_model_from_module(astronet, "astronet")
        elif "cnn" in model_name_lower:
            from src.models import classical_cnn
            model = instantiate_model_from_module(classical_cnn, "classical_cnn")
        elif "ssm" in model_name_lower or "mamba" in model_name_lower:
            from src.models import ssm_mamba
            model = instantiate_model_from_module(ssm_mamba, "ssm_mamba")
        elif "tcn" in model_name_lower:
            from src.models import tcn
            model = instantiate_model_from_module(tcn, "tcn")
        elif "transformer" in model_name_lower:
            from src.models import ts_transformer
            model = instantiate_model_from_module(ts_transformer, "ts_transformer")
        elif "xenopulse" in model_name_lower:
            from src.models import xenopulse_net
            model = instantiate_model_from_module(xenopulse_net, "xenopulse_net")
        else:
            raise ValueError(f"Unsupported model target: '{model_name}'")

    except Exception as e:
        raise ImportError(f"❌ Failed to instantiate model '{model_name}'. Details: {e}")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"❌ Weights file not found at: {weights_path}")

    print(f"📦 Loading weights from {weights_path}...")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model


def prepare_inputs_for_eval(model, inputs, override_dim=None):
    num_features = inputs.shape[-1]
    target_dim = override_dim or (200 if num_features in [200, 201] else num_features)
    
    if num_features > target_dim:
        inputs = inputs[:, :target_dim]

    has_conv1d = any(isinstance(m, torch.nn.Conv1d) for m in model.modules())
    if has_conv1d and inputs.ndim == 2:
        inputs = inputs.unsqueeze(1)

    return inputs


def evaluate_model(
    model, 
    test_loader, 
    model_name="Model", 
    device="cuda" if torch.cuda.is_available() else "cpu",
    output_fig_dir="reports/figures",
    target_input_dim=None
):
    os.makedirs(output_fig_dir, exist_ok=True)
    model.to(device)
    model.eval()

    y_true, y_probs = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            inputs = prepare_inputs_for_eval(model, inputs, override_dim=target_input_dim)

            outputs = model(inputs)

            if outputs.shape[-1] == 1 or outputs.ndim == 1:
                probs = torch.sigmoid(outputs.squeeze(-1)).cpu().numpy()
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

    # Probability Diagnostics
    print(f"\n==================== {model_name.upper()} - PROBABILITY DIAGNOSTICS ====================")
    print(f"Probabilities Range   : Min = {y_probs.min():.4f} | Max = {y_probs.max():.4f}")
    print(f"Probabilities Dist.   : Mean = {y_probs.mean():.4f} | Median = {np.median(y_probs):.4f}")
    print(f"Optimal Threshold     : {best_threshold:.4f} (Youden's J Index)")

    # Classification Metric Comparison
    print(f"\n==================== EVALUATION @ DEFAULT THRESHOLD (0.50) ====================")
    print(classification_report(y_true, y_preds_default, target_names=["Non-Transit", "Exoplanet Transit"], zero_division=0))

    print(f"\n==================== EVALUATION @ OPTIMAL THRESHOLD ({best_threshold:.4f}) ====================")
    print(classification_report(y_true, y_preds_optimal, target_names=["Non-Transit", "Exoplanet Transit"], zero_division=0))
    print(f"Test Set ROC-AUC Score: {test_auc:.4f}\n" + "="*70)

    # Save Confusion Matrix Plot using optimal threshold
    cm = confusion_matrix(y_true, y_preds_optimal)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Transit", "Transit"],
                yticklabels=["Non-Transit", "Transit"])
    plt.title(f"{model_name} Test CM (Thresh={best_threshold:.4f})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_path = os.path.join(output_fig_dir, f"test_cm_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # Save ROC Curve Plot
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {test_auc:.4f}", color="darkorange", lw=2)
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.scatter(fpr[best_idx], tpr[best_idx], color="red", marker="o", label=f"Opt Thresh = {best_threshold:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} Test ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(output_fig_dir, f"test_roc_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    print(f"📊 Evaluation plots saved to:\n - {cm_path}\n - {roc_path}")
    return test_auc


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on holdout test set.")
    parser.add_argument("--weights", type=str, default="models/saved/astronet_weights.pth")
    parser.add_argument("--model", type=str, default="AstroNet")
    parser.add_argument("--data", type=str, default="data/processed/master_lightcurves.csv")
    parser.add_argument("--input-dim", type=int, default=None)
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Running evaluation on: {device}")

    try:
        test_loader = load_test_dataloader(csv_path=args.data)
        print(f"✅ Successfully loaded test dataset with {len(test_loader.dataset)} samples.")
        
        model = load_model_from_checkpoint(model_name=args.model, weights_path=args.weights, device=device)
        evaluate_model(
            model=model, 
            test_loader=test_loader, 
            model_name=args.model, 
            device=device,
            target_input_dim=args.input_dim
        )
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")


if __name__ == "__main__":
    main()