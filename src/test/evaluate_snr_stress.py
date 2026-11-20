"""
SNR Stress Test - EVALUATION ONLY (no training, no checkpoint writes).

Pipeline per model and per SNR level:
    existing test data  ->  SNR-based Gaussian noise (in memory, src/test/snr_noise.py)
                        ->  existing model input format  ->  existing checkpoint  ->  inference  ->  metrics

Where the noise is injected (the actual model input, unchanged representation):
  * Classical models : the 200-bin test light curves exactly as produced by
                       ``src.data.dataset.prepare_dataloaders`` (MinMax-scaled to [0, pi],
                       scaler fitted on the clean training split; same split as the clean evaluators).
  * Quantum models   : the saved latent test features ``data/processed/latent/X_test_{cae,pca}_{q}.npy``.
                       The fitted StandardScaler / PCA / CAE were never persisted by
                       ``src/features/dim_reduction.py``, so noise cannot be injected in flux space
                       for quantum models without retraining the CAE. Quantum SNR is therefore
                       defined in latent space, and is NOT numerically equivalent to classical
                       flux-space SNR at the same nominal value.

Metrics: accuracy, precision, recall, F1 (threshold 0.50, as in the benchmark scripts),
ROC-AUC from the sigmoid probabilities, and confusion-matrix counts (TN, FP, FN, TP).

Models covered (only those with an existing checkpoint AND an existing evaluator loader):
  classical: 1D CNN, AstroNet, iTransformer, NiMamba-2, xLSTM
  quantum  : QNN (data_reuploading, vqc) x (cae, pca) x (4, 8) qubits, QCNN (cae, pca) x (4, 8) qubits
XenoPulse-Net is deliberately excluded. Models without saved checkpoints are reported as skipped.

Example (run from the repository root):
    python src/test/evaluate_snr_stress.py \
        --data-csv data/processed/master_lightcurves.csv \
        --latent-dir data/processed/latent \
        --weights-dir models/saved \
        --output logs/snr_stress_test_results.csv
"""
import argparse
import contextlib
import csv
import os
import sys
import tempfile

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(REPO_ROOT)

from src.test.snr_noise import DEFAULT_SNR_LEVELS, POWER_MODES, add_gaussian_noise_snr  # noqa: E402

# Existing benchmark outputs that this script must never write to.
PROTECTED_OUTPUTS = {
    "classical_benchmark_results.csv",
    "master_benchmark_results.csv",
    "qnn_benchmark_results.csv",
    "qnn_benchmark_16q_results.csv",
    "qcnn_benchmark_results.csv",
    "res_qnn_benchmark_results.csv",
    "classical_benchmark_summary.txt",
}

CLASSICAL_MODELS = [
    # (display name, checkpoint filename)
    ("1D CNN", "1d_cnn_weights.pth"),
    ("AstroNet", "astronet_weights.pth"),
    ("iTransformer", "itransformer_weights.pth"),
    ("NiMamba-2", "nimamba2_weights.pth"),
    ("xLSTM", "xlstm_weights.pth"),
]

QUANTUM_MODELS = (
    [("qnn", arch, latent, q) for q in (4, 8) for latent in ("cae", "pca") for arch in ("data_reuploading", "vqc")]
    + [("qcnn", "qcnn", latent, q) for q in (4, 8) for latent in ("cae", "pca")]
)


@contextlib.contextmanager
def isolated_cwd():
    """
    The existing ``src/models/train_*.py`` modules create ``logs/*_train_<ts>.txt`` and
    ``models/saved/`` relative to the CWD at import time. The existing evaluator loaders import
    them to rebuild the architectures, so loading is done inside a throwaway directory to keep
    the repository and data directories untouched.
    """
    prev = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="snr_eval_", ignore_cleanup_errors=True) as tmp:
        os.chdir(tmp)
        try:
            yield tmp
        finally:
            os.chdir(prev)


# ---------------------------------------------------------------------------
# Model loading (reuses the existing evaluator loaders; no architecture changes)
# ---------------------------------------------------------------------------
def load_classical(name, weights_path, device):
    """Returns (model, input_prep_fn) exactly as the corresponding existing evaluator uses them."""
    with isolated_cwd():
        if name == "1D CNN":
            from src.test import evaluate_1d_cnn as ev
            return ev.load_1d_cnn_model(weights_path, device=device), None
        if name == "AstroNet":
            from src.test import evaluate_astronet as ev
            model = ev.load_model_from_checkpoint("AstroNet", weights_path, device=device)
            model.to(device).eval()
            return model, ev.prepare_inputs_for_eval
        if name == "iTransformer":
            from src.test import evaluate_itransformer as ev
            return ev.load_itransformer_model(weights_path, device=device), ev.prepare_inputs_for_eval
        if name == "NiMamba-2":
            from src.test import evaluate_nimamba2 as ev
            return ev.load_nimamba2_model(weights_path, device=device), None
        if name == "xLSTM":
            from src.test import evaluate_xlstm as ev
            return ev.load_xlstm_model(weights_path, device=device), None
    raise ValueError(f"Unsupported classical model: {name}")


def load_quantum(kind, arch, n_qubits, weights_path):
    """Returns (model, torch_device) built with the existing evaluator classes."""
    with isolated_cwd():
        if kind == "qnn":
            from src.test.evaluate_qnn import QuantumClassifier
            model = QuantumClassifier(n_qubits=n_qubits, n_layers=3, architecture=arch)
            model.load_state_dict(torch.load(weights_path))
            model.eval()
            return model, torch.device("cpu")
        from src.test.evaluate_qcnn import QCNNModel, get_execution_hardware
        torch_device, dev, diff_method = get_execution_hardware(n_qubits)
        model = QCNNModel(n_qubits, dev, diff_method).to(torch_device)
        model.load_state_dict(torch.load(weights_path, map_location=torch_device))
        model.eval()
        return model, torch_device


# ---------------------------------------------------------------------------
# Inference & metrics
# ---------------------------------------------------------------------------
def predict_probs(model, X, device, prep_fn=None, batch_size=32):
    """Sigmoid/softmax positive-class probabilities, mirroring the existing evaluators."""
    probs_all = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            inputs = torch.tensor(X[start:start + batch_size], dtype=torch.float32).to(device)
            if prep_fn is not None:
                inputs = prep_fn(model, inputs)
            outputs = model(inputs)
            if outputs.ndim > 1 and outputs.shape[-1] == 1:
                outputs = outputs.squeeze(-1)
            if outputs.ndim == 1:
                probs = torch.sigmoid(outputs)
            else:
                probs = torch.softmax(outputs, dim=1)[:, 1]
            probs_all.extend(np.atleast_1d(probs.cpu().numpy()))
    return np.asarray(probs_all, dtype=np.float64)


def compute_metrics(y_true, probs, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1_score": f1_score(y_true, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def snr_rng(seed, snr):
    """Independent, reproducible stream per SNR level (same noise realisation for every model of a family)."""
    return np.random.default_rng([int(seed), int(round(float(snr) * 1000))])


def run_levels(label, family, X_clean, y, infer_fn, snr_levels, seed, power_mode, extra):
    rows = []
    checksum = X_clean.tobytes()
    for snr in [None] + list(snr_levels):
        if snr is None:
            X_eval, sigma = X_clean, np.zeros(len(X_clean))
        else:
            X_eval, sigma = add_gaussian_noise_snr(
                X_clean, snr, rng=snr_rng(seed, snr), power_mode=power_mode, return_std=True
            )
        m = compute_metrics(y, infer_fn(X_eval))
        row = {
            "model": label, "family": family, **extra,
            "snr": "clean" if snr is None else snr,
            "power_mode": power_mode, "seed": seed, "n_samples": len(y),
            "mean_noise_std": float(np.mean(sigma)), **m,
        }
        rows.append(row)
        print(f"  {str(row['snr']):>6} | Acc {m['accuracy']*100:6.2f}% | Prec {m['precision']:.4f} | "
              f"Rec {m['recall']:.4f} | F1 {m['f1_score']:.4f} | AUC {m['roc_auc']:.4f} | "
              f"TN {m['tn']} FP {m['fp']} FN {m['fn']} TP {m['tp']}", flush=True)
    if X_clean.tobytes() != checksum:
        raise RuntimeError("Clean test inputs were modified in memory - aborting.")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluation-only SNR stress test on existing checkpoints.")
    parser.add_argument("--data-csv", default="data/processed/master_lightcurves.csv")
    parser.add_argument("--latent-dir", default="data/processed/latent")
    parser.add_argument("--weights-dir", default="models/saved")
    parser.add_argument("--output", default="logs/snr_stress_test_results.csv")
    parser.add_argument("--snr", type=float, nargs="+", default=list(DEFAULT_SNR_LEVELS),
                        help="Linear SNR levels (default: 10 5 1 0.1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--power-mode", choices=POWER_MODES, default="variance")
    parser.add_argument("--families", nargs="+", choices=["classical", "quantum"], default=["classical", "quantum"])
    parser.add_argument("--models", nargs="*", default=None,
                        help="Optional case-insensitive substrings to restrict which models run")
    args = parser.parse_args()

    data_csv = os.path.abspath(args.data_csv)
    latent_dir = os.path.abspath(args.latent_dir)
    weights_dir = os.path.abspath(args.weights_dir)
    output = os.path.abspath(args.output)
    if os.path.basename(output) in PROTECTED_OUTPUTS:
        raise SystemExit(f"Refusing to write to existing benchmark file: {output}")

    def wanted(label):
        return not args.models or any(s.lower() in label.lower() for s in args.models)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows, skipped = [], []

    print("SNR Stress Test\n===============")
    print(f"SNR levels: clean + {args.snr} | power_mode={args.power_mode} | seed={args.seed}")

    if "classical" in args.families:
        if not os.path.exists(data_csv):
            raise SystemExit(f"Classical test data not found: {data_csv}")
        from src.data.dataset import prepare_dataloaders
        _, _, test_loader, _, _ = prepare_dataloaders(csv_path=data_csv, batch_size=32)
        X_cls = np.asarray(test_loader.dataset.features, dtype=np.float32)
        y_cls = np.asarray(test_loader.dataset.labels).astype(int)
        for name, ckpt in CLASSICAL_MODELS:
            if not wanted(name):
                continue
            wpath = os.path.join(weights_dir, ckpt)
            if not os.path.exists(wpath):
                skipped.append((name, f"checkpoint missing: {wpath}"))
                continue
            print(f"\n[classical] {name}  ({wpath})")
            try:
                model, prep = load_classical(name, wpath, device)
            except Exception as e:  # pre-existing loader issues are reported, not fixed
                skipped.append((name, f"load failed: {e}"))
                continue
            rows += run_levels(name, "classical", X_cls, y_cls,
                               lambda X, m=model, p=prep: predict_probs(m, X, device, p),
                               args.snr, args.seed, args.power_mode,
                               {"n_qubits": "", "latent_type": "", "noise_domain": "minmax_scaled_flux"})

    if "quantum" in args.families:
        y_path = os.path.join(latent_dir, "y_test.npy")
        for kind, arch, latent, q in QUANTUM_MODELS:
            label = f"{arch}-{latent}-{q}q"
            if not wanted(label):
                continue
            ckpt = (f"qnn_{arch}_{latent}_{q}q_weights.pth" if kind == "qnn"
                    else f"qcnn_{latent}_{q}q_weights.pth")
            wpath = os.path.join(weights_dir, ckpt)
            x_path = os.path.join(latent_dir, f"X_test_{latent}_{q}.npy")
            missing = [p for p in (wpath, x_path, y_path) if not os.path.exists(p)]
            if missing:
                skipped.append((label, f"missing: {missing}"))
                continue
            print(f"\n[quantum] {label}  ({wpath})")
            X_q = np.load(x_path).astype(np.float32)
            y_q = np.load(y_path).astype(int)
            try:
                model, qdev = load_quantum(kind, arch, q, wpath)
            except Exception as e:
                skipped.append((label, f"load failed: {e}"))
                continue
            rows += run_levels(label, "quantum", X_q, y_q,
                               lambda X, m=model, d=qdev: predict_probs(m, X, d),
                               args.snr, args.seed, args.power_mode,
                               {"n_qubits": q, "latent_type": latent, "noise_domain": f"{latent}_latent"})

    for name, reason in [("Res-QNN (4/8/16q)", "no checkpoints are saved by train_res_qnn.py"),
                         ("QNN 16q", "no checkpoints are saved by train_qnn_16q.py")]:
        skipped.append((name, reason))

    if rows:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        fields = ["model", "family", "n_qubits", "latent_type", "noise_domain", "snr", "power_mode", "seed",
                  "n_samples", "mean_noise_std", "accuracy", "precision", "recall", "f1_score", "roc_auc",
                  "tn", "fp", "fn", "tp"]
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved SNR stress-test results to: {output}")

    if skipped:
        print("\nSkipped / not covered:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")


if __name__ == "__main__":
    main()
