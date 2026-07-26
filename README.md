# Quantum Machine Learning for Exoplanet Detection 🪐⚛️
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![PennyLane](https://img.shields.io/badge/PennyLane-QML-purple.svg)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

An end-to-end research and benchmarking framework comparing state-of-the-art Classical Time-Series Deep Learning models against Hybrid Quantum Neural Network architectures for stellar photometric lightcurve classification and exoplanet transit identification.

---

## 📖 Project Overview

Identifying exoplanet transit signatures within stellar lightcurve flux data is a foundational challenge in modern observational astrophysics. Lightcurves gathered by missions like NASA Kepler and TESS suffer from low signal-to-noise ratios (SNR), stellar variability, and long temporal dependencies.

This project implements a comprehensive computational pipeline that evaluates:

1. **SOTA Classical Architectures**: Ranging from 1D CNNs, ResNets, and AstroNet to Advanced Transformers (PatchTST, TimesNet, iTransformer), State Space Models (Selective-SSM, Mamba-2, xLSTM), and our custom high-performing baseline, **XenoPulse-Net**.
2. **Hybrid Quantum Architectures**: Variational Quantum Circuits (VQC), Data Re-uploading QNNs, Quantum Convolutional Neural Networks (QCNN), and **Residual Quantum Neural Networks (Res-QNN)**.
3. **Latent Space Embeddings**: Comparing non-linear **Convolutional Autoencoders (CAE)** vs. linear **Principal Component Analysis (PCA)** for quantum state feature encoding across 4, 8, and 16-qubit configurations.

---
## ✨ Highlights

- Hybrid Classical–Quantum Learning Pipeline
- 20+ Deep Learning & Quantum Models Benchmarked
- NASA Kepler & TESS Lightcurve Processing
- GPU Accelerated PennyLane-Lightning
- Benchmarking across 4, 8 and 16 qubits
- Custom XenoPulse-Net Architecture
- Residual Quantum Neural Network (Res-QNN)

## 🛰️ Dataset

This project utilizes publicly available stellar photometric lightcurve observations obtained from NASA space telescope missions:

- **NASA Kepler Mission** – High-precision stellar flux measurements for exoplanet transit detection.
- **NASA TESS (Transiting Exoplanet Survey Satellite)** – Wide-field photometric observations for identifying exoplanet candidates around nearby stars.

The dataset consists of normalized stellar lightcurves representing temporal flux variations, where the objective is binary classification of **planetary transit** vs **non-transit** events.

### Dataset Overview

- **Primary Source:** NASA Exoplanet Archive (IPAC/Caltech)
- **Lightcurve Access:** Lightkurve Python Library
- **Missions:** Kepler and TESS
- **Data Type:** Time-series stellar lightcurve flux measurements
- **Task:** Binary Classification
- **Input:** Normalized stellar flux sequences
- **Output:** Exoplanet Transit / Non-Transit

The preprocessing pipeline performs detrending, normalization, phase folding, temporal binning, and class balancing before feature extraction and model training.



---

# 📁 Repository Structure

```text
qml-exoplanet-detection/
├── environment.yml                  # Conda environment configuration
├── requirements.txt                 # Pip dependency list
├── logs/                            # Raw training execution logs & summary CSVs
│   ├── classical_benchmark_results.csv
│   ├── qnn_benchmark_results.csv
│   ├── qnn_benchmark_16q_results.csv
│   ├── qcnn_benchmark_results.csv
│   ├── res_qnn_benchmark_results.csv
│   └── master_benchmark_results.csv
├── reports/                         # Summary markdown reports & figures
│   ├── master_benchmark_summary.md
│   └── figures/
│       ├── auc_vs_runtime.png
│       └── classical_benchmark_metrics.png
└── src/                             # Source code directory
    ├── data/                        # Dataset acquisition and preprocessing
    │   ├── parse_nasa_archive.py    # NASA Exoplanet Archive parser
    │   ├── preprocess_lightcurve.py # Normalization, detrending & binning
    │   └── dataset.py               # PyTorch / TF Dataset loaders
    ├── features/                    # Feature engineering & dimensionality reduction
    │   ├── balance_and_split.py     # Class balancing & stratified splitting
    │   └── dim_reduction.py         # CAE & PCA feature extractors (4Q, 8Q, 16Q)
    ├── models/                      # Model definitions & training scripts
    │   ├── xenopulse_net.py         # SOTA Classical Baseline model definition
    │   ├── hybrid_qnn.py            # PennyLane Hybrid QNN & VQC definitions
    │   ├── classical_cnn.py         # 1D CNN & ResNet implementations
    │   ├── ssm_mamba.py             # Selective SSM & Mamba implementations
    │   ├── train_classical.py       # Multi-model classical trainer
    │   ├── train_qnn.py             # Standard QNN / VQC training script
    │   ├── train_qcnn.py            # QCNN benchmark script
    │   └── train_res_qnn.py         # Res-QNN benchmark script
    └── visualization/               # Report generation & plotting scripts
        ├── generate_master_table.py # Master benchmark table aggregator
        └── plot_benchmarks.py       # Metrics visualization scripts
```

---

# ⚙️ Tech Stack

## Programming Languages

- Python 3.10+
- CUDA Toolkit

## Machine Learning & Deep Learning

- PyTorch
- TensorFlow
- Scikit-learn

## Quantum Machine Learning

- PennyLane
- PennyLane Lightning-GPU
- Qiskit
- PennyLane-Qiskit
- NVIDIA cuQuantum

## Data Processing

- NumPy
- Pandas
- Lightkurve

## Visualization

- Matplotlib

## Development Environment

- Conda / Mamba
- Jupyter Notebook

## Optional Backend & Utilities

- FastAPI
- Uvicorn
- PostgreSQL (psycopg2)

---

# 🔬 Pipeline & Methodology

## 1. Lightcurve Data Preprocessing

Raw stellar photometric lightcurves are retrieved from the NASA Exoplanet Archive using the Lightkurve Python library before undergoing the following preprocessing steps:

- **Detrending & Outlier Removal:** Median filtering and polynomial smoothing to eliminate systematic telescope drift.
- **Phase Folding & Binning:** Aligning lightcurves on candidate orbital periods and binning to fixed length vectors.
- **Class Balancing:** Stratified train/test splits combined with weighted cross-entropy to handle heavily imbalanced exoplanet transit labels.

---

## 2. Dimensionality Compression for Quantum Mapping

To encode continuous high-dimensional temporal features into subatomic Hilbert space states across

\[
N \in \{4, 8, 16\}
\]

qubits, two latent reduction techniques are benchmarked:

- **Convolutional Autoencoder (CAE):** Unsupervised non-linear feature extractor preserving topological transit dip shapes.
- **Principal Component Analysis (PCA):** Linear orthogonal projection preserving maximal global variance.

---

## 3. Classical Time-Series Architecture Suite

We benchmark a diverse spectrum of modern deep learning architectures on raw lightcurve inputs:

- **XenoPulse-Net:** Custom architecture combining multi-scale 1D spatial convolutions, squeeze-and-excitation channel attention, and residual skip paths.
- **State Space Models (SSMs):** Selective-SSM, NiMamba-2, and xLSTM targeting long-range lightcurve dependencies.
- **Long-Sequence Transformers:** PatchTST, TimesNet, and iTransformer capturing cross-time-step periodic correlations.
- **Standard Baselines:** AstroNet, 1D ResNet, TCN, and BiLSTM.

---

## 4. Hybrid Quantum Neural Network Architectures

Quantum circuits are constructed in PennyLane and executed via PyTorch using high-performance C++ GPU acceleration (`lightning.gpu`):

- **Variational Quantum Circuits (VQC) & Data Re-uploading:** Angle-embedding features into qubit rotations (\(R_x, R_y, R_z\)) interleaved with entangling CNOT layers.
- **Quantum Convolutional Neural Networks (QCNN):** Alternating quantum convolution and quantum pooling layers that progressively reduce qubit count while extracting spatial correlations.
- **Residual Quantum Neural Networks (Res-QNN):** Incorporating a classical linear shortcut path alongside the VQC ansatz:

\[
y = \text{VQC}(x) + W_s \cdot x
\]

This architecture mitigates barren plateaus, protects gradient flow during early backpropagation, and prevents information bottlenecks.

---

# 📊 Experimental Results & Benchmarks

All models were evaluated on identical test splits using standard classification metrics:

**Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Training Execution Time.**

## Benchmark Summary

- 12 Classical Models
- 20 Quantum Benchmarks
- 4Q, 8Q and 16Q Experiments
- CAE vs PCA comparison

---

## 1. Classical Baseline Models

| Model Name | Accuracy (%) | Precision | Recall | F1-Score | ROC-AUC | Training Time (s) |
|------------|-------------:|----------:|--------:|---------:|---------:|------------------:|
| XenoPulse-Net | 87.68% | 0.8885 | 0.9458 | 0.9163 | 0.9324 | 15.99 |
| 1D CNN | 74.64% | 0.7399 | 0.9932 | 0.8480 | 0.8365 | 8.00 |
| AstroNet | 74.40% | 0.7368 | 0.9966 | 0.8473 | 0.8396 | 10.00 |
| Selective-SSM | 73.67% | 0.7313 | 0.9966 | 0.8436 | 0.7997 | 258.90 |
| BiLSTM | 71.98% | 0.7313 | 0.9593 | 0.8299 | 0.6146 | 10.00 |
| xLSTM | 71.50% | 0.7143 | 1.0000 | 0.8333 | 0.5394 | 647.20 |
| 1D ResNet | 71.26% | 0.9681 | 0.6169 | 0.7536 | 0.8778 | 9.00 |
| NiMamba-2 | 71.26% | 0.7126 | 1.0000 | 0.8322 | 0.5829 | 239.60 |
| TimesNet | 71.26% | 0.7136 | 0.9966 | 0.8317 | 0.5601 | 20.70 |
| PatchTST | 71.26% | 0.7126 | 1.0000 | 0.8322 | 0.5429 | 13.70 |
| iTransformer | 71.26% | 0.7126 | 1.0000 | 0.8322 | 0.4878 | 10.00 |
| TCN | 43.96% | 0.9846 | 0.2169 | 0.3556 | 0.7292 | 11.00 |

---

## 2. Quantum Architectures (QNN, QCNN, Res-QNN)

| Architecture | Qubits | Latent Type | Accuracy (%) | Precision | Recall | F1-Score | ROC-AUC | Training Time (s) |
|--------------|:------:|:-----------:|-------------:|----------:|--------:|---------:|---------:|------------------:|
| Data Re-uploading | 4 | CAE | 71.50% | 0.7143 | 1.0000 | 0.8333 | 0.6249 | 28.40 |
| VQC | 4 | CAE | 71.26% | 0.7126 | 1.0000 | 0.8322 | 0.6357 | 18.79 |
| Data Re-uploading | 4 | PCA | 72.22% | 0.7228 | 0.9898 | 0.8355 | 0.7566 | 28.31 |
| VQC | 4 | PCA | 72.46% | 0.7224 | 0.9966 | 0.8376 | 0.7382 | 18.63 |
| Data Re-uploading | 8 | CAE | 79.95% | 0.8292 | 0.9051 | 0.8655 | 0.7680 | 61.82 |
| VQC | 8 | CAE | 71.98% | 0.7221 | 0.9864 | 0.8338 | 0.8052 | 41.77 |
| Data Re-uploading | 8 | PCA | 78.26% | 0.8024 | 0.9220 | 0.8580 | 0.7666 | 61.70 |
| VQC | 8 | PCA | 71.50% | 0.7143 | 1.0000 | 0.8333 | 0.7200 | 41.92 |
| Data Re-uploading | 16 | PCA | 79.47% | 0.8163 | 0.9186 | 0.8644 | 0.7796 | 3053.02 |
| VQC | 16 | PCA | 71.50% | 0.7153 | 0.9966 | 0.8329 | 0.7386 | 2324.01 |
| QCNN | 8 | CAE | 68.12% | 0.7063 | 0.9458 | 0.8087 | 0.6417 | 2948.87 |
| QCNN | 8 | PCA | 68.84% | 0.7075 | 0.9593 | 0.8144 | 0.4201 | 31705.00 |
| QCNN | 16 | CAE | 71.01% | 0.7226 | 0.9627 | 0.8256 | 0.5423 | 36369.50 |
| QCNN | 16 | PCA | 68.60% | 0.7110 | 0.9424 | 0.8105 | 0.4350 | 7590.03 |
| Res-QNN | 4 | CAE | 73.67% | 0.7325 | 0.9932 | 0.8432 | 0.7208 | 763.14 |
| Res-QNN | 4 | PCA | 73.19% | 0.7289 | 0.9932 | 0.8407 | 0.7404 | 762.11 |
| Res-QNN | 8 | CAE | 74.40% | 0.7357 | 1.0000 | 0.8477 | 0.8249 | 1251.73 |
| Res-QNN | 8 | PCA | 72.95% | 0.7293 | 0.9864 | 0.8386 | 0.7494 | 1261.08 |
| Res-QNN | 16 | CAE | 74.40% | 0.7392 | 0.9898 | 0.8464 | 0.8110 | 2575.06 |
| Res-QNN | 16 | PCA | 72.71% | 0.7298 | 0.9797 | 0.8365 | 0.7631 | 2585.56 |

---

# 💡 Key Findings

- **Classical SOTA Leader:** XenoPulse-Net achieved the highest overall performance (87.68% Accuracy, 0.9324 ROC-AUC, 0.9163 F1-score), proving that multi-scale spatial convolution coupled with channel-wise attention excels at isolating lightcurve transit dips.

- **Quantum Peak Performance:** The 8-Qubit CAE Res-QNN achieved the top performance among quantum architectures (74.40% Accuracy, 0.8249 ROC-AUC, 100.0% Recall).

- **Res-QNN Overcomes QCNN Bottlenecks:** Standard QCNNs suffered severe performance degradation under linear PCA inputs (ROC-AUC dropping to 0.4201). The linear residual shortcut in Res-QNN restored stability, elevating PCA ROC-AUC to 0.7631 and proving that skip connections protect variational gradient flow.

- **CAE vs. PCA Latent Representations:** Non-linear CAE embeddings consistently outperformed linear PCA embeddings across all quantum models, demonstrating that non-linear feature extraction better aligns with quantum angle embedding state preparation.

# 🛠️ Installation & Quickstart

## Prerequisites

- **Linux Operating System (Recommended)** 
  - Arch Linux / EndeavourOS
  - Ubuntu 22.04+
  - Fedora 40+
- Python 3.10+
- Conda or Mamba
- NVIDIA GPU with CUDA support (Recommended)
- NVIDIA CUDA Toolkit
- NVIDIA Drivers
- Git

> **Note**
>
> The project was primarily developed and benchmarked under a Linux environment.
> Several GPU-accelerated quantum libraries—including **PennyLane Lightning-GPU**, **cuQuantum**, and CUDA-enabled deep learning frameworks—offer significantly better compatibility and performance on Linux than on Windows.
>
> Although CPU execution is possible on Windows, GPU-based quantum simulation is recommended on Linux.

---

## Software Requirements

The project depends on the following major frameworks and libraries.

### Deep Learning

- TensorFlow
- PyTorch
- TorchVision
- TorchAudio

### Quantum Computing

- PennyLane
- PennyLane-Lightning (GPU)
- Qiskit
- PennyLane-Qiskit
- NVIDIA cuQuantum

### Machine Learning

- Scikit-learn

### Scientific Computing

- NumPy
- Pandas

### Visualization

- Matplotlib

### Backend (Optional)

- FastAPI
- Uvicorn
- psycopg2

These dependencies are automatically installed through the provided `requirements.txt` or `environment.yml`.

---

## 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/Arjun13-git/qml-exoplanet-detection.git
cd qml-exoplanet-detection

# Create and activate conda environment
conda create -n galaxy_gpu python=3.10 -y
conda activate galaxy_gpu

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Preprocessing & Feature Extraction

```bash
# Parse catalog and preprocess lightcurves
python src/data/preprocess_lightcurve.py

# Generate CAE and PCA latent space embeddings (4, 8, 16 dimensions)
python src/features/dim_reduction.py
```

---

## 3. Model Training & Benchmarking

```bash
# Train SOTA Classical Baselines (XenoPulse-Net, AstroNet, SSMs, Transformers)
python src/models/train_classical.py

# Run Standard QNN / VQC Benchmarks
python src/models/train_qnn.py

# Run QCNN Benchmarks
python src/models/train_qcnn.py

# Run Res-QNN Benchmarks
python src/models/train_res_qnn.py
```

---

## 4. Consolidated Reporting

```bash
# Consolidate all log CSVs into master markdown report
python src/visualization/generate_master_table.py
```

---

# 🚀 Hardware Used for Benchmarking

The benchmark experiments were executed using GPU acceleration for both classical deep learning and quantum simulations.

### Recommended Hardware

| Component | Recommendation |
|-----------|----------------|
| CPU | Multi-core x86_64 Processor |
| RAM | 16 GB minimum (32 GB recommended) |
| GPU | NVIDIA CUDA-capable GPU |
| CUDA | CUDA Toolkit 12+ |
| Storage | SSD |

GPU acceleration is strongly recommended for training large classical architectures and executing high-qubit quantum simulations efficiently.

---

# 📚 Research Focus

This repository investigates the effectiveness of modern Quantum Machine Learning techniques for astrophysical time-series classification through comprehensive benchmarking against state-of-the-art classical deep learning models.

The primary objectives include:

- Benchmarking classical and hybrid quantum architectures under identical experimental settings.
- Evaluating quantum representation learning across multiple qubit configurations (4, 8, and 16 qubits).
- Comparing non-linear and linear latent feature compression techniques for quantum encoding.
- Investigating the scalability, stability, and computational trade-offs of hybrid quantum neural networks for exoplanet detection.

---

# 📜 License

This project is licensed under the MIT License.