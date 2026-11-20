# Setup & Execution Guide — `qml-exoplanet-detection`

This guide describes how to set up and run the repository **exactly as it currently exists**.
All commands are run from the **repository root** unless stated otherwise.
Where this file differs from the Quickstart in `README.md`, follow this file.

---

## 1. Prerequisites

> ### ⚠️ Linux (x86_64) is REQUIRED
> This project must be set up and run on a **64-bit Linux (x86_64)** machine.
> It was developed, executed and validated only on Linux (kernel 6.18, x86_64).
> Windows and macOS are **not supported** for this repository as configured.

**Why Linux is required**

1. **The pinned environment only installs on Linux.** `requirements.txt` is a full snapshot of the
   working environment and pins Linux-only binary wheels, e.g. `torch==2.13.0`
   (`manylinux_2_28_x86_64`, bundled CUDA 13 runtime), `pennylane_lightning_gpu==0.42.0`
   (`manylinux_2_28_x86_64`), `cuquantum==26.6.0` / `custatevec-cu13==1.14.0`
   (`manylinux2014_x86_64`), the `nvidia-*` CUDA wheels, `tensorflow==2.21.0` and `tenseal==0.3.16`
   (`manylinux_2_27_x86_64`). These have no Windows/macOS builds at these versions.
2. **The Lightkurve/MAST data-acquisition workflow was only run and validated on Linux.**
   `lightkurve` itself is a pure-Python package (`py3-none-any`); the Linux requirement comes from
   the environment it is installed into (point 1) and from the fact that the data pipeline in
   `src/data/` (MAST search → download → Savitzky–Golay flatten → fold → bin) was executed only
   on Linux, writing POSIX paths (`data/raw/...`, `data/processed/...`) and using Lightkurve's
   on-disk MAST download cache in the user's home directory (`~/.lightkurve/cache`).
   No other platform has been tested.
3. **Hard-coded POSIX absolute paths** exist in several training scripts (see §8.3).

**Required software**

| Requirement | Version used for this project | Required? |
|---|---|---|
| OS | Linux x86_64 (glibc-based; wheels need manylinux_2_28 ⇒ glibc ≥ 2.28) | **Required** |
| Python | **3.10** (environment uses Python 3.10.20) | **Required** |
| Conda **or** Mamba (Miniforge recommended) | conda 26.5.3 / mamba 2.8.1 | Required (to create the Python 3.10 env) |
| pip | 26.1.2 (inside the env) | Required |
| Git | any recent version | Only if cloning |
| Internet access | MAST + NASA Exoplanet Archive | Required **only** for dataset generation |
| Disk space | ~16 GB for the installed environment (+ ~0.5 GB data/cache) | Required |
| NVIDIA GPU + driver | Dev machine: RTX 3050 Laptop (4 GB), driver 615.78.08, CUDA 13.0 runtime (from pip wheels) | **Recommended, not strictly required** |

No separate system CUDA Toolkit installation is needed: the CUDA runtime libraries are installed
as pip wheels (`nvidia-*`) by `requirements.txt`. Only an NVIDIA **driver** is needed for GPU use.

**CPU-only machines:** the code falls back automatically —
PyTorch uses `cuda` only if available; `train_qcnn.py`, `train_res_qnn.py`, `train_qnn_16q.py`
and `evaluate_qcnn.py` fall back from `lightning.gpu` to `default.qubit`; `train_qnn.py` /
`evaluate_qnn.py` always use `default.qubit` (CPU). Everything runs on CPU, but 16-qubit and QCNN
training is very slow (logged GPU training times are already 2,000–36,000 s per configuration).

---

## 2. Conda / Mamba environment

The project environment is named **`galaxy_gpu`** and uses **Python 3.10**.

Using **Mamba** (recommended, faster):

```bash
mamba create -n galaxy_gpu python=3.10 -y
mamba activate galaxy_gpu        # or: conda activate galaxy_gpu
```

Using **Conda**:

```bash
conda create -n galaxy_gpu python=3.10 -y
conda activate galaxy_gpu
```

Verify the interpreter belongs to the new environment:

```bash
which python            # should end in .../envs/galaxy_gpu/bin/python
python --version        # Python 3.10.x
python -m pip --version
```

Deactivate when finished:

```bash
conda deactivate        # (or: mamba deactivate)
```

> `environment.yml` in the repository is an older, **unpinned** environment description
> (it installs the latest versions of each package). It is **not** the tested environment —
> use `requirements.txt` as described below.

---

## 3. Repository setup

Either extract the submitted ZIP, or clone:

```bash
git clone https://github.com/Arjun13-git/qml-exoplanet-detection.git
cd qml-exoplanet-detection
```

(If extracting a ZIP, `cd` into the extracted `qml-exoplanet-detection/` folder.)

Verify that you are in the repository root and the expected directories exist:

```bash
ls README.md setup.md requirements.txt
ls -d src/data src/features src/models src/test src/visualization logs reports/figures
```

Expected layout (source tree):

```text
qml-exoplanet-detection/
├── requirements.txt         # pinned environment (pip freeze snapshot)
├── setup.md                 # this file
├── README.md
├── environment.yml          # older, unpinned (not used by this guide)
├── logs/                    # existing benchmark logs/CSVs
├── reports/                 # master_benchmark_summary.md, figures/
└── src/
    ├── data/                # catalog building + Lightkurve light-curve pipeline
    ├── features/            # dim_reduction.py (PCA / 1D-CAE), balance_and_split.py
    ├── models/              # model definitions + training scripts
    ├── test/                # evaluation scripts + SNR stress test
    └── visualization/       # report table + plots
```

---

## 4. Python dependency installation

There are **no extra system packages** to install via `apt`/`pacman` beyond a working
Linux + NVIDIA driver (for GPU). Install the Python packages inside the activated environment:

```bash
conda activate galaxy_gpu
python -m pip install --upgrade pip
python -m pip install --no-deps -r requirements.txt
```

**Why `--no-deps`:** `requirements.txt` is a *complete* snapshot (`pip freeze`, 755 packages) of
the working environment, so every dependency is already listed with its exact version. The
snapshot contains one pre-existing pin conflict (`datasets 4.8.5` declares `fsspec<=2026.2.0`,
but the working environment has `fsspec 2026.6.0`). Without `--no-deps`, pip's resolver refuses
the file; with `--no-deps` pip installs the exact versions that were actually used.

After installation, `python -m pip check` is expected to report only that one known line:

```text
datasets 4.8.5 has requirement fsspec[http]<=2026.2.0,>=2023.1.0, but you have fsspec 2026.6.0.
```

This does not affect this project (`datasets` is not imported by any project code).

> The snapshot is large (it also contains packages present in the environment but not imported by
> the project code, e.g. TensorFlow, Qiskit, LangChain/Langflow). Expect a long download and
> ~16 GB of disk usage.

---

## 5. Lightkurve (MAST light-curve access)

- **Package / version:** `lightkurve==2.6.0` (with `astropy==6.1.7`, `astroquery==0.4.11`),
  installed by `requirements.txt` — no separate installation step is needed.
- **Used by:** `src/data/download_archive.py`, `src/data/preprocess_lightcurve.py`,
  `src/data/generate_raw_dataset.py` (search and download of TESS SPOC / Kepler light curves
  from MAST, then `flatten` → `fold` → `bin`).
- **Platform:** Linux is mandatory for this project as configured (see §1).
- **Network:** requires outbound HTTPS access to MAST (`lk.search_lightcurve` / `.download()`)
  and, for `patch_catalog.py`, to `exoplanetarchive.ipac.caltech.edu`.
- **Cache / size:** downloaded MAST products are cached under `~/.lightkurve/cache`. Outputs can be
  large (e.g. the WASP-121 stitched raw light curve written by `download_archive.py` to
  `data/raw/` is ~436 MB).
- **Savitzky–Golay:** the pipeline's `lc.flatten(window_length=901)` (`generate_raw_dataset.py`)
  is Lightkurve's Savitzky–Golay detrending: it calls `scipy.signal.savgol_filter`
  (`window_length=901`, default `polyorder=2`, 3 iterations with 3σ outlier masking applied only
  to the trend fit) and divides the flux by the fitted trend.

Verify:

```bash
python -B -c "import lightkurve; print(lightkurve.__version__)"
```

Expected output ends with `2.6.0`. A `UserWarning` about the `tpfmodel`/`oktopus` submodule may
be printed on import; it is harmless (that submodule is not used).

---

## 6. Dataset / data requirements

**The dataset is NOT included in the source package.** `.gitignore` excludes `data/raw/`,
`data/processed/`, `models/saved/` and `*.csv` / `*.pt` files (the CSVs in `logs/` are benchmark
results, not the dataset). You must either receive the data files from the project team or
regenerate them (internet required).

### 6.1 Expected data layout

```text
data/
├── raw/
│   ├── nasa_planetary_systems.csv     # manual download (see 6.2), ~78 MB
│   └── full_stellar_catalog.csv       # generated (2,100 rows: 1,500 positives + 600 negatives)
└── processed/
    ├── master_lightcurves.csv         # generated: 2,069 rows × 202 cols (target_name, label, flux_0..flux_199)
    ├── failed_targets.txt             # generated by generate_raw_dataset.py
    └── latent/                        # generated by dim_reduction.py
        ├── X_train_{pca,cae}_{4,8,16}.npy
        ├── X_test_{pca,cae}_{4,8,16}.npy
        ├── y_train.npy
        └── y_test.npy
```

Create the folders (if you are placing supplied files):

```bash
mkdir -p data/raw data/processed/latent models/saved
```

### 6.2 Source data

1. **NASA Exoplanet Archive — Planetary Systems table** (positives).
   The repository does not contain a download URL for this file. The authors' local copy is a
   CSV export of the Archive's *Planetary Systems* table (header: *"This file was produced by the
   NASA Exoplanet Archive http://exoplanetarchive.ipac.caltech.edu"*, exported 2026-07-14, all
   columns). Download that table as CSV from the Archive and save it as
   `data/raw/nasa_planetary_systems.csv`. Required columns: `hostname`, `pl_orbper`, `pl_tranmid`,
   and `tran_flag` or `discoverymethod`.
2. **Kepler KOI cumulative table — FALSE POSITIVE KOIs** (negatives). Fetched automatically by
   `src/data/patch_catalog.py` from the Archive TAP URL embedded in that script.
3. **Light curves** — downloaded automatically from MAST via Lightkurve.

Because the Archive and MAST change over time, a regenerated dataset may not be identical to the
2,069-sample dataset used for the logged results.

### 6.3 Data-generation steps (in this exact order)

```bash
python src/data/parse_nasa_archive.py    # -> data/raw/full_stellar_catalog.csv (1,500 unique transiting hosts, label 1)
python src/data/patch_catalog.py         # appends 600 Kepler FALSE POSITIVE KOIs (label 0) — needs internet
python src/data/full_stellar_catalog.py  # repairs the column order of the appended rows (run ONCE, in place)
python src/data/generate_raw_dataset.py  # MAST download + SG flatten + fold + bin -> data/processed/master_lightcurves.csv
python src/features/dim_reduction.py     # StandardScaler + PCA / 1D-CAE (4, 8, 16) -> data/processed/latent/*.npy
```

Notes:
- `generate_raw_dataset.py` is long-running (thousands of MAST downloads) and **resumable**: it
  skips targets already in `master_lightcurves.csv` or listed in `failed_targets.txt`.
- `dim_reduction.py` retrains the CAE (no fixed seed). **Regenerating `data/processed/latent/`
  makes the existing quantum checkpoints inconsistent with the new latent features.** If you
  received the supplied `latent/` files together with checkpoints, do not regenerate them.
- Single-star demo scripts (not needed for the benchmark): `src/data/download_archive.py`,
  `src/data/plot_lightcurve.py`, `src/data/preprocess_lightcurve.py` (WASP-121b, TIC 260128333).

---

## 7. Model / checkpoint requirements

**Checkpoints are NOT included in the source package** (`models/saved/` is git-ignored). They
must be obtained separately from the project team (~14 MB total) or regenerated by training (§8.2).
Place them in `models/saved/`:

| Checkpoint (`models/saved/…`) | Used by |
|---|---|
| `1d_cnn_weights.pth` | `src/test/evaluate_1d_cnn.py`, SNR test |
| `astronet_weights.pth` | `src/test/evaluate_astronet.py`, SNR test |
| `itransformer_weights.pth` | `src/test/evaluate_itransformer.py`, SNR test |
| `nimamba2_weights.pth` | `src/test/evaluate_nimamba2.py`, SNR test |
| `xlstm_weights.pth` | `src/test/evaluate_xlstm.py`, SNR test |
| `xenopulse_net_weights.pth` | `src/test/evaluate_xenopulse_net.py` (experimental model; not in SNR test) |
| `qnn_{data_reuploading,vqc}_{cae,pca}_{4,8}q_weights.pth` (8 files) | `src/test/evaluate_qnn.py`, SNR test |
| `qcnn_{cae,pca}_{4,8}q_weights.pth` (4 files) | `src/test/evaluate_qcnn.py`, SNR test |

Not available as checkpoints: Res-QNN (`train_res_qnn.py` saves none), 16-qubit QNN
(`train_qnn_16q.py` saves none), 16-qubit QCNN (current `train_qcnn.py` trains 4/8 qubits only).

Classical evaluators additionally need `data/processed/master_lightcurves.csv`; quantum evaluators
need `data/processed/latent/X_test_*.npy` and `y_test.npy`.

---

## 8. Running the project

Always:

```bash
conda activate galaxy_gpu
cd /path/to/qml-exoplanet-detection     # repository root — all paths are relative to it
```

> **Files overwritten by the existing scripts.** Training scripts append to
> `logs/classical_benchmark_results.csv` / `logs/classical_benchmark_summary.txt`, write
> `logs/<model>_train_<timestamp>.txt`, and overwrite `models/saved/*.pth`; quantum training scripts
> overwrite `logs/qnn_benchmark*.txt/.csv`, `logs/qcnn_benchmark*`, `logs/res_qnn_benchmark*`;
> evaluators overwrite `reports/figures/test_cm_*.png` and `test_roc_*.png`; reporting scripts
> overwrite `reports/master_benchmark_summary.md` and `reports/figures/*.png`.
> Work on a copy if you need to keep the submitted results untouched.

### 8.1 Data preparation
See §6.3.

### 8.2 Training (optional — retrains and overwrites checkpoints/logs)

Classical models (each trains 30 epochs, 40 for PatchTST, on the 80/10/10 split from `src/data/dataset.py`):

```bash
python src/models/train_1d_cnn.py
python src/models/train_astronet.py
python src/models/train_resnet1d.py
python src/models/train_tcn.py
python src/models/train_lstm.py          # BiLSTM
python src/models/train_patchtst.py
python src/models/train_timesnet.py
python src/models/train_itransformer.py
python src/models/train_ssm.py           # Selective-SSM
python src/models/train_nimamba2.py
python src/models/train_xlstm.py
python src/models/xenopulse_net.py       # experimental model
```

Quantum models (require `data/processed/latent/`):

```bash
python src/models/train_qnn.py           # VQC + Data Re-uploading, 4 & 8 qubits, CAE & PCA (CPU, default.qubit)
python src/models/train_qnn_16q.py       # VQC + Data Re-uploading, 16 qubits, PCA (lightning.gpu if available)
python src/models/train_qcnn.py          # QCNN, 4 & 8 qubits, CAE & PCA
python src/models/train_res_qnn.py       # Res-QNN, 4/8/16 qubits, CAE & PCA
```

### 8.3 Known path limitation (hard-coded absolute paths)

These scripts read the dataset **only** from the absolute path
`/home/arjunshenoy13/qml-exoplanet-detection/data/processed/master_lightcurves.csv`:
`train_astronet.py`, `train_resnet1d.py`, `train_tcn.py`, `train_lstm.py`, `train_patchtst.py`,
`train_timesnet.py`, `train_itransformer.py`, `train_ssm.py`, `xenopulse_net.py`.
On another machine they fail with `FileNotFoundError` unless the file exists at that exact path
(e.g. created by an administrator with
`sudo mkdir -p /home/arjunshenoy13/qml-exoplanet-detection/data/processed` and a symlink to your
`master_lightcurves.csv`). `train_1d_cnn.py`, `train_nimamba2.py` and `train_xlstm.py` first try the
relative path `data/processed/master_lightcurves.csv` and work from the repository root.
All **evaluation** scripts and the SNR stress test use relative/CLI paths and are not affected.

### 8.4 Evaluation of existing checkpoints (inference only)

Classical (default paths `models/saved/<name>_weights.pth`, `data/processed/master_lightcurves.csv`;
override with `--weights` / `--data`):

```bash
python src/test/evaluate_1d_cnn.py
python src/test/evaluate_astronet.py         # options: --weights --model --data --input-dim
python src/test/evaluate_itransformer.py
python src/test/evaluate_nimamba2.py
python src/test/evaluate_xlstm.py
python src/test/evaluate_xenopulse_net.py    # experimental model
```

Quantum (latent features from `data/processed/latent/`, checkpoints from `models/saved/`):

```bash
python src/test/evaluate_qnn.py --qubits 4   # options: --arch {all,data_reuploading,vqc} --latent {all,cae,pca}
python src/test/evaluate_qnn.py --qubits 8
python src/test/evaluate_qcnn.py --qubits 4  # options: --latent {all,cae,pca}
python src/test/evaluate_qcnn.py --qubits 8
```

Each evaluator prints a classification report at threshold 0.50 and at the Youden-J threshold,
the ROC-AUC, and saves a confusion-matrix and ROC figure to `reports/figures/`.

### 8.5 Reporting

```bash
python src/visualization/generate_master_table.py   # -> reports/master_benchmark_summary.md
python src/visualization/plot_benchmarks.py         # -> reports/figures/benchmark_metrics.png, auc_vs_runtime.png
```

### 8.6 Other existing scripts (legacy / auxiliary)

- `src/features/balance_and_split.py` → `data/processed/split_tensors.pt`, used only by the early
  experimental scripts `src/models/train_classical.py` and `src/models/train_transformer.py`
  (3-class mini-batch prototypes; not part of the reported benchmarks).
- `retrain_classical_baselines.py` and `check_models_status.py` (repository root) are auxiliary
  wrappers; the per-model scripts in §8.2 are the actual training entry points.

---

## 9. SNR stress test (evaluation-time Gaussian noise)

`src/test/evaluate_snr_stress.py` (with the noise utility `src/test/snr_noise.py`):

- **Does not train or modify any model.** It loads the existing checkpoints listed in §7, reuses the
  existing evaluators' model loaders, injects noise **in memory** into the test inputs, runs inference
  and computes metrics. Files on disk (data, latents, checkpoints) are never modified.
- **Noise:** white Gaussian noise at a requested **linear** SNR, per sample:
  `P_signal = Var(x)` (default `--power-mode variance`; or `mean_square`),
  `σ = sqrt(P_signal / SNR)`, `x_noisy = x + N(0, σ²)`. Deterministic for a given `--seed` (default 42).
  This is separate from the small fixed training augmentation (σ = 0.001) in `src/data/dataset.py`.
- **SNR levels:** clean + **10, 5, 1, 0.1** by default (any positive values via `--snr`).
- **Noise domain:** classical models — the 200-bin MinMax-scaled light curves used by the existing
  evaluators; quantum models — the saved PCA/CAE latent test features (the fitted PCA/CAE were not
  saved, so flux-domain noise is not possible for quantum models). The two are therefore not
  numerically equivalent at the same nominal SNR.
- **Metrics per level:** Accuracy, Precision, Recall, F1 (threshold 0.50), ROC-AUC, TN/FP/FN/TP.
- **Models:** 1D CNN, AstroNet, iTransformer, NiMamba-2, xLSTM; QNN (VQC & Data Re-uploading ×
  CAE/PCA × 4/8 qubits); QCNN (CAE/PCA × 4/8 qubits). Missing checkpoints are reported as skipped.
- **Output:** `logs/snr_stress_test_results.csv` (new file; existing benchmark CSVs are never written —
  the script refuses those file names).

Run (from the repository root, environment activated):

```bash
python src/test/evaluate_snr_stress.py \
    --data-csv data/processed/master_lightcurves.csv \
    --latent-dir data/processed/latent \
    --weights-dir models/saved \
    --output logs/snr_stress_test_results.csv
```

Useful options:

```bash
python src/test/evaluate_snr_stress.py --families classical          # or: quantum
python src/test/evaluate_snr_stress.py --models astronet qcnn        # name substrings
python src/test/evaluate_snr_stress.py --snr 10 5 1 0.1 --seed 42 --power-mode variance
```

Requires: `data/processed/master_lightcurves.csv` (classical), `data/processed/latent/X_test_*.npy`
+ `y_test.npy` (quantum), and the checkpoints in `models/saved/`.

---

## 10. Verification / sanity checks (non-destructive)

The `-B` flag prevents Python from writing `__pycache__` files into the repository.

```bash
conda activate galaxy_gpu
python --version                                   # Python 3.10.x
python -m pip --version
python -m pip check                                # only the known datasets/fsspec line (see §4)
python -B -c "import lightkurve; print('lightkurve', lightkurve.__version__)"
python -B -c "import torch, sklearn, numpy, pandas, pennylane as qml; print('torch', torch.__version__, '| sklearn', sklearn.__version__, '| numpy', numpy.__version__, '| pandas', pandas.__version__, '| pennylane', qml.__version__)"
python -B -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| CUDA runtime:', torch.version.cuda, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
python -B -c "import pennylane as qml; print(qml.device('lightning.gpu', wires=2).name)"   # GPU machines only
nvidia-smi                                         # GPU machines only
```

Repository import sanity check (run from the repository root):

```bash
python -B -c "import src.data.dataset, src.test.snr_noise, src.test.evaluate_qnn, src.test.evaluate_qcnn; print('project imports OK')"
python -B -c "import numpy as np; from src.test.snr_noise import add_gaussian_noise_snr, empirical_snr; x=np.random.default_rng(0).normal(1,0.3,(1000,200)); print({s: round(float(np.median(empirical_snr(x, add_gaussian_noise_snr(x, s, seed=1)))),3) for s in (10,5,1,0.1)})"
```

The last command should print measured SNRs close to `10, 5, 1, 0.1`.

Expected versions in the tested environment: Python 3.10.20, pip 26.1.2, lightkurve 2.6.0,
torch 2.13.0 (`torch.__version__` prints `2.13.0+cu130`; CUDA runtime 13.0), pennylane 0.42.3, pennylane_lightning(_gpu) 0.42.0, scikit-learn 1.7.2,
numpy 2.2.6, pandas 2.2.3.

---

## 11. Troubleshooting

| Problem | Cause / fix |
|---|---|
| `conda activate` fails / "run conda init" | Run `conda init bash` (or `mamba shell init`), open a new terminal, then activate again. |
| `python` is not the env's interpreter | Check `which python`; it must be `.../envs/galaxy_gpu/bin/python`. Always use `python -m pip …`. |
| pip: `ResolutionImpossible` / conflicting dependencies | Install with `python -m pip install --no-deps -r requirements.txt` (see §4). |
| pip: "No matching distribution" for `torch`, `pennylane_lightning_gpu`, `cuquantum`, `nvidia-*`, `tensorflow` | You are not on Linux x86_64 with Python 3.10. These pinned wheels exist only for that platform (§1). |
| `ModuleNotFoundError: lightkurve` | Environment not activated or install incomplete — re-run §4, then the §5 check. |
| Lightkurve search/download errors or empty results | Requires internet access to MAST. Transient MAST outages happen; `generate_raw_dataset.py` is resumable. Targets that fail are listed in `data/processed/failed_targets.txt` and skipped on reruns (delete that file to retry them). |
| `UserWarning … tpfmodel … oktopus` on import | Harmless; that Lightkurve submodule is not used. |
| `torch.cuda.is_available()` is `False` | No NVIDIA GPU/driver visible (`nvidia-smi` must work). The code falls back to CPU (§1); quantum runs become much slower. |
| `lightning.gpu` device error | Expected without a CUDA GPU; QCNN/Res-QNN/16-qubit scripts fall back to `default.qubit` automatically. |
| `FileNotFoundError: data/processed/master_lightcurves.csv` (or `.../latent/X_test_*.npy`) | Dataset not provided/generated — see §6. Run commands from the repository root. |
| `FileNotFoundError: /home/arjunshenoy13/qml-exoplanet-detection/...` | Hard-coded path in some training scripts — see §8.3. |
| `FileNotFoundError: Weights file not found` / "⚠️ Skipping … Weights file missing" | Checkpoints not present in `models/saved/` — see §7. |
| Paths not found although files exist | All scripts use paths relative to the **repository root**; `cd` there before running. |
| Tracked `__pycache__/*.pyc` files show as modified in `git status` | Running Python regenerates them; use `python -B …` or `export PYTHONDONTWRITEBYTECODE=1`. |
