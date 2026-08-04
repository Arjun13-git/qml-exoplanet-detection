import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("reports/figures", exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

def find_column(df, candidates):
    """Finds the first matching candidate column in df (case-insensitive & space-insensitive)."""
    normalized_cols = {str(col).strip().lower(): col for col in df.columns}
    for cand in candidates:
        cand_norm = cand.strip().lower()
        if cand_norm in normalized_cols:
            return normalized_cols[cand_norm]
    for cand in candidates:
        for norm_col, orig_col in normalized_cols.items():
            if cand.lower() in norm_col:
                return orig_col
    return None

def load_classical_data():
    class_path = "logs/classical_benchmark_results.csv"
    if not os.path.exists(class_path):
        return pd.DataFrame()

    df = pd.read_csv(class_path, on_bad_lines='skip', engine='python')
    df.columns = df.columns.str.strip()

    model_col = find_column(df, ['model_name', 'model', 'architecture', 'name'])
    acc_col = find_column(df, ['test_acc', 'accuracy', 'acc', 'test_accuracy'])
    prec_col = find_column(df, ['precision', 'prec'])
    rec_col = find_column(df, ['recall', 'rec'])
    f1_col = find_column(df, ['f1_score', 'f1', 'f1-score'])
    auc_col = find_column(df, ['roc_auc', 'auc', 'roc_auc_score'])
    time_col = find_column(df, ['train_time_sec', 'training_time_sec', 'training_time', 'time_sec', 'time'])

    c_df = pd.DataFrame()
    c_df['model_name'] = df[model_col].astype(str).str.strip()
    c_df['test_acc'] = pd.to_numeric(df[acc_col], errors='coerce') if acc_col else np.nan
    c_df['precision'] = pd.to_numeric(df[prec_col], errors='coerce') if prec_col else np.nan
    c_df['recall'] = pd.to_numeric(df[rec_col], errors='coerce') if rec_col else np.nan
    c_df['f1_score'] = pd.to_numeric(df[f1_col], errors='coerce') if f1_col else np.nan
    c_df['roc_auc'] = pd.to_numeric(df[auc_col], errors='coerce') if auc_col else np.nan
    c_df['train_time_sec'] = pd.to_numeric(df[time_col], errors='coerce') if time_col else np.nan

    # Target Classical Models filter
    allowed_classical = {
        'itransformer': 'iTransformer',
        'astronet': 'AstroNet',
        'xenopulse-net': 'XenoPulse-Net',
        'xenopulse': 'XenoPulse-Net',
        'xlstm': 'xLSTM',
        'nimamba-2': 'NiMamba-2',
        'nimamba2': 'NiMamba-2',
        '1d cnn': '1D CNN',
        '1d-cnn': '1D CNN',
        '1d_cnn': '1D CNN'
    }

    c_df['norm_key'] = c_df['model_name'].str.lower()
    c_df = c_df[c_df['norm_key'].isin(allowed_classical.keys())].copy()
    c_df['model_name'] = c_df['norm_key'].map(allowed_classical)
    c_df['category'] = 'Classical ML'
    c_df.drop(columns=['norm_key'], inplace=True)
    c_df = c_df.groupby('model_name').last().reset_index()

    return c_df

def load_quantum_data():
    q_files = [
        "qnn_benchmark_results.csv",
        "qcnn_benchmark_results.csv"
    ]

    q_dfs = []
    for qf in q_files:
        qp = os.path.join("logs", qf)
        if os.path.exists(qp):
            try:
                df = pd.read_csv(qp, on_bad_lines='skip', engine='python')
                df.columns = df.columns.str.strip()
                q_dfs.append(df)
            except Exception:
                pass

    if not q_dfs:
        return pd.DataFrame()

    raw_q = pd.concat(q_dfs, ignore_index=True)

    arch_col = find_column(raw_q, ['architecture', 'model', 'model_name'])
    qubit_col = find_column(raw_q, ['n_qubits', 'qubits', 'num_qubits'])
    latent_col = find_column(raw_q, ['latent_type', 'latent', 'dim_red'])
    acc_col = find_column(raw_q, ['accuracy', 'acc', 'test_acc'])
    prec_col = find_column(raw_q, ['precision', 'prec'])
    rec_col = find_column(raw_q, ['recall', 'rec'])
    f1_col = find_column(raw_q, ['f1_score', 'f1', 'f1-score'])
    auc_col = find_column(raw_q, ['roc_auc', 'auc'])
    time_col = find_column(raw_q, ['training_time_sec', 'training_time', 'time_sec', 'time'])

    q_df = pd.DataFrame()
    q_df['arch_raw'] = raw_q[arch_col].astype(str).str.strip() if arch_col else ''
    q_df['qubits'] = pd.to_numeric(raw_q[qubit_col], errors='coerce').fillna(0).astype(int) if qubit_col else 0
    q_df['latent'] = raw_q[latent_col].astype(str).str.strip().str.upper() if latent_col else ''
    q_df['test_acc'] = pd.to_numeric(raw_q[acc_col], errors='coerce') if acc_col else np.nan
    q_df['precision'] = pd.to_numeric(raw_q[prec_col], errors='coerce') if prec_col else np.nan
    q_df['recall'] = pd.to_numeric(raw_q[rec_col], errors='coerce') if rec_col else np.nan
    q_df['f1_score'] = pd.to_numeric(raw_q[f1_col], errors='coerce') if f1_col else np.nan
    q_df['roc_auc'] = pd.to_numeric(raw_q[auc_col], errors='coerce') if auc_col else np.nan
    q_df['train_time_sec'] = pd.to_numeric(raw_q[time_col], errors='coerce') if time_col else np.nan

    # Filter strictly for 4 and 8 Qubits, and PCA/CAE Latents
    q_df = q_df[q_df['qubits'].isin([4, 8])].copy()
    q_df = q_df[q_df['latent'].isin(['PCA', 'CAE'])].copy()

    # Standardize Model Display Names
    def construct_q_name(row):
        arch = row['arch_raw'].lower()
        q_str = f"{row['qubits']}Q"
        latent_str = row['latent']

        if 'data_reuploading' in arch or 'reuploading' in arch or 'data-reuploading' in arch:
            return f"Data-Reuploading-{latent_str}-{q_str}"
        elif 'vqc' in arch:
            return f"VQC-{latent_str}-{q_str}"
        elif 'qcnn' in arch:
            return f"QCNN-{latent_str}-{q_str}"
        else:
            return f"QML-{latent_str}-{q_str}"

    q_df['model_name'] = q_df.apply(construct_q_name, axis=1)
    q_df['category'] = q_df['qubits'].apply(lambda q: f"Quantum ML ({q}Q)")
    
    q_df = q_df.groupby('model_name').last().reset_index()
    return q_df[['model_name', 'test_acc', 'precision', 'recall', 'f1_score', 'roc_auc', 'train_time_sec', 'category']]

def plot_benchmark_metrics(df):
    fig, ax = plt.subplots(figsize=(18, 8), dpi=300)

    x = np.arange(len(df))
    width = 0.25

    acc_normalized = df['test_acc'] / 100.0 if df['test_acc'].max() > 1.0 else df['test_acc']

    bars1 = ax.bar(x - width, df['roc_auc'], width, label='ROC-AUC', color='#2b5c8f', edgecolor='black', alpha=0.9)
    bars2 = ax.bar(x, df['f1_score'], width, label='F1-Score', color='#d95f02', edgecolor='black', alpha=0.9)
    bars3 = ax.bar(x + width, acc_normalized, width, label='Accuracy', color='#7570b3', edgecolor='black', alpha=0.9)

    ax.set_ylabel('Score (0.0 – 1.0)', fontsize=12, fontweight='bold')
    ax.set_title(f'Combined Classical & Quantum Exoplanet Detection Benchmark ({len(df)} Models)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(df['model_name'], rotation=45, ha='right', fontsize=10, fontweight='semibold')
    ax.set_ylim(0, 1.20)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    for bar in bars1:
        yval = bar.get_height()
        if not np.isnan(yval):
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.012, f"{yval:.3f}", 
                    ha='center', va='bottom', fontsize=7.0, fontweight='bold', rotation=90)

    for bar in bars2:
        yval = bar.get_height()
        if not np.isnan(yval):
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.012, f"{yval:.3f}", 
                    ha='center', va='bottom', fontsize=7.0, fontweight='bold', rotation=90)

    for bar in bars3:
        yval = bar.get_height()
        if not np.isnan(yval):
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.012, f"{yval * 100:.1f}%", 
                    ha='center', va='bottom', fontsize=7.0, fontweight='bold', rotation=90, color='#381d63')

    plt.tight_layout()
    output_path = "reports/figures/benchmark_metrics.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"📊 Saved Combined Metric Summary Plot: {output_path}")

def plot_auc_vs_runtime(df):
    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=300)
    
    sns.scatterplot(
        data=df,
        x='train_time_sec',
        y='roc_auc',
        hue='category',
        style='category',
        s=200,
        ax=ax,
        palette='Set1'
    )

    ax.set_xscale('log')
    ax.set_xlabel('Training Runtime in Seconds (Log Scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('ROC-AUC Score', fontsize=12, fontweight='bold')
    ax.set_title('Computational Efficiency Trade-off: ROC-AUC vs Training Time (Classical + QML)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    
    for _, row in df.iterrows():
        if not np.isnan(row['train_time_sec']) and not np.isnan(row['roc_auc']):
            ax.annotate(
                row['model_name'],
                (row['train_time_sec'], row['roc_auc']),
                textcoords="offset points",
                xytext=(0, 8),
                ha='center',
                fontsize=8.5,
                fontweight='bold'
            )

    ax.set_ylim(0.35, 1.02)
    plt.legend(title='Model Category', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    
    output_path = "reports/figures/auc_vs_runtime.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"⏱️ Saved Combined Efficiency Trade-off Plot: {output_path}")

if __name__ == "__main__":
    c_df = load_classical_data()
    q_df = load_quantum_data()

    combined_df = pd.concat([c_df, q_df], ignore_index=True)
    
    if combined_df.empty:
        print("❌ No matching classical or quantum model data found.")
    else:
        # Sort classical models first, then quantum
        combined_df['sort_order'] = combined_df['category'].apply(
            lambda cat: 0 if 'Classical' in cat else (1 if '4Q' in cat else 2)
        )
        combined_df = combined_df.sort_values(['sort_order', 'roc_auc'], ascending=[True, False]).reset_index(drop=True)

        plot_benchmark_metrics(combined_df)
        plot_auc_vs_runtime(combined_df)