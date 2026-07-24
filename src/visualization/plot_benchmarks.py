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

CSV_PATH = "logs/classical_benchmark_results.csv"

def load_and_clean_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results CSV not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Filter out corrupt entries where model_name was written as a float string
    def is_valid_name(val):
        try:
            float(val)
            return False
        except ValueError:
            return True

    df = df[df['model_name'].astype(str).apply(is_valid_name)].copy()

    numeric_cols = ['test_acc', 'precision', 'recall', 'f1_score', 'roc_auc', 'train_time_sec']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.groupby('model_name').last().reset_index()
    df = df.sort_values('roc_auc', ascending=False).reset_index(drop=True)
    
    family_map = {
        'XenoPulse-Net': 'Hybrid Multi-Paradigm',
        '1D ResNet': 'Residual / CNN',
        'AstroNet': 'Dual-Branch CNN',
        '1D CNN': 'Standard CNN',
        'Selective-SSM': 'State Space (Mamba-1)',
        'TCN': 'Causal Conv',
        'BiLSTM': 'Recurrent',
        'NiMamba-2': 'State Space (Mamba-2)',
        'TimesNet': 'Periodicity / FFT',
        'PatchTST': 'Transformer',
        'xLSTM': 'Extended Recurrent',
        'iTransformer': 'Transformer'
    }
    df['family'] = df['model_name'].map(family_map).fillna('Other')
    return df

def plot_master_metrics(df):
    fig, ax = plt.subplots(figsize=(15, 7.5), dpi=300)

    x = np.arange(len(df))
    width = 0.26

    acc_normalized = df['test_acc'] / 100.0 if df['test_acc'].max() > 1.0 else df['test_acc']

    bars1 = ax.bar(x - width, df['roc_auc'], width, label='ROC-AUC', color='#2b5c8f', edgecolor='black', alpha=0.9)
    bars2 = ax.bar(x, df['f1_score'], width, label='F1-Score', color='#d95f02', edgecolor='black', alpha=0.9)
    bars3 = ax.bar(x + width, acc_normalized, width, label='Accuracy', color='#7570b3', edgecolor='black', alpha=0.9)

    ax.set_ylabel('Score (0.0 – 1.0)', fontsize=12, fontweight='bold')
    ax.set_title(f'Exoplanet Detection Benchmark: Performance Across {len(df)} Architectures', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(df['model_name'], rotation=35, ha='right', fontsize=11, fontweight='semibold')
    ax.set_ylim(0, 1.20)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.012, f"{yval:.3f}", 
                ha='center', va='bottom', fontsize=7.5, fontweight='bold', rotation=90)

    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.012, f"{yval:.3f}", 
                ha='center', va='bottom', fontsize=7.5, fontweight='bold', rotation=90)

    for bar in bars3:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.012, f"{yval * 100:.2f}%", 
                ha='center', va='bottom', fontsize=7.5, fontweight='bold', rotation=90, color='#381d63')

    plt.tight_layout()
    output_path = "reports/figures/classical_benchmark_metrics.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"📊 Saved Clean Metric Summary Plot: {output_path}")

def plot_auc_vs_runtime(df):
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    
    sns.scatterplot(
        data=df,
        x='train_time_sec',
        y='roc_auc',
        hue='family',
        style='family',
        s=220,
        ax=ax,
        palette='Set1'
    )

    ax.set_xscale('log')
    ax.set_xlabel('Training Runtime in Seconds (Log Scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('ROC-AUC Score', fontsize=12, fontweight='bold')
    ax.set_title('Computational Efficiency Trade-off: ROC-AUC vs Training Time', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    
    for _, row in df.iterrows():
        ax.annotate(
            row['model_name'],
            (row['train_time_sec'], row['roc_auc']),
            textcoords="offset points",
            xytext=(0, 9),
            ha='center',
            fontsize=9.5,
            fontweight='bold'
        )

    ax.set_ylim(0.40, 1.00)
    plt.legend(title='Architecture Family', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    
    output_path = "reports/figures/auc_vs_runtime.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"⏱️ Saved Clean Efficiency Trade-off Plot: {output_path}")

if __name__ == "__main__":
    df = load_and_clean_data(CSV_PATH)
    plot_master_metrics(df)
    plot_auc_vs_runtime(df)