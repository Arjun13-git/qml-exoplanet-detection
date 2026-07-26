import os
import glob
import pandas as pd

def consolidate_benchmarks():
    logs_dir = "logs"
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)

    csv_files = [
        "classical_benchmark_results.csv",
        "qnn_benchmark_results.csv",
        "qnn_benchmark_16q_results.csv",
        "qcnn_benchmark_results.csv",
        "res_qnn_benchmark_results.csv"
    ]

    all_dfs = []

    for fname in csv_files:
        fpath = os.path.join(logs_dir, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            all_dfs.append(df)
            print(f"Loaded: {fpath}")

    if not all_dfs:
        print("No CSV log files found in logs/")
        return

    master_df = pd.concat(all_dfs, ignore_index=True)

    # Standardize Column Names
    time_col = [c for c in master_df.columns if "time" in c.lower()]
    if time_col:
        master_df.rename(columns={time_col[0]: "training_time_sec"}, inplace=True)

    # Fill NaNs for non-quantum models
    if "n_qubits" in master_df.columns:
        master_df["n_qubits"] = master_df["n_qubits"].fillna("-")
    if "latent_type" in master_df.columns:
        master_df["latent_type"] = master_df["latent_type"].fillna("-")

    # Save consolidated CSV
    master_csv_path = os.path.join(logs_dir, "master_benchmark_results.csv")
    master_df.to_csv(master_csv_path, index=False)
    print(f"\n✅ Consolidated CSV saved to: {master_csv_path}")

    # Build Markdown Report
    md_path = os.path.join(reports_dir, "master_benchmark_summary.md")
    with open(md_path, "w") as f:
        f.write("# Master Model Benchmark Summary\n\n")
        f.write("Consolidated comparison of all Classical, QNN, QCNN, and Res-QNN architectures.\n\n")
        
        f.write("## 1. Classical Baseline Architectures\n\n")
        classical_df = master_df[master_df["architecture"].str.contains("classical|1d_cnn|astronet|resnet|tcn|lstm|patchtst|timesnet|itransformer|ssm|mamba", case=False, na=False)]
        if not classical_df.empty:
            f.write(classical_df.to_markdown(index=False))
            f.write("\n\n")

        f.write("## 2. Quantum Architectures (QNN, QCNN, Res-QNN)\n\n")
        quantum_df = master_df[~master_df.index.isin(classical_df.index)]
        if not quantum_df.empty:
            f.write(quantum_df.to_markdown(index=False))
            f.write("\n\n")

    print(f"✅ Master Markdown Report saved to: {md_path}")

if __name__ == "__main__":
    consolidate_benchmarks()