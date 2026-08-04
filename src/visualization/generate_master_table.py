import os
import pandas as pd

def find_column(df, candidates):
    """Finds the first matching candidate column in df (case-insensitive & space-insensitive)."""
    normalized_cols = {col.strip().lower(): col for col in df.columns}
    for cand in candidates:
        cand_norm = cand.strip().lower()
        if cand_norm in normalized_cols:
            return normalized_cols[cand_norm]
    # Partial match fallback
    for cand in candidates:
        for norm_col, orig_col in normalized_cols.items():
            if cand.lower() in norm_col:
                return orig_col
    return None

def generate_reports():
    logs_dir = "logs"
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)

    # ----------------------------------------------------
    # 1. Classical Baseline Architectures Table
    # ----------------------------------------------------
    class_path = os.path.join(logs_dir, "classical_benchmark_results.csv")
    if os.path.exists(class_path):
        try:
            raw_c = pd.read_csv(class_path, on_bad_lines='skip', engine='python')
        except Exception as e:
            print(f"⚠️ Warning reading {class_path}: {e}")
            raw_c = pd.DataFrame()

        if not raw_c.empty:
            raw_c.columns = raw_c.columns.str.strip()

            col_mappings = {
                'Model Name': ['model_name', 'model', 'architecture', 'name'],
                'Accuracy (%)': ['test_acc', 'accuracy', 'acc', 'test_accuracy'],
                'Precision': ['precision', 'prec'],
                'Recall': ['recall', 'rec'],
                'F1-Score': ['f1_score', 'f1', 'f1-score'],
                'ROC-AUC': ['roc_auc', 'auc', 'roc_auc_score'],
                'Training Time (s)': ['training_time_sec', 'training_time', 'time_sec', 'time', 'execution_time_sec', 'runtime']
            }

            c_df = pd.DataFrame()
            for out_name, candidate_list in col_mappings.items():
                matched_col = find_column(raw_c, candidate_list)
                if matched_col:
                    c_df[out_name] = raw_c[matched_col]

            # Standardize fractional accuracy to percentage if needed
            if 'Accuracy (%)' in c_df.columns and pd.api.types.is_numeric_dtype(c_df['Accuracy (%)']):
                if c_df['Accuracy (%)'].max() <= 1.0:
                    c_df['Accuracy (%)'] = c_df['Accuracy (%)'] * 100
        else:
            c_df = pd.DataFrame()
    else:
        c_df = pd.DataFrame()

    # ----------------------------------------------------
    # 2. Quantum Architectures Table
    # ----------------------------------------------------
    q_files = [
        "qnn_benchmark_results.csv",
        "qnn_benchmark_16q_results.csv",
        "qcnn_benchmark_results.csv",
        "res_qnn_benchmark_results.csv"
    ]

    q_dfs = []
    for qf in q_files:
        qp = os.path.join(logs_dir, qf)
        if os.path.exists(qp):
            try:
                df = pd.read_csv(qp, on_bad_lines='skip', engine='python')
                df.columns = df.columns.str.strip()
                q_dfs.append(df)
            except Exception as e:
                print(f"⚠️ Warning reading {qp}: {e}")

    if q_dfs:
        raw_q = pd.concat(q_dfs, ignore_index=True)

        q_col_mappings = {
            'Architecture': ['architecture', 'model', 'model_name'],
            'Qubits': ['n_qubits', 'qubits', 'num_qubits'],
            'Latent Type': ['latent_type', 'latent', 'dim_red'],
            'Accuracy (%)': ['accuracy', 'acc', 'test_acc'],
            'Precision': ['precision'],
            'Recall': ['recall'],
            'F1-Score': ['f1_score', 'f1'],
            'ROC-AUC': ['roc_auc', 'auc'],
            'Training Time (s)': ['training_time_sec', 'training_time', 'time_sec', 'execution_time_sec', 'time']
        }

        q_df = pd.DataFrame()
        for out_name, candidate_list in q_col_mappings.items():
            matched_col = find_column(raw_q, candidate_list)
            if matched_col:
                q_df[out_name] = raw_q[matched_col]

        # Clean qubit counts
        if 'Qubits' in q_df.columns:
            q_df['Qubits'] = pd.to_numeric(q_df['Qubits'], errors='coerce').fillna(0).astype(int)

        # Convert accuracy fraction to percentage if needed
        if 'Accuracy (%)' in q_df.columns and pd.api.types.is_numeric_dtype(q_df['Accuracy (%)']):
            if q_df['Accuracy (%)'].max() <= 1.0:
                q_df['Accuracy (%)'] = q_df['Accuracy (%)'] * 100
    else:
        q_df = pd.DataFrame()

    # ----------------------------------------------------
    # 3. Write Separate Markdown Sections
    # ----------------------------------------------------
    md_path = os.path.join(reports_dir, "master_benchmark_summary.md")
    with open(md_path, "w") as f:
        f.write("# Master Model Benchmark Summary\n\n")
        f.write("Consolidated comparison of all Classical and Quantum architectures.\n\n")

        f.write("## 1. Classical Baseline Architectures\n\n")
        if not c_df.empty:
            f.write(c_df.to_markdown(index=False, floatfmt=".4f"))
        else:
            f.write("_No classical benchmark results found._\n")

        f.write("\n\n---\n\n")

        f.write("## 2. Quantum Architectures (QNN, QCNN, Res-QNN)\n\n")
        if not q_df.empty:
            f.write(q_df.to_markdown(index=False, floatfmt=".4f"))
        else:
            f.write("_No quantum benchmark results found._\n")

    print(f"✅ Generated two isolated benchmark tables successfully at: {md_path}")

if __name__ == "__main__":
    generate_reports()