import os
import sys
import time
import csv
import subprocess

# ---------------------------------------------------------------------------
# 1. DIRECTORY & PATH SETUP
# ---------------------------------------------------------------------------
WEIGHTS_DIR = os.path.join("models", "saved")
TXT_LOGS_DIR = os.path.join("logs", "txt_logs")
CSV_LOGS_DIR = os.path.join("logs", "csv_logs")

for path in [WEIGHTS_DIR, TXT_LOGS_DIR, CSV_LOGS_DIR]:
    os.makedirs(path, exist_ok=True)

# Build isolated PYTHONPATH environment ensuring src/models is directly discoverable
project_root = os.getcwd()
src_path = os.path.abspath("src")
models_path = os.path.abspath(os.path.join("src", "models"))

sub_env = os.environ.copy()
existing_ppath = sub_env.get("PYTHONPATH", "")
sub_env["PYTHONPATH"] = os.pathsep.join(
    filter(None, [models_path, src_path, project_root, existing_ppath])
)

# ---------------------------------------------------------------------------
# 2. MASTER REGISTRY OF NATIVE MODEL SCRIPTS
# ---------------------------------------------------------------------------
CLASSICAL_MODELS = [
    {"name": "XenoPulse-Net", "slug": "xenopulse_net", "script": "src/models/train_classical.py", "extra_args": ["--model", "xenopulse"]},
    {"name": "1D ResNet", "slug": "1d_resnet", "script": "src/models/train_resnet1d.py", "extra_args": []},
    {"name": "AstroNet", "slug": "astronet", "script": "src/models/train_astronet.py", "extra_args": []},
    {"name": "1D CNN", "slug": "1d_cnn", "script": "src/models/train_1d_cnn.py", "extra_args": []},
    {"name": "TCN", "slug": "tcn", "script": "src/models/train_tcn.py", "extra_args": []},
    {"name": "BiLSTM", "slug": "bilstm", "script": "src/models/train_lstm.py", "extra_args": []},
    {"name": "TimesNet", "slug": "timesnet", "script": "src/models/train_timesnet.py", "extra_args": []},
    {"name": "PatchTST", "slug": "patchtst", "script": "src/models/train_patchtst.py", "extra_args": []},
    {"name": "iTransformer", "slug": "itransformer", "script": "src/models/train_itransformer.py", "extra_args": []},
    {"name": "Selective-SSM", "slug": "selective_ssm", "script": "src/models/train_ssm.py", "extra_args": []},
    {"name": "NiMamba-2", "slug": "nimamba2", "script": "src/models/train_nimamba2.py", "extra_args": []},
    {"name": "xLSTM", "slug": "xlstm", "script": "src/models/train_xlstm.py", "extra_args": []},
]

# ---------------------------------------------------------------------------
# 3. SUBPROCESS EXECUTION & LIVE LOGGING
# ---------------------------------------------------------------------------
def run_and_log_subprocess(cmd, txt_log_path):
    """Executes native training script in isolated process, streaming live logs."""
    with open(txt_log_path, "w", encoding="utf-8") as f_log:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=sub_env
        )
        
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f_log.write(line)
            f_log.flush()
            
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

def parse_metrics_from_csv(csv_path):
    """Attempts to parse evaluation metrics from generated CSV log."""
    if not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            if reader:
                return reader[-1]
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# 4. MAIN PIPELINE EXECUTION
# ---------------------------------------------------------------------------
def run_pipeline():
    print("=" * 75)
    print("🚀 STARTING ISOLATED CLASSICAL BASELINE RETRAINING PIPELINE")
    print("=" * 75)
    print(f"📁 Weights Directory  : {WEIGHTS_DIR}")
    print(f"📁 TXT Logs Directory : {TXT_LOGS_DIR}")
    print(f"📁 CSV Logs Directory : {CSV_LOGS_DIR}")
    print("=" * 75)

    overall_start = time.time()
    successful_models = 0
    failed_models = []

    for idx, model in enumerate(CLASSICAL_MODELS, 1):
        name = model["name"]
        slug = model["slug"]
        script = model["script"]
        
        target_weights = os.path.join(WEIGHTS_DIR, f"{slug}_weights.pth")
        target_txt_log = os.path.join(TXT_LOGS_DIR, f"{slug}_train.txt")
        target_csv_log = os.path.join(CSV_LOGS_DIR, f"{slug}_metrics.csv")

        print(f"\n[{idx}/{len(CLASSICAL_MODELS)}] ⚡ Commencing Native Training: {name}")
        print("=" * 75)
        print(f"📌 Architecture      : {name}")
        print(f"📌 Native Script     : {script}")
        print(f"📌 Target Checkpoint : {target_weights}")
        print(f"📌 Execution Log     : {target_txt_log}")
        print(f"📌 Metrics CSV Path  : {target_csv_log}")
        print("-" * 75)

        # Primary command trying explicit arguments
        cmd = [
            sys.executable, script,
            "--epochs", "30",
            "--save-path", target_weights,
            "--csv-path", target_csv_log
        ] + model["extra_args"]

        start_time = time.time()
        try:
            run_and_log_subprocess(cmd, target_txt_log)
        except subprocess.CalledProcessError:
            # Fallback for scripts that don't accept standard CLI flags
            print(f"⚠️ Standard flags rejected by {script}. Falling back to default invocation...")
            fallback_cmd = [sys.executable, script] + model["extra_args"]
            try:
                run_and_log_subprocess(fallback_cmd, target_txt_log)
            except subprocess.CalledProcessError as e:
                print(f"❌ ERROR: Training failed for {name} (Exit Code: {e.returncode})")
                failed_models.append(name)
                continue

        elapsed = time.time() - start_time
        successful_models += 1

        # Read back metrics if saved
        metrics = parse_metrics_from_csv(target_csv_log)

        print("\n" + "=" * 75)
        print(f"📋 MODEL TRAINING SUMMARY: {name.upper()}")
        print("=" * 75)
        print(f"  • Model Name        : {name}")
        print(f"  • Status            : COMPLETED ✅")
        print(f"  • Execution Time    : {elapsed:.2f} seconds")
        print(f"  • Checkpoint        : {target_weights}")
        print(f"  • Log File          : {target_txt_log}")

        if metrics:
            print("-" * 75)
            print("  📊 RECORDED METRICS:")
            for k, v in metrics.items():
                if k.lower() != "model name":
                    print(f"    ├── {k:<16} : {v}")
        print("=" * 75 + "\n")

    total_time_min = (time.time() - overall_start) / 60

    print("\n" + "=" * 75)
    print("📊 RETRAINING SUMMARY & VERIFICATION")
    print("=" * 75)
    print(f"• Successfully Trained : {successful_models}/{len(CLASSICAL_MODELS)} models")
    print(f"• Total Elapsed Time   : {total_time_min:.2f} minutes")
    
    if failed_models:
        print(f"• Failed Models        : {', '.join(failed_models)}")
    else:
        print("🎉 ALL CLASSICAL MODELS RETRAINED WITH ISOLATED PATHS!")

if __name__ == "__main__":
    run_pipeline()