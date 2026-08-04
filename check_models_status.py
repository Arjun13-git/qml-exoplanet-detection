import os
import csv

MODELS = [
    ("XenoPulse-Net", "xenopulse_net"),
    ("1D ResNet", "1d_resnet"),
    ("AstroNet", "astronet"),
    ("1D CNN", "1d_cnn"),
    ("TCN", "tcn"),
    ("BiLSTM", "bilstm"),
    ("TimesNet", "timesnet"),
    ("PatchTST", "patchtst"),
    ("iTransformer", "itransformer"),
    ("Selective-SSM", "selective_ssm"),
    ("NiMamba-2", "nimamba2"),
    ("xLSTM", "xlstm"),
]

print("=" * 85)
print(f"{'MODEL NAME':<16} | {'WEIGHTS FILE':<26} | {'SIZE':<9} | {'ACCURACY':<10} | {'STATUS'}")
print("=" * 85)

completed_count = 0

for name, slug in MODELS:
    weight_path = os.path.join("models", "saved", f"{slug}_weights.pth")
    csv_path = os.path.join("logs", "csv_logs", f"{slug}_metrics.csv")
    
    weight_exists = os.path.exists(weight_path) and os.path.getsize(weight_path) > 0
    size_str = "N/A"
    acc_str = "N/A"
    status = "❌ INCOMPLETE"
    
    if weight_exists:
        size_mb = os.path.getsize(weight_path) / (1024 * 1024)
        size_str = f"{size_mb:.2f} MB"
        status = "✅ SAVED"
        completed_count += 1

    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = list(csv.DictReader(f))
                if reader:
                    row = reader[-1]
                    acc_val = row.get("Accuracy (%)") or row.get("Accuracy")
                    if acc_val:
                        acc_str = f"{float(acc_val):.2f}%"
        except Exception:
            pass

    weight_filename = f"{slug}_weights.pth"
    print(f"{name:<16} | {weight_filename:<26} | {size_str:<9} | {acc_str:<10} | {status}")

print("=" * 85)
print(f"📊 SUMMARY: {completed_count}/{len(MODELS)} models fully retrained and saved.")
print("=" * 85)