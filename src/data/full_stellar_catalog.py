import pandas as pd

catalog_path = 'data/raw/full_stellar_catalog.csv'
df = pd.read_csv(catalog_path)

# Identify shifted rows (where label is a Julian date instead of 0 or 1)
shifted_mask = df['label'] > 1.0

print(f"Fixing {shifted_mask.sum()} shifted false-positive rows...")

# Shift values back into correct columns for corrupted rows
df.loc[shifted_mask, 'epoch'] = df.loc[shifted_mask, 'label']
df.loc[shifted_mask, 'label'] = 0.0

# Save back to disk
df.to_csv(catalog_path, index=False)
print("Catalog repaired successfully! New class distribution:")
print(df['label'].value_counts())