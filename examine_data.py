import pandas as pd
import numpy as np

# Load the credit risk dataset
df = pd.read_csv('data/credit_risk_dataset.csv')

print("=" * 80)
print("CREDIT RISK DATASET ANALYSIS")
print("=" * 80)

print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")

print(f"\nColumns:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print(f"\nData Types:")
print(df.dtypes)

print(f"\nMissing Values:")
print(df.isnull().sum())

print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nLast column (likely target):")
target_col = df.columns[-1]
print(f"Column name: {target_col}")
print(f"Value counts:")
print(df[target_col].value_counts())
print(f"Class distribution: {df[target_col].value_counts(normalize=True)}")

print(f"\nNumeric summary:")
print(df.describe())

print("\n" + "=" * 80)
