"""
Script to generate and save the synthetic non-linear dataset.
Run this to create a CSV file with the dataset.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dashboard.app import generate_synthetic_nonlinear_data

# Generate the dataset
print("Generating synthetic non-linear dataset...")
df = generate_synthetic_nonlinear_data(n_samples=10000, n_features=10, random_state=42)

# Save to data directory
output_path = Path(__file__).parent / "data" / "synthetic_nonlinear_dataset.csv"
output_path.parent.mkdir(exist_ok=True)
df.to_csv(output_path, index=False)

print(f"Dataset saved to: {output_path}")
print(f"Shape: {df.shape}")
print(f"Target distribution:")
print(df['target'].value_counts())
print(f"Class 1 rate: {df['target'].mean():.2%}")

