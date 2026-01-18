"""
Test script to verify quantum model outperforms classical on real credit risk data.
"""

import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

from src.data_loader import load_credit_risk_data
from src.quantum_model import train_quantum_model
from src.classical_model import train_classical_model

print("=" * 80)
print("QUANTUM VS CLASSICAL MODEL COMPARISON - REAL CREDIT RISK DATA")
print("=" * 80)

# Load real credit risk data
print("\n[1/4] Loading real credit risk dataset...")
X_train, X_test, y_train, y_test = load_credit_risk_data(
    test_size=0.2,
    random_state=42
)

print(f"\n✓ Data loaded successfully!")
print(f"  Training samples: {X_train.shape[0]}")
print(f"  Test samples: {X_test.shape[0]}")
print(f"  Features: {X_train.shape[1]}")

# Train quantum model
print("\n[2/4] Training Quantum Model...")
print("-" * 80)

def progress_callback(progress, message):
    print(f"  [{progress*100:.0f}%] {message}")

q_model, q_preds, q_metrics = train_quantum_model(
    X_train, y_train, X_test, y_test,
    progress_callback=progress_callback
)

print(f"\n✓ Quantum model trained!")
print(f"  Accuracy: {q_metrics['accuracy']:.4f} ({q_metrics['accuracy']*100:.2f}%)")

# Train classical model
print("\n[3/4] Training Classical Model...")
print("-" * 80)

c_model, c_preds, c_metrics = train_classical_model(
    X_train, y_train, X_test, y_test,
    model="logistic",
    progress_callback=progress_callback
)

print(f"\n✓ Classical model trained!")
print(f"  Accuracy: {c_metrics['accuracy']:.4f} ({c_metrics['accuracy']*100:.2f}%)")

# Compare results
print("\n[4/4] Comparison Results")
print("=" * 80)

print(f"\n📊 ACCURACY COMPARISON:")
print(f"  Quantum Model:    {q_metrics['accuracy']:.4f} ({q_metrics['accuracy']*100:.2f}%)")
print(f"  Classical Model:  {c_metrics['accuracy']:.4f} ({c_metrics['accuracy']*100:.2f}%)")

improvement = (q_metrics['accuracy'] - c_metrics['accuracy']) * 100
print(f"\n  Improvement:      {improvement:+.2f}%")

if q_metrics['accuracy'] > c_metrics['accuracy']:
    print(f"\n✅ SUCCESS: Quantum model outperforms classical by {improvement:.2f}%!")
elif q_metrics['accuracy'] == c_metrics['accuracy']:
    print(f"\n⚠️  TIED: Both models have equal performance")
else:
    print(f"\n❌ FAILED: Classical model outperforms quantum by {-improvement:.2f}%")
    print("   Consider adjusting quantum feature map or hyperparameters")

# Detailed metrics
print(f"\n📈 DETAILED METRICS:")
print(f"\nQuantum Model:")
if 'report' in q_metrics:
    report = q_metrics['report']
    for key in ['0', '1', 'macro avg', 'weighted avg']:
        if key in report:
            metrics = report[key]
            if isinstance(metrics, dict):
                print(f"  {key:12s}: precision={metrics.get('precision', 0):.3f}, recall={metrics.get('recall', 0):.3f}, f1={metrics.get('f1-score', 0):.3f}")

print(f"\nClassical Model:")
if 'report' in c_metrics:
    report = c_metrics['report']
    for key in ['0', '1', 'macro avg', 'weighted avg']:
        if key in report:
            metrics = report[key]
            if isinstance(metrics, dict):
                print(f"  {key:12s}: precision={metrics.get('precision', 0):.3f}, recall={metrics.get('recall', 0):.3f}, f1={metrics.get('f1-score', 0):.3f}")

print("\n" + "=" * 80)
print("Test complete! Run the dashboard to see interactive visualizations:")
print("  streamlit run dashboard/app.py")
print("=" * 80)
