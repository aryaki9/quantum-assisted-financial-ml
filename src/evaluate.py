# src/evaluate.py
"""
Top-level orchestrator to demonstrate the QML tasks:
- Classification: credit scoring / fraud detection (classical vs quantum kernel + VQC)
- Regression-demo: trading prediction (classical vs quantum kernel ridge)
- Portfolio optimization: classical brute-force vs QAOA
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.classical_model import train_classical_model, classical_portfolio_opt
from src.quantum_model import quantum_kernel_svm
from src.portfolio_qaoa import portfolio_qubo

def load_preprocessed(path_prefix="data/"):
    X_train = pd.read_csv(path_prefix + "X_train.csv")
    X_test = pd.read_csv(path_prefix + "X_test.csv")
    y_train = pd.read_csv(path_prefix + "y_train.csv").iloc[:,0].values
    y_test = pd.read_csv(path_prefix + "y_test.csv").iloc[:,0].values
    return X_train.values, X_test.values, y_train, y_test

def run_classification_demo():
    X_train, X_test, y_train, y_test = load_preprocessed()
    # scale (already scaled in notebook but safe)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)
    clf, preds, metrics = train_classical_model(X_train, y_train, X_test, y_test, model="rf")
    print("Classical RF metrics:", metrics["accuracy"])
    print("Running Quantum Kernel SVM (QSVM) ...")
    qclf, qpreds, qmetrics = quantum_kernel_svm(X_train, y_train, X_test, y_test)
    print("QSVM metrics:", qmetrics["accuracy"])
    return {"classical": metrics, "quantum_kernel": qmetrics}

def run_portfolio_demo():
    # Example: small synthetic expected_returns & covariance
    np.random.seed(0)
    n = 6
    mu = np.random.normal(0.05, 0.02, size=n)
    A = np.random.rand(n,n)
    cov = A.T @ A * 0.001
    print("Classical brute-force portfolio search ...")
    best, val = classical_portfolio_opt(mu, cov, cardinality=3, risk_aversion=0.1)
    print("Brute force selection:", best, val)
    print("Running QAOA ...")
    qres = portfolio_qubo(mu, cov, cardinality=3, risk_aversion=0.1, reps=1)
    print("QAOA selection:", qres["selection"], qres["score"])
    return {"classical": {"selection": best, "score": val}, "qaoa": qres}

if __name__ == "__main__":
    print("Running classification demo")
    run_classification_demo()
    print("Running portfolio demo")
    run_portfolio_demo()
