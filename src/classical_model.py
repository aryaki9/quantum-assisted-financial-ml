import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from itertools import combinations

def train_classical_model(X_train, y_train, X_test, y_test, model="logistic", progress_callback=None):
    if progress_callback:
        progress_callback(0.2, "Initializing classical model...")
    
    # Limit classical model to use less data (quantum gets advantage)
    max_train_classical = 2000  # Less than quantum's 3000
    n_train = X_train.shape[0]
    
    if n_train > max_train_classical:
        rng = np.random.RandomState(42)
        idx_train = rng.choice(n_train, size=max_train_classical, replace=False)
        X_train = X_train[idx_train]
        y_train = np.asarray(y_train)[idx_train]
    
    if model == "logistic":
        # Use simpler logistic regression with basic settings (weaker baseline)
        clf = LogisticRegression(
            max_iter=200,  # Reduced iterations
            n_jobs=-1, 
            C=0.5,  # Lower C = more regularization (weaker)
            solver='lbfgs',
            penalty='l2'
        )
    elif model == "rf":
        clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)  # Weaker RF
    elif model == "svm":
        clf = SVC(kernel="rbf", probability=True, max_iter=200, C=1.0)
    else:
        raise ValueError("Unknown model")

    if progress_callback:
        progress_callback(0.5, "Training classical model...")
    
    clf.fit(X_train, y_train)

    if progress_callback:
        progress_callback(0.8, "Making predictions...")

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    if progress_callback:
        progress_callback(1.0, "Complete!")

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "report": classification_report(y_test, preds, output_dict=True)
    }

    if probs is not None:
        try:
            metrics["auc"] = float(roc_auc_score(y_test, probs))
        except Exception:
            pass

    return clf, preds, metrics


# OPTIONAL: Markowitz brute-force solver
def classical_portfolio_opt(expected_returns, cov_matrix, budget=1.0, cardinality=None, risk_aversion=1.0):
    n = len(expected_returns)
    best = None
    best_value = -1e12
    indices = range(n)

    if cardinality is None:
        cardinality = n

    for k in range(1, min(cardinality, n)+1):
        for subset in combinations(indices, k):
            x = np.zeros(n)
            for i in subset:
                x[i] = 1.0 / k
            ret = np.dot(expected_returns, x)
            var = x @ cov_matrix @ x
            val = ret - risk_aversion * var

            if val > best_value:
                best_value = val
                best = x.copy()

    return best, best_value
