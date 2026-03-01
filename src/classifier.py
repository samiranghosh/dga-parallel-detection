"""
Classification Module
======================
Owner: Member 4 (Classifier & ML Evaluation)

Implements Random Forest (primary) and Decision Tree (comparison)
classifiers. RF provides Layer 2 parallelism via n_jobs=-1.

Experiments supported:
- E5: RF hyperparameter sweep (n_estimators)
- E6: DT vs RF comparison
- E7: Feature ablation
"""

import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from typing import Dict, Any


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                        n_estimators: int = 100,
                        n_jobs: int = -1,
                        random_state: int = 42) -> RandomForestClassifier:
    """Train a Random Forest classifier with parallel tree construction.

    This is Layer 2 parallelism — joblib distributes tree training
    across all available cores.

    Args:
        X_train: Feature matrix, shape (N, 6).
        y_train: Labels, shape (N,). 1=DGA, 0=Normal.
        n_estimators: Number of trees in the forest.
        n_jobs: Cores for joblib (-1 = all).
        random_state: Seed for reproducibility.

    Returns:
        Trained RandomForestClassifier.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    rf.fit(X_train, y_train)
    return rf


def train_decision_tree(X_train: np.ndarray, y_train: np.ndarray,
                        max_depth: int = None,
                        random_state: int = 42) -> DecisionTreeClassifier:
    """Train a single Decision Tree (mimicking J48 from the base paper).

    Uses max_depth pruning as the Python equivalent of C4.5 pruning.

    Args:
        X_train: Feature matrix, shape (N, 6).
        y_train: Labels.
        max_depth: Maximum tree depth (None = unlimited).
        random_state: Seed.

    Returns:
        Trained DecisionTreeClassifier.
    """
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state,
    )
    dt.fit(X_train, y_train)
    return dt


def evaluate_model(model, X_test: np.ndarray,
                   y_test: np.ndarray) -> Dict[str, Any]:
    """Evaluate a trained model on test data.

    Returns:
        Dict with keys: accuracy, precision, recall, f1,
        confusion_matrix, classification_report, inference_latency_ms.
    """
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_ms = (time.perf_counter() - t0) * 1000

    return {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred),
        'inference_latency_ms': inference_ms,
    }


def run_hyperparameter_sweep(X_train, y_train, X_test, y_test,
                             estimator_values: list = None) -> list:
    """Experiment E5: Vary n_estimators and record accuracy vs. time.

    Returns:
        List of dicts: [{n_estimators, accuracy, f1, train_time_sec}, ...]
    """
    if estimator_values is None:
        estimator_values = [50, 100, 200, 300, 500]

    results = []
    for n_est in estimator_values:
        t0 = time.perf_counter()
        model = train_random_forest(X_train, y_train, n_estimators=n_est)
        train_time = time.perf_counter() - t0
        metrics = evaluate_model(model, X_test, y_test)
        results.append({
            'n_estimators': n_est,
            'accuracy': metrics['accuracy'],
            'f1': metrics['f1'],
            'train_time_sec': train_time,
        })
    return results


def run_dt_vs_rf_comparison(X_train, y_train,
                            X_test, y_test) -> Dict[str, Dict]:
    """Experiment E6: Compare Decision Tree vs Random Forest.

    Returns:
        {'decision_tree': {metrics...}, 'random_forest': {metrics...}}
    """
    t0 = time.perf_counter()
    dt = train_decision_tree(X_train, y_train)
    dt_time = time.perf_counter() - t0
    dt_metrics = evaluate_model(dt, X_test, y_test)
    dt_metrics['train_time_sec'] = dt_time

    t0 = time.perf_counter()
    rf = train_random_forest(X_train, y_train)
    rf_time = time.perf_counter() - t0
    rf_metrics = evaluate_model(rf, X_test, y_test)
    rf_metrics['train_time_sec'] = rf_time

    return {'decision_tree': dt_metrics, 'random_forest': rf_metrics}


def run_feature_ablation(X_train, y_train, X_test, y_test,
                         feature_names: list = None) -> list:
    """Experiment E7: Train with subsets of 1-6 features.

    Returns:
        List of dicts: [{n_features, features_used, accuracy, f1}, ...]
    """
    if feature_names is None:
        feature_names = [
            'length', 'numerical_ratio', 'meaningful_word_ratio',
            'pronounceability', 'lms_percentage', 'levenshtein'
        ]

    results = []
    for n_feat in range(1, len(feature_names) + 1):
        used_features = feature_names[:n_feat]
        model = train_random_forest(X_train[:, :n_feat], y_train)
        metrics = evaluate_model(model, X_test[:, :n_feat], y_test)
        results.append({
            'n_features': n_feat,
            'features_used': used_features,
            'accuracy': metrics['accuracy'],
            'f1': metrics['f1'],
        })
    return results
