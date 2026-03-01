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
    # TODO: Implement
    raise NotImplementedError


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
    # TODO: Implement
    raise NotImplementedError


def evaluate_model(model, X_test: np.ndarray,
                   y_test: np.ndarray) -> Dict[str, Any]:
    """Evaluate a trained model on test data.

    Returns:
        Dict with keys: accuracy, precision, recall, f1,
        confusion_matrix, classification_report, inference_latency_ms.
    """
    # TODO: Implement
    # 1. Predict on X_test, measure inference time
    # 2. Compute all metrics
    # 3. Return as dict
    raise NotImplementedError


def run_hyperparameter_sweep(X_train, y_train, X_test, y_test,
                             estimator_values: list = None) -> list:
    """Experiment E5: Vary n_estimators and record accuracy vs. time.

    Returns:
        List of dicts: [{n_estimators, accuracy, f1, train_time_sec}, ...]
    """
    # TODO: Implement
    raise NotImplementedError


def run_dt_vs_rf_comparison(X_train, y_train,
                            X_test, y_test) -> Dict[str, Dict]:
    """Experiment E6: Compare Decision Tree vs Random Forest.

    Returns:
        {'decision_tree': {metrics...}, 'random_forest': {metrics...}}
    """
    # TODO: Implement
    raise NotImplementedError


def run_feature_ablation(X_train, y_train, X_test, y_test,
                         feature_names: list = None) -> list:
    """Experiment E7: Train with subsets of 1-6 features.

    Returns:
        List of dicts: [{n_features, features_used, accuracy, f1}, ...]
    """
    # TODO: Implement
    # For i in 1..6: train RF on X[:, :i], evaluate, record
    raise NotImplementedError
