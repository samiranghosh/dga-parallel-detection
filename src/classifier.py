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
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from scipy import stats
from typing import Dict, Any, List


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


# ── Priority 1: 5-Feature Production Configuration ──

def compare_5v6_features(X_train, y_train, X_test, y_test,
                         n_estimators: int = 100) -> Dict[str, Dict]:
    """Compare 6-feature vs 5-feature (no Levenshtein) configurations.

    The 5-feature configuration was identified by E7 ablation as achieving
    higher accuracy (93.18% vs 92.60%) because Levenshtein distance between
    adjacent domains in shuffled datasets is noise.

    Returns:
        {'6_features': {metrics + train_time}, '5_features': {metrics + train_time}}
    """
    results = {}

    for label, n_feat in [('6_features', 6), ('5_features', 5)]:
        X_tr = X_train[:, :n_feat]
        X_te = X_test[:, :n_feat]

        t0 = time.perf_counter()
        model = train_random_forest(X_tr, y_train, n_estimators=n_estimators)
        train_time = time.perf_counter() - t0

        metrics = evaluate_model(model, X_te, y_test)
        metrics['train_time_sec'] = train_time
        metrics['n_features'] = n_feat
        results[label] = metrics

    return results


# ── Priority 3: Statistical Rigor — Cross-Validation ──

def cross_validate_model(X: np.ndarray, y: np.ndarray,
                         model_type: str = 'rf',
                         n_splits: int = 5,
                         n_estimators: int = 100,
                         random_state: int = 42) -> Dict[str, Any]:
    """Run stratified k-fold cross-validation with detailed metrics.

    Args:
        X: Feature matrix, shape (N, F).
        y: Labels, shape (N,).
        model_type: 'rf' for Random Forest, 'dt' for Decision Tree.
        n_splits: Number of CV folds.
        n_estimators: RF trees (ignored for DT).
        random_state: Seed for reproducibility.

    Returns:
        Dict with per-fold scores and summary statistics:
        {
            'accuracy_scores': [...],
            'f1_scores': [...],
            'accuracy_mean': float,
            'accuracy_std': float,
            'f1_mean': float,
            'f1_std': float,
            'fit_times': [...],
        }
    """
    if model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=n_estimators, n_jobs=-1, random_state=random_state
        )
    elif model_type == 'dt':
        model = DecisionTreeClassifier(random_state=random_state)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = ['accuracy', 'f1']
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring,
                                return_train_score=False)

    return {
        'accuracy_scores': cv_results['test_accuracy'].tolist(),
        'f1_scores': cv_results['test_f1'].tolist(),
        'accuracy_mean': float(cv_results['test_accuracy'].mean()),
        'accuracy_std': float(cv_results['test_accuracy'].std()),
        'f1_mean': float(cv_results['test_f1'].mean()),
        'f1_std': float(cv_results['test_f1'].std()),
        'fit_times': cv_results['fit_time'].tolist(),
    }


def paired_significance_test(X: np.ndarray, y: np.ndarray,
                             n_splits: int = 5,
                             n_estimators: int = 100,
                             random_state: int = 42) -> Dict[str, Any]:
    """Paired t-test comparing RF vs DT across the same CV folds.

    Uses the same fold splits for both models to ensure a fair comparison.
    This tests whether the RF accuracy advantage over DT is statistically
    significant (p < 0.05).

    Returns:
        Dict with per-fold scores, t-statistic, p-value, and interpretation.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators, n_jobs=-1, random_state=random_state
    )
    dt = DecisionTreeClassifier(random_state=random_state)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rf_scores = []
    dt_scores = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rf.fit(X_train, y_train)
        rf_scores.append(accuracy_score(y_test, rf.predict(X_test)))

        dt.fit(X_train, y_train)
        dt_scores.append(accuracy_score(y_test, dt.predict(X_test)))

    rf_arr = np.array(rf_scores)
    dt_arr = np.array(dt_scores)
    diffs = rf_arr - dt_arr

    t_stat, p_value = stats.ttest_rel(rf_scores, dt_scores)

    return {
        'rf_fold_scores': rf_scores,
        'dt_fold_scores': dt_scores,
        'rf_mean': float(rf_arr.mean()),
        'dt_mean': float(dt_arr.mean()),
        'rf_std': float(rf_arr.std()),
        'dt_std': float(dt_arr.std()),
        'diff_mean': float(diffs.mean()),
        'diff_std': float(diffs.std()),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant_at_005': bool(p_value < 0.05),
    }


def cross_validate_5v6(X: np.ndarray, y: np.ndarray,
                       n_splits: int = 5,
                       n_estimators: int = 100,
                       random_state: int = 42) -> Dict[str, Any]:
    """Paired t-test comparing 5-feature vs 6-feature RF on the same folds.

    Tests whether dropping Levenshtein statistically improves accuracy.

    Returns:
        Dict with per-fold scores for both configurations, t-stat, and p-value.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores_6 = []
    scores_5 = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 6-feature
        rf6 = RandomForestClassifier(
            n_estimators=n_estimators, n_jobs=-1, random_state=random_state
        )
        rf6.fit(X_train, y_train)
        scores_6.append(accuracy_score(y_test, rf6.predict(X_test)))

        # 5-feature (drop Levenshtein = last column)
        rf5 = RandomForestClassifier(
            n_estimators=n_estimators, n_jobs=-1, random_state=random_state
        )
        rf5.fit(X_train[:, :5], y_train)
        scores_5.append(accuracy_score(y_test, rf5.predict(X_test[:, :5])))

    arr_6 = np.array(scores_6)
    arr_5 = np.array(scores_5)
    diffs = arr_5 - arr_6  # positive = 5-feature is better

    t_stat, p_value = stats.ttest_rel(scores_5, scores_6)

    return {
        '6feat_fold_scores': scores_6,
        '5feat_fold_scores': scores_5,
        '6feat_mean': float(arr_6.mean()),
        '5feat_mean': float(arr_5.mean()),
        '6feat_std': float(arr_6.std()),
        '5feat_std': float(arr_5.std()),
        'improvement_mean': float(diffs.mean()),
        'improvement_std': float(diffs.std()),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        '5feat_significantly_better': bool(p_value < 0.05 and diffs.mean() > 0),
    }
