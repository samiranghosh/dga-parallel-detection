"""
Benchmarking & Profiling Module
================================
Owner: Member 5 (Benchmarking & Report)

Runs the full experiment suite (E1-E8) as defined in the P1 design
document Section 15. Collects timing, CPU utilization, IPC overhead,
and memory usage. Generates plots for the P3 deliverable.

Experiments:
  E1 — Strong scaling (K = 1,2,4,8,16)
  E2 — High-core efficiency (K = 8,12,16)
  E3 — Dataset scaling (N = 10K..1M)
  E4 — Chunk size sweep
  E5 — RF hyperparameter (trees = 50..500)
  E6 — DT vs RF comparison
  E7 — Feature ablation (1..6 features)
  E8 — Weak scaling (N/K = 125K constant)
"""

import time
import json
import os
import pickle
import numpy as np
import psutil
import threading
from typing import Dict, List, Any


# ── Timing Utilities ──

def measure_wall_time(func, *args, **kwargs):
    """Measure wall-clock execution time of a function.

    Returns:
        Tuple of (result, elapsed_seconds).
    """
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return result, elapsed


class CPUMonitor:
    """Background thread that samples per-core CPU utilization."""

    def __init__(self, interval: float = 0.5):
        """
        Args:
            interval: Sampling interval in seconds.
        """
        self.interval = interval
        self.samples = []   # List of lists: [[core0%, core1%, ...], ...]
        self._stop = threading.Event()
        self._thread = None

    def _monitor_loop(self):
        while not self._stop.is_set():
            sample = psutil.cpu_percent(interval=None, percpu=True)
            self.samples.append(sample)
            time.sleep(self.interval)

    def start(self):
        """Start background CPU monitoring."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> List[List[float]]:
        """Stop monitoring and return all samples.

        Returns:
            List of per-core utilization snapshots.
        """
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return self.samples

    def summary(self) -> Dict[str, float]:
        """Compute summary statistics from collected samples.

        Returns:
            Dict with avg_utilization, max_utilization, min_utilization.
        """
        if not self.samples:
            return {'avg_utilization': 0.0, 'max_utilization': 0.0, 'min_utilization': 0.0}
        all_vals = [v for sample in self.samples for v in sample]
        return {
            'avg_utilization': float(np.mean(all_vals)),
            'max_utilization': float(np.max(all_vals)),
            'min_utilization': float(np.min(all_vals)),
        }


def measure_ipc_overhead(chunks: list, results: list) -> Dict[str, float]:
    """Instrument pickle serialization/deserialization overhead.

    Validates the theoretical IPC estimate from P1 Section 2.3.1.

    Returns:
        Dict with serialize_time_sec, deserialize_time_sec,
        total_overhead_sec, overhead_percentage.
    """
    serialize_time = 0.0
    deserialize_time = 0.0

    for chunk in chunks:
        t0 = time.perf_counter()
        data = pickle.dumps(chunk)
        serialize_time += time.perf_counter() - t0

    for result in results:
        t0 = time.perf_counter()
        data = pickle.dumps(result)
        t1 = time.perf_counter()
        pickle.loads(data)
        deserialize_time += time.perf_counter() - t1

    total = serialize_time + deserialize_time
    # Compute overhead percentage against a rough baseline (estimation)
    return {
        'serialize_time_sec': serialize_time,
        'deserialize_time_sec': deserialize_time,
        'total_overhead_sec': total,
    }


def measure_memory_usage() -> Dict[str, float]:
    """Snapshot current memory usage.

    Returns:
        Dict with rss_mb, vms_mb, percent, available_mb.
    """
    proc = psutil.Process()
    mem_info = proc.memory_info()
    vm = psutil.virtual_memory()
    return {
        'rss_mb': mem_info.rss / (1024 ** 2),
        'vms_mb': mem_info.vms / (1024 ** 2),
        'percent': proc.memory_percent(),
        'available_mb': vm.available / (1024 ** 2),
    }


# ── Experiment Runners ──

def run_experiment_e1(domain_list, dictionary, ngram_table,
                      k_values: list = None, reps: int = 5) -> list:
    """E1: Strong scaling — fixed N, vary K.

    Returns:
        List of dicts: [{k, rep, time_sec, speedup, cpu_util}, ...]
    """
    from src.features import extract_all_sequential
    from src.parallel_engine import parallel_extract_features

    if k_values is None:
        k_values = [1, 2, 4, 8]

    results = []

    # Baseline: sequential time (used for speedup calculation)
    print(f"  [E1] Running sequential baseline ({reps} reps)...")
    seq_times = []
    for _ in range(reps):
        _, t = measure_wall_time(extract_all_sequential, domain_list, dictionary, ngram_table)
        seq_times.append(t)
    seq_baseline = float(np.median(seq_times))
    print(f"  [E1] Sequential baseline: {seq_baseline:.2f}s")

    for k in k_values:
        print(f"  [E1] K={k}...")
        for rep in range(reps):
            monitor = CPUMonitor(interval=0.5)
            monitor.start()
            _, elapsed = measure_wall_time(parallel_extract_features, domain_list, k, dictionary, ngram_table)
            cpu_stats = monitor.stop()
            cpu_summary = CPUMonitor().summary() if not cpu_stats else {'avg_utilization': float(np.mean([v for s in cpu_stats for v in s]))}
            results.append({
                'k': k,
                'rep': rep,
                'time_sec': elapsed,
                'speedup': seq_baseline / elapsed if elapsed > 0 else 0,
                'cpu_util_avg': cpu_summary.get('avg_utilization', 0),
            })

    return results


def run_experiment_e3(domain_list, dictionary, ngram_table,
                      n_values: list = None, k: int = 8,
                      reps: int = 3) -> list:
    """E3: Dataset scaling — fixed K, vary N.

    Returns:
        List of dicts: [{n, rep, time_sec, throughput}, ...]
    """
    from src.parallel_engine import parallel_extract_features

    if n_values is None:
        n_values = [10000, 50000, 100000, 500000, 1000000]

    results = []
    for n in n_values:
        subset = domain_list[:min(n, len(domain_list))]
        actual_k = min(k, len(subset))
        print(f"  [E3] N={len(subset)}, K={actual_k}...")
        for rep in range(reps):
            _, elapsed = measure_wall_time(parallel_extract_features, subset, actual_k, dictionary, ngram_table)
            results.append({
                'n': len(subset),
                'rep': rep,
                'time_sec': elapsed,
                'throughput': len(subset) / elapsed if elapsed > 0 else 0,
            })
    return results


def run_experiment_e8(domain_list, dictionary, ngram_table,
                      domains_per_worker: int = 125000,
                      k_values: list = None, reps: int = 3) -> list:
    """E8: Weak scaling — N/K constant, vary K.

    Returns:
        List of dicts: [{k, n, rep, time_sec}, ...]
    """
    from src.parallel_engine import parallel_extract_features

    if k_values is None:
        k_values = [1, 2, 4, 8]

    results = []
    for k in k_values:
        n = min(domains_per_worker * k, len(domain_list))
        subset = domain_list[:n]
        print(f"  [E8] K={k}, N={n}...")
        for rep in range(reps):
            _, elapsed = measure_wall_time(parallel_extract_features, subset, k, dictionary, ngram_table)
            results.append({'k': k, 'n': n, 'rep': rep, 'time_sec': elapsed})
    return results


# ── Plotting ──

def plot_speedup_curve(results_e1: list, output_path: str):
    """Plot 1: Measured speedup vs Amdahl's theoretical curve (P=0.95)."""
    import matplotlib.pyplot as plt

    k_values = sorted(set(r['k'] for r in results_e1))
    avg_speedup = {}
    for k in k_values:
        times = [r['speedup'] for r in results_e1 if r['k'] == k]
        avg_speedup[k] = float(np.mean(times))

    # Amdahl's Law: S = 1 / ((1 - P) + P/k), P=0.95
    P = 0.95
    k_theory = np.linspace(1, max(k_values), 100)
    amdahl = 1.0 / ((1 - P) + P / k_theory)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(avg_speedup.keys()), list(avg_speedup.values()),
            'bo-', label='Measured Speedup', linewidth=2, markersize=8)
    ax.plot(k_theory, amdahl, 'r--', label="Amdahl's Law (P=0.95)", linewidth=1.5)
    ax.plot(k_theory, k_theory, 'g:', label='Ideal Linear', linewidth=1.5)
    ax.set_xlabel('Number of Workers (K)')
    ax.set_ylabel('Speedup')
    ax.set_title('E1: Strong Scaling — Measured vs Amdahl\'s Law')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  [PLOT] Saved speedup curve to {output_path}")


def plot_cpu_heatmap(cpu_samples: List[List[float]], output_path: str):
    """Plot 2: Per-core CPU utilization heatmap over time."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not cpu_samples:
        return

    data = np.array(cpu_samples).T  # shape: (n_cores, n_timepoints)
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(data, ax=ax, cmap='YlOrRd', vmin=0, vmax=100,
                cbar_kws={'label': 'CPU Utilization (%)'})
    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('CPU Core')
    ax.set_title('CPU Core Utilization Heatmap During Parallel Extraction')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  [PLOT] Saved CPU heatmap to {output_path}")


def plot_throughput_scaling(results_e3: list, output_path: str):
    """Plot 3: Throughput vs dataset size."""
    import matplotlib.pyplot as plt

    n_vals = sorted(set(r['n'] for r in results_e3))
    avg_tp = {}
    for n in n_vals:
        tp_vals = [r['throughput'] for r in results_e3 if r['n'] == n]
        avg_tp[n] = float(np.mean(tp_vals))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(avg_tp.keys()), list(avg_tp.values()), 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Dataset Size (N)')
    ax.set_ylabel('Throughput (domains/sec)')
    ax.set_title('E3: Dataset Scaling — Throughput vs N')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  [PLOT] Saved throughput scaling to {output_path}")


def plot_time_breakdown(breakdown_data: dict, output_path: str):
    """Plot 4: Stacked bar chart of phase timings per K."""
    import matplotlib.pyplot as plt

    k_labels = list(breakdown_data.keys())
    phases = list(breakdown_data[k_labels[0]].keys()) if k_labels else []
    bottom = np.zeros(len(k_labels))

    fig, ax = plt.subplots(figsize=(8, 5))
    for phase in phases:
        vals = [breakdown_data[k].get(phase, 0) for k in k_labels]
        ax.bar(k_labels, vals, bottom=bottom, label=phase)
        bottom += np.array(vals)
    ax.set_xlabel('K (Workers)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Phase Timing Breakdown by Worker Count')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  [PLOT] Saved time breakdown to {output_path}")


def plot_feature_ablation(results_e7: list, output_path: str):
    """Plot 5: Accuracy vs number of features."""
    import matplotlib.pyplot as plt

    n_feats = [r['n_features'] for r in results_e7]
    acc_vals = [r['accuracy'] for r in results_e7]
    f1_vals = [r['f1'] for r in results_e7]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_feats, acc_vals, 'bo-', label='Accuracy', linewidth=2, markersize=8)
    ax.plot(n_feats, f1_vals, 'rs-', label='F1', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Score')
    ax.set_title('E7: Feature Ablation Study')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  [PLOT] Saved feature ablation to {output_path}")


def plot_dt_vs_rf(results_e6: dict, output_path: str):
    """Plot 6: Grouped bar chart comparing DT and RF metrics."""
    import matplotlib.pyplot as plt

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    dt_vals = [results_e6['decision_tree'][m] for m in metrics]
    rf_vals = [results_e6['random_forest'][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, dt_vals, width, label='Decision Tree', color='steelblue')
    ax.bar(x + width / 2, rf_vals, width, label='Random Forest', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.1)
    ax.set_title('E6: Decision Tree vs Random Forest')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  [PLOT] Saved DT vs RF comparison to {output_path}")


# ── Suite Runner ──

def run_benchmark_suite(args) -> Dict[str, Any]:
    """Run all (or specified) experiments and save results.

    Args:
        args: Parsed command-line arguments from main.py.

    Returns:
        Dict of all experiment results.
    """
    import pandas as pd
    from src.shared_resources import initialize_shared_resources
    from src.classifier import (
        train_random_forest, evaluate_model,
        run_hyperparameter_sweep, run_dt_vs_rf_comparison, run_feature_ablation
    )
    from src.features import extract_all_sequential

    data_path = args.data_path
    output_dir = args.output_dir
    reps = args.repetitions
    experiment = args.experiment

    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    print("[BENCH] Loading preprocessed data...")
    train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))

    dictionary, ngram_table = initialize_shared_resources(data_path)

    domain_list = train_df['domain'].tolist()
    y_train_labels = train_df['label'].values
    y_test_labels = test_df['label'].values
    test_domains = test_df['domain'].tolist()

    all_results = {}

    # E1: Strong scaling
    if experiment in ('all', 'E1', 'E2'):
        print("\n[BENCH] Running E1: Strong scaling...")
        e1_results = run_experiment_e1(domain_list, dictionary, ngram_table,
                                       k_values=[1, 2, 4, 8], reps=reps)
        all_results['E1'] = e1_results
        plot_speedup_curve(e1_results, os.path.join(plots_dir, 'e1_speedup.png'))

    # E3: Dataset scaling
    if experiment in ('all', 'E3'):
        print("\n[BENCH] Running E3: Dataset scaling...")
        e3_results = run_experiment_e3(domain_list, dictionary, ngram_table,
                                       n_values=[10000, 50000, 100000, 500000],
                                       k=8, reps=reps)
        all_results['E3'] = e3_results
        plot_throughput_scaling(e3_results, os.path.join(plots_dir, 'e3_throughput.png'))

    # E5-E7 need a trained feature matrix; extract with parallel at K=8
    if experiment in ('all', 'E5', 'E6', 'E7'):
        print("\n[BENCH] Extracting features for E5/E6/E7 (sequential)...")
        from src.parallel_engine import parallel_extract_features
        import psutil
        k = psutil.cpu_count(logical=False) or 8
        X_train, _ = measure_wall_time(parallel_extract_features, domain_list, k, dictionary, ngram_table)
        X_test, _ = measure_wall_time(parallel_extract_features, test_domains, k, dictionary, ngram_table)

    if experiment in ('all', 'E5'):
        print("\n[BENCH] Running E5: RF hyperparameter sweep...")
        e5_results = run_hyperparameter_sweep(X_train, y_train_labels, X_test, y_test_labels)
        all_results['E5'] = e5_results

    if experiment in ('all', 'E6'):
        print("\n[BENCH] Running E6: DT vs RF comparison...")
        e6_results = run_dt_vs_rf_comparison(X_train, y_train_labels, X_test, y_test_labels)
        all_results['E6'] = e6_results
        plot_dt_vs_rf(e6_results, os.path.join(plots_dir, 'e6_dt_vs_rf.png'))

    if experiment in ('all', 'E7'):
        print("\n[BENCH] Running E7: Feature ablation...")
        e7_results = run_feature_ablation(X_train, y_train_labels, X_test, y_test_labels)
        all_results['E7'] = e7_results
        plot_feature_ablation(e7_results, os.path.join(plots_dir, 'e7_ablation.png'))

    if experiment in ('all', 'E8'):
        print("\n[BENCH] Running E8: Weak scaling...")
        e8_results = run_experiment_e8(domain_list, dictionary, ngram_table,
                                       k_values=[1, 2, 4, 8], reps=reps)
        all_results['E8'] = e8_results

    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[BENCH] Results saved to {metrics_path}")

    return all_results
