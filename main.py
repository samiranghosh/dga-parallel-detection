"""
DGA Parallel Detection — Main Entry Point
==========================================
AMLCCZG516 — ML System Optimization — Group 09

Usage:
    python main.py --mode preprocess              # Prepare datasets
    python main.py --mode sequential              # Baseline single-threaded run
    python main.py --mode parallel --workers 8    # Parallel run with 8 workers
    python main.py --mode benchmark               # Full experiment suite (E1-E8)

Platform: HP OMEN 16 / AMD Ryzen 7 7840HS (8C/16T) / 32 GB DDR5 / Windows 11
"""

import argparse
import os
import sys
import time
import multiprocessing

def get_args():
    parser = argparse.ArgumentParser(
        description="Parallel DGA Malware Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode preprocess
  python main.py --mode sequential
  python main.py --mode parallel --workers 8
  python main.py --mode benchmark --experiment E1 --repetitions 5
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["preprocess", "sequential", "parallel", "benchmark"],
        default="parallel",
        help="Execution mode (default: parallel)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes. 0 = auto-detect physical cores (default: 0)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/",
        help="Path to the data directory (default: data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/",
        help="Directory for benchmark results (default: results/)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        help="Specific experiment to run: E1-E8 or 'all' (default: all)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of Random Forest trees (default: 100)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Number of repetitions per benchmark config (default: 5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging",
    )
    return parser.parse_args()


def detect_workers(requested: int) -> int:
    """Determine the number of worker processes.

    If requested is 0, auto-detect the number of physical CPU cores.
    On the Ryzen 7 7840HS, this returns 8 (physical) vs 16 (logical).
    We default to physical cores for CPU-bound workloads.
    """
    if requested > 0:
        return requested
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
        print(f"[INFO] Detected {physical} physical cores, {logical} logical threads")
        return physical or logical or 4
    except ImportError:
        count = multiprocessing.cpu_count()
        # Assume SMT: physical ≈ logical / 2
        return max(count // 2, 1)


def print_system_info():
    """Print system information for reproducibility."""
    import platform
    try:
        import psutil
        mem = psutil.virtual_memory()
        mem_str = f"{mem.total / (1024**3):.1f} GB total, {mem.available / (1024**3):.1f} GB available"
    except ImportError:
        mem_str = "psutil not installed"

    print("=" * 70)
    print("DGA Parallel Detection — System Information")
    print("=" * 70)
    print(f"  Python:    {platform.python_version()} ({platform.python_implementation()})")
    print(f"  OS:        {platform.system()} {platform.release()}")
    print(f"  Machine:   {platform.machine()}")
    print(f"  CPU cores: {multiprocessing.cpu_count()} logical")
    print(f"  Memory:    {mem_str}")
    print("=" * 70)


def main():
    args = get_args()

    if args.verbose:
        print_system_info()

    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)

    if args.mode == "preprocess":
        print("[MODE] Preprocessing — preparing datasets...")
        import src.preprocess
        src.preprocess.run_preprocessing(args.data_path)

    elif args.mode == "sequential":
        print("[MODE] Sequential — single-threaded baseline...")
        import pandas as pd
        from src.shared_resources import initialize_shared_resources
        from src.features import extract_all_sequential
        from src.classifier import train_random_forest, evaluate_model

        train_df = pd.read_csv(os.path.join(args.data_path, 'train.csv'))
        test_df = pd.read_csv(os.path.join(args.data_path, 'test.csv'))
        dictionary, ngram_table = initialize_shared_resources(args.data_path)

        print(f"  Extracting features for {len(train_df)} train domains (sequential)...")
        t0 = time.time()
        X_train = extract_all_sequential(train_df['domain'].tolist(), dictionary, ngram_table)
        t_feat = time.time() - t0
        print(f"  Feature extraction: {t_feat:.2f}s")

        t0 = time.time()
        X_test = extract_all_sequential(test_df['domain'].tolist(), dictionary, ngram_table)
        print(f"  Test feature extraction: {time.time() - t0:.2f}s")

        print(f"  Training Random Forest ({args.n_estimators} trees)...")
        t0 = time.time()
        model = train_random_forest(X_train, train_df['label'].values,
                                    n_estimators=args.n_estimators)
        t_train = time.time() - t0
        print(f"  Training: {t_train:.2f}s")

        metrics = evaluate_model(model, X_test, test_df['label'].values)
        print(f"\n  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  Inference: {metrics['inference_latency_ms']:.1f} ms")
        if args.verbose:
            print(metrics['classification_report'])

    elif args.mode == "parallel":
        n_workers = detect_workers(args.workers)
        print(f"[MODE] Parallel — {n_workers} workers...")
        import pandas as pd
        from src.shared_resources import initialize_shared_resources
        from src.parallel_engine import parallel_extract_features
        from src.classifier import train_random_forest, evaluate_model

        train_df = pd.read_csv(os.path.join(args.data_path, 'train.csv'))
        test_df = pd.read_csv(os.path.join(args.data_path, 'test.csv'))
        dictionary, ngram_table = initialize_shared_resources(args.data_path)

        print(f"  Extracting features for {len(train_df)} train domains ({n_workers} workers)...")
        t0 = time.time()
        X_train = parallel_extract_features(train_df['domain'].tolist(), n_workers, dictionary, ngram_table)
        t_feat = time.time() - t0
        print(f"  Feature extraction: {t_feat:.2f}s")

        t0 = time.time()
        X_test = parallel_extract_features(test_df['domain'].tolist(), n_workers, dictionary, ngram_table)
        print(f"  Test feature extraction: {time.time() - t0:.2f}s")

        print(f"  Training Random Forest ({args.n_estimators} trees, n_jobs=-1)...")
        t0 = time.time()
        model = train_random_forest(X_train, train_df['label'].values,
                                    n_estimators=args.n_estimators)
        t_train = time.time() - t0
        print(f"  Training: {t_train:.2f}s")

        metrics = evaluate_model(model, X_test, test_df['label'].values)
        print(f"\n  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  Inference: {metrics['inference_latency_ms']:.1f} ms")
        if args.verbose:
            print(metrics['classification_report'])

    elif args.mode == "benchmark":
        n_workers = detect_workers(args.workers)
        print(f"[MODE] Benchmark — experiment: {args.experiment}, {args.repetitions} reps...")
        from src.benchmark import run_benchmark_suite
        run_benchmark_suite(args)

    print("\n[DONE] Pipeline complete.")



if __name__ == "__main__":
    # Required for multiprocessing on Windows
    multiprocessing.freeze_support()
    main()
