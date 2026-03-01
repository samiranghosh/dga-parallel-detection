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
        # TODO: Import and call src.preprocess.run_preprocessing(args.data_path)
        raise NotImplementedError("Implement in src/preprocess.py (Phase 2)")

    elif args.mode == "sequential":
        print("[MODE] Sequential — single-threaded baseline...")
        # TODO: Import and call sequential pipeline
        # 1. Load preprocessed data from args.data_path
        # 2. Extract features sequentially (src.features.extract_all_sequential)
        # 3. Train classifier (src.classifier.train_random_forest)
        # 4. Evaluate and print results
        raise NotImplementedError("Implement in Phase 3 + Phase 7")

    elif args.mode == "parallel":
        n_workers = detect_workers(args.workers)
        print(f"[MODE] Parallel — {n_workers} workers...")
        # TODO: Import and call parallel pipeline
        # 1. Load preprocessed data
        # 2. Initialize shared resources (src.shared_resources)
        # 3. Create overlapping chunks (src.chunker)
        # 4. Parallel feature extraction (src.parallel_engine)
        # 5. Train classifier with n_jobs=-1
        # 6. Evaluate and print results
        raise NotImplementedError("Implement in Phases 4-7")

    elif args.mode == "benchmark":
        n_workers = detect_workers(args.workers)
        print(f"[MODE] Benchmark — experiment: {args.experiment}, {args.repetitions} reps...")
        # TODO: Import and call benchmark suite
        # src.benchmark.run_benchmark_suite(args)
        raise NotImplementedError("Implement in Phase 8")

    print("\n[DONE] Pipeline complete.")


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    multiprocessing.freeze_support()
    main()
