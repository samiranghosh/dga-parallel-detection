"""
Single-Request Latency Harness  (Batch 1.3 — Shared Primitive #1)
=================================================================
Measures *single-request* feature-extraction + inference latency for one
domain at a time, the metric that matters for the RQ3 edge SLA (<1 ms).

Why this primitive exists:
- The dissertation must NOT conflate single-request latency with amortised
  batch throughput. Batch numbers hide per-request cost; the SLA is per request.
- Per CLAUDE.md/AGENTS.md the cost is dominated by the two O(m^2) dictionary
  features (#3 meaningful_word_ratio, #5 lms_percentage), NOT RF inference
  (sub-microsecond). This harness reports feature vs predict time separately so
  that claim is measured, not assumed.

Method (matches the plan):
- perf_counter_ns timing, >=1000 reps, warm-up iterations discarded.
- Reports p50 / p95 / p99 + mean +/- SD (all in microseconds).
- Provides a batch-latency contrast (amortised per-domain) for the same domains.

Serves acceptance IDs C1, C2 and RQ3. Reused as a before/after primitive by the
Aho-Corasick/DAWG kernel experiment (Batch 3).

Run standalone:
    python -m src.latency_harness --data-path data/ --model results/baseline_model_5feat.pkl
"""

import os
import time
import pickle
import logging
from typing import Dict, Any, List, Optional, Sequence

import numpy as np

from src.features import extract_features

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Warm-up count and repetition count are module-level so tests can shrink them.
DEFAULT_WARMUP = 100
DEFAULT_REPS = 1000


def _summarize_ns(samples_ns: Sequence[int]) -> Dict[str, float]:
    """Reduce a list of nanosecond timings to percentile + moment stats (microseconds)."""
    arr = np.asarray(samples_ns, dtype=np.float64) / 1000.0  # ns -> microseconds
    return {
        "n": int(arr.size),
        "mean_us": float(arr.mean()),
        "sd_us": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p50_us": float(np.percentile(arr, 50)),
        "p95_us": float(np.percentile(arr, 95)),
        "p99_us": float(np.percentile(arr, 99)),
        "min_us": float(arr.min()),
        "max_us": float(arr.max()),
    }


def measure_single_request_latency(domains: List[str],
                                   dictionary: Any,
                                   ngram_table: dict,
                                   model,
                                   skip_levenshtein: bool = True,
                                   reps: int = DEFAULT_REPS,
                                   warmup: int = DEFAULT_WARMUP) -> Dict[str, Any]:
    """Time per-domain (feature extraction + single predict), one request at a time.

    Args:
        domains: Pool of domain strings to cycle through (order fixed for determinism).
        dictionary, ngram_table: Loaded shared resources.
        model: Trained classifier with .predict on a (1, n_features) array.
        skip_levenshtein: True => 5-feature production config.
        reps: Number of timed single requests (>= 1000 recommended).
        warmup: Warm-up requests discarded before timing.

    Returns:
        Dict with 'total', 'feature', 'predict' timing summaries (microseconds)
        plus config metadata.
    """
    if not domains:
        raise ValueError("domains must be non-empty")

    n = len(domains)
    # Deterministic prev-domain pairing mirrors sequential extraction semantics.
    def _prev(idx: int) -> str:
        return domains[idx - 1] if idx > 0 else domains[idx]

    # --- Warm-up (JIT of numpy paths, dict/trigram cache warm, discarded) ---
    for k in range(warmup):
        i = k % n
        feats = extract_features(domains[i], _prev(i), dictionary, ngram_table,
                                 skip_levenshtein=skip_levenshtein)
        model.predict(feats.reshape(1, -1))

    total_ns: List[int] = []
    feat_ns: List[int] = []
    pred_ns: List[int] = []

    for k in range(reps):
        i = k % n
        d, p = domains[i], _prev(i)

        t0 = time.perf_counter_ns()
        feats = extract_features(d, p, dictionary, ngram_table,
                                 skip_levenshtein=skip_levenshtein)
        t1 = time.perf_counter_ns()
        model.predict(feats.reshape(1, -1))
        t2 = time.perf_counter_ns()

        feat_ns.append(t1 - t0)
        pred_ns.append(t2 - t1)
        total_ns.append(t2 - t0)

    return {
        "config": {
            "reps": reps,
            "warmup": warmup,
            "n_features": 5 if skip_levenshtein else 6,
            "domain_pool_size": n,
        },
        "total": _summarize_ns(total_ns),
        "feature": _summarize_ns(feat_ns),
        "predict": _summarize_ns(pred_ns),
    }


def measure_batch_latency(domains: List[str],
                          dictionary: Any,
                          ngram_table: dict,
                          model,
                          skip_levenshtein: bool = True,
                          reps: int = 10) -> Dict[str, Any]:
    """Amortised batch contrast: extract+predict the whole list, report per-domain.

    This is deliberately NOT the single-request number — it exists so reports can
    show the gap between amortised batch cost and true per-request latency.
    """
    from src.features import extract_all_sequential

    n = len(domains)
    per_domain_us: List[float] = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        X = extract_all_sequential(domains, dictionary, ngram_table,
                                   skip_levenshtein=skip_levenshtein)
        model.predict(X)
        elapsed_ns = time.perf_counter_ns() - t0
        per_domain_us.append((elapsed_ns / n) / 1000.0)

    arr = np.asarray(per_domain_us)
    return {
        "config": {"reps": reps, "batch_size": n,
                   "n_features": 5 if skip_levenshtein else 6},
        "amortised_per_domain_us": {
            "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min": float(arr.min()),
        },
    }


def _load_model(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def run(data_path: str = "data/",
        model_path: Optional[str] = None,
        sample_size: int = 2000,
        reps: int = DEFAULT_REPS,
        skip_levenshtein: bool = True) -> Dict[str, Any]:
    """Load resources + model, sample domains, run single-request + batch measures.

    The model is loaded from a pickle if provided; otherwise a 5-feature RF is
    trained on the fly from data/train.csv (heavier, but self-contained).
    """
    import pandas as pd
    from src.shared_resources import initialize_shared_resources

    dictionary, ngram_table = initialize_shared_resources(data_path)

    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
    domains = test_df["domain"].astype(str).tolist()[:sample_size]

    if model_path and os.path.exists(model_path):
        logger.info(f"Loading model from {model_path}")
        model = _load_model(model_path)
    else:
        logger.info("No model pickle given — training a 5-feature RF from train.csv")
        from src.features import extract_all_sequential
        from src.classifier import train_random_forest
        train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
        Xtr = extract_all_sequential(train_df["domain"].astype(str).tolist(),
                                     dictionary, ngram_table, skip_levenshtein=True)
        model = train_random_forest(Xtr, train_df["label"].values, n_estimators=50)

    single = measure_single_request_latency(
        domains, dictionary, ngram_table, model,
        skip_levenshtein=skip_levenshtein, reps=reps)
    batch = measure_batch_latency(
        domains, dictionary, ngram_table, model, skip_levenshtein=skip_levenshtein)

    result = {"single_request": single, "batch_contrast": batch}
    logger.info("Single-request total p50=%.1f us  p95=%.1f us  p99=%.1f us",
                single["total"]["p50_us"], single["total"]["p95_us"],
                single["total"]["p99_us"])
    logger.info("Feature share of p50: %.1f%%",
                100.0 * single["feature"]["p50_us"] / max(single["total"]["p50_us"], 1e-9))
    return result


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Single-request latency harness (Batch 1.3)")
    ap.add_argument("--data-path", default="data/")
    ap.add_argument("--model", default=None, help="Path to a model pickle (optional)")
    ap.add_argument("--reps", type=int, default=DEFAULT_REPS)
    ap.add_argument("--sample-size", type=int, default=2000)
    ap.add_argument("--output", default=None, help="Optional JSON output path")
    args = ap.parse_args()

    res = run(args.data_path, args.model, args.sample_size, args.reps)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        logger.info("Wrote %s", args.output)
    else:
        print(json.dumps(res, indent=2))
