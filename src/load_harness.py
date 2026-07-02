"""
RQ1 Load Harness  (Batch 2 / Step 5)
====================================
Orchestrates the RQ1 comparative load sweep across THREE engines and THREE
traffic profiles:

  engines  : adaptive (1..K, proportional) · static K · sequential (1 process)
  profiles : uniform (constant) · bursty (Poisson) · ramp (linearly rising)
  sweep    : offered load rho ∈ {0.1, 0.3, 0.5, 0.7, 0.9, 1.05} of the
             K-worker saturation throughput μ (uniform + bursty; ramp traverses
             0.2μ→1.0μ within a single run).

For every run it records throughput (dom/s), latency percentiles, and the
worker-count + CPU time-series. The low-level primitives live in
`load_profiles.py` (generators) and `benchmark_adaptive.py` (engine drivers);
this module is the orchestration + CLI entry consumed by Step 6 (cost–benefit
curve, stability, 2×-peak, ≤2-core analysis).

Run:
    python -m src.load_harness --data-path data/ --reps 5 --subset 50000 \
        --output results/rq1_load_sweep.json
"""

import os
import json
import logging

import numpy as np

from src.benchmark_adaptive import (
    run_streaming_benchmark,
    run_sequential_streaming,
    measure_saturation_throughput,
)
from src.load_profiles import uniform_load, poisson_bursty_load, ramp_load

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_RHOS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.05]
ENGINES = ("adaptive", "static", "sequential")


def _run_engine(engine, test_domains, dictionary, ngram_table, gen, max_workers):
    """Dispatch a single streaming run to the requested engine."""
    if engine == "adaptive":
        return run_streaming_benchmark(test_domains, dictionary, ngram_table, gen,
                                       min_workers=1, max_workers=max_workers)
    if engine == "static":
        return run_streaming_benchmark(test_domains, dictionary, ngram_table, gen,
                                       min_workers=max_workers, max_workers=max_workers)
    if engine == "sequential":
        return run_sequential_streaming(test_domains, dictionary, ngram_table, gen)
    raise ValueError(f"unknown engine: {engine}")


def _make_gen(profile, test_domains, rate):
    if profile == "uniform":
        return uniform_load(test_domains, rate=rate, batch_size=100)
    if profile == "bursty":
        return poisson_bursty_load(test_domains, mean_rate=rate, burst_factor=3.0, batch_size=100)
    raise ValueError(f"{profile} is not a rho-swept profile")


def run_load_sweep(domain_list, dictionary, ngram_table, reps=3, subset=50000,
                   rhos=None, include_sequential=True, max_workers=8):
    """Full RQ1 sweep. Returns a nested dict:

        result["profiles"][profile][engine] -> list over rho of [reps stats]
        (ramp has a single rho slot: one traversal per rep)

    Each `stats` is the dict returned by the engine driver (throughput, latency
    percentiles, worker/cpu time-series).
    """
    rhos = list(rhos) if rhos is not None else list(DEFAULT_RHOS)
    engines = list(ENGINES) if include_sequential else ["adaptive", "static"]

    mu = measure_saturation_throughput(domain_list, dictionary, ngram_table, k=max_workers)
    test_domains = domain_list[:subset]

    result = {
        "mu": mu, "rhos": rhos, "subset": len(test_domains),
        "reps": reps, "max_workers": max_workers, "engines": engines,
        "profiles": {},
    }

    # Uniform + bursty: swept over rho.
    for profile in ("uniform", "bursty"):
        result["profiles"][profile] = {e: [] for e in engines}
        for rho in rhos:
            rate = rho * mu
            logger.info("[SWEEP] %s rho=%.2f rate=%.1f dom/s", profile, rho, rate)
            for e in engines:
                reps_stats = [
                    _run_engine(e, test_domains, dictionary, ngram_table,
                                _make_gen(profile, test_domains, rate), max_workers)
                    for _ in range(reps)
                ]
                result["profiles"][profile][e].append(reps_stats)

    # Ramp: a single traversal 0.2μ -> 1.0μ per engine (rho rises within the run).
    logger.info("[SWEEP] ramp 0.2*mu -> 1.0*mu")
    result["profiles"]["ramp"] = {e: [] for e in engines}
    for e in engines:
        reps_stats = [
            _run_engine(e, test_domains, dictionary, ngram_table,
                        ramp_load(test_domains, start_rate=0.2 * mu, end_rate=1.0 * mu, batch_size=100),
                        max_workers)
            for _ in range(reps)
        ]
        result["profiles"]["ramp"][e].append(reps_stats)

    return result


def _mean_throughput(reps_stats):
    return float(np.mean([s["throughput_domains_per_sec"] for s in reps_stats])) if reps_stats else 0.0


def summarize(result):
    """Compact console table of mean throughput per (profile, engine, rho)."""
    print(f"\nmu (K={result['max_workers']} saturation) = {result['mu']:.1f} dom/s | "
          f"subset={result['subset']} | reps={result['reps']}")
    for profile, per_engine in result["profiles"].items():
        print(f"\n[{profile}] mean throughput (dom/s)")
        rhos = result["rhos"] if profile != "ramp" else ["0.2->1.0"]
        header = "  engine".ljust(14) + "".join(f"rho={r}".rjust(12) for r in rhos)
        print(header)
        for e in result["engines"]:
            row = per_engine.get(e, [])
            cells = "".join(f"{_mean_throughput(rs):12.0f}" for rs in row)
            print(f"  {e}".ljust(14) + cells)


def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


if __name__ == "__main__":
    import argparse
    import pandas as pd
    from src.shared_resources import initialize_shared_resources

    ap = argparse.ArgumentParser(description="RQ1 load sweep harness (Batch 2 / Step 5)")
    ap.add_argument("--data-path", default="data/")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--subset", type=int, default=50000)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--no-sequential", action="store_true",
                    help="skip the sequential arm (adaptive vs static only)")
    ap.add_argument("--output", default="results/rq1_load_sweep.json")
    args = ap.parse_args()

    dictionary, ngram_table = initialize_shared_resources(args.data_path)
    df = pd.read_csv(os.path.join(args.data_path, "test.csv"))
    domains = df["domain"].astype(str).tolist()

    result = run_load_sweep(
        domains, dictionary, ngram_table, reps=args.reps, subset=args.subset,
        include_sequential=not args.no_sequential, max_workers=args.max_workers,
    )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=_json_default)
        logger.info("Wrote %s", args.output)

    summarize(result)
