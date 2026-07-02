"""B6 Step 2: ONE bounded RQ1 adaptive-vs-static datum under Profile A (2c).

Runs the rq1-adaptive branch's streaming engines - mounted read-only at
$RQ1_TREE (default /rq1), exported via `git archive rq1-adaptive` so RQ
branches stay unmixed - inside the full image under the Profile-A cgroup.
Uniform load only, rho in {0.5, 0.9} of the measured 2-worker saturation
throughput, reps=2, 30k-domain subset. This is a C3-under-cgroup datum,
NOT a re-run of the B2 sweep; the B2 finding (<=2-core: adaptive <=
static-2 <= sequential) is only being spot-checked under a real cgroup.

The rq1 tree is B2-era code (legacy feature kernel) by design - the datum
is an engine phenomenon, measured as that branch stands.
"""
import os
import sys
import json
import time
import argparse

RQ1_TREE = os.environ.get("RQ1_TREE", "/rq1")
sys.path.insert(0, RQ1_TREE)
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

RHOS = (0.5, 0.9)
REPS = 2
SUBSET = 30_000
MAX_WORKERS = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="data/")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from cgroup_info import cgroup_summary

    from src.benchmark_adaptive import (run_streaming_benchmark,
                                        run_sequential_streaming,
                                        measure_saturation_throughput)
    from src.load_profiles import uniform_load
    from src.shared_resources import initialize_shared_resources
    import pandas as pd

    dictionary, ngram = initialize_shared_resources(args.data_path)
    domains = (pd.read_csv(os.path.join(args.data_path, "test.csv"))["domain"]
               .astype(str).tolist()[:SUBSET])

    t0 = time.perf_counter()
    mu = measure_saturation_throughput(domains, dictionary, ngram,
                                       k=MAX_WORKERS)
    out = {
        "protocol": f"uniform load, rho in {RHOS}, reps={REPS}, "
                    f"subset={SUBSET}, max_workers={MAX_WORKERS} "
                    "(Profile A: 2 cores), rq1-adaptive tree as-is "
                    "(legacy kernel)",
        "cgroup": cgroup_summary(),
        "mu_domains_per_sec_k2": mu,
        "runs": {},
    }
    for rho in RHOS:
        rate = rho * mu
        for engine in ("adaptive", "static", "sequential"):
            stats = []
            for _ in range(REPS):
                gen = uniform_load(domains, rate=rate, batch_size=100)
                if engine == "adaptive":
                    r = run_streaming_benchmark(domains, dictionary, ngram,
                                                gen, min_workers=1,
                                                max_workers=MAX_WORKERS)
                elif engine == "static":
                    r = run_streaming_benchmark(domains, dictionary, ngram,
                                                gen, min_workers=MAX_WORKERS,
                                                max_workers=MAX_WORKERS)
                else:
                    r = run_sequential_streaming(domains, dictionary, ngram,
                                                 gen)
                stats.append({k: v for k, v in r.items()
                              if not isinstance(v, (list, dict))
                              or k == "latency_percentiles"})
            out["runs"][f"{engine}_rho{rho}"] = stats
            tp = [s.get("throughput_domains_per_sec") for s in stats]
            print(f"{engine:11s} rho={rho}: throughput {tp}")
    out["wall_sec_total"] = round(time.perf_counter() - t0, 1)

    text = json.dumps(out, indent=2, default=str)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text[:2000])


if __name__ == "__main__":
    main()
