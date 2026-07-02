"""B6 Step 2 in-container measurement - runs in BOTH images.

Deliberately imports NO pandas / sklearn / psutil so the serve-onnx image
stays minimal: domains come from data/*.csv via the csv module, RSS from
/proc/self/status. Kernel mode comes from the FEATURE_KERNEL env var
(fast | fast_marisa), which multiprocessing children inherit.

Parts (one JSON each; every JSON embeds the cgroup evidence):
  idle     staged serving RSS: interpreter -> ort session -> serving
           resources (DictKernel, set released - B6 Step 0a) -> first
           inference. VmRSS + VmHWM (peak) captured. The C5 idle-first rule.
  latency  CANONICAL single-request protocol (first 2000 test.csv domains in
           file order, 200 warmup, 1000 reps, perf_counter_ns) against the
           given ONNX model; total/feature/predict p50/p95/p99.
  batch    Pool.map k=effective-cgroup-cores over the full train+test corpus
           (999,927 domains), 3 reps, wall time incl. pool startup - the B5
           batch protocol under a profile.

Usage (inside a container):
  python scripts/rq3/measure_serve.py --part latency --model models/dt12.onnx \
      --out /out/latency_dt12_fast_A.json
"""
import os
import sys
import csv
import gc
import json
import time
import argparse

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cgroup_info import cgroup_summary, effective_cpus  # noqa: E402

DATA = os.path.join(REPO, "data")


def rss_mib():
    gc.collect()
    out = {}
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(("VmRSS", "VmHWM")):
                key, val = line.split(":")
                out[key] = round(int(val.split()[0]) / 1024, 1)  # kB -> MiB
    return out


def load_domains(split, limit=None):
    path = os.path.join(DATA, f"{split}.csv")
    domains = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            domains.append(str(row["domain"]))
            if limit and len(domains) >= limit:
                break
    return domains


def part_idle(model_path):
    stages = {}

    def snap(label):
        stages[label] = rss_mib()

    snap("interpreter")
    from src.onnx_model import OnnxPredictor
    model = OnnxPredictor(model_path)
    snap("model_onnx_session")

    from src import features
    from src.shared_resources import initialize_serving_resources
    t0 = time.perf_counter()
    dictionary, ngram = initialize_serving_resources(DATA)
    t_res = time.perf_counter() - t0
    snap("serving_resources_kernel_only")

    feats = features.extract_features("googleadservices", "googleadservices",
                                      dictionary, ngram, skip_levenshtein=True)
    model.predict(feats.reshape(1, -1))
    snap("first_inference")

    return {
        "stages_rss_mib": stages,
        "idle_rss_mib": stages["first_inference"]["VmRSS"],
        "peak_rss_mib": stages["first_inference"]["VmHWM"],
        "resource_load_sec": round(t_res, 3),
        "dict_form": type(dictionary).__name__,
    }


def part_latency(model_path):
    from src.latency_harness import (measure_single_request_latency,
                                     CANONICAL_SAMPLE, CANONICAL_WARMUP,
                                     CANONICAL_REPS)
    from src.onnx_model import OnnxPredictor
    from src.shared_resources import initialize_serving_resources

    dictionary, ngram = initialize_serving_resources(DATA)
    model = OnnxPredictor(model_path)
    domains = load_domains("test", limit=CANONICAL_SAMPLE)
    out = measure_single_request_latency(
        domains, dictionary, ngram, model,
        skip_levenshtein=True, reps=CANONICAL_REPS, warmup=CANONICAL_WARMUP,
        serving_n_jobs=1)
    out["config"]["protocol"] = "CANONICAL-B4"
    out["rss_after_mib"] = rss_mib()
    return out


def part_batch(model_path, reps=3):
    from src.shared_resources import initialize_shared_resources
    from src.parallel_engine import parallel_extract_features

    k = max(1, int(effective_cpus()))
    dictionary, ngram = initialize_shared_resources(DATA)
    domains = load_domains("train") + load_domains("test")

    walls, rates = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        parallel_extract_features(domains, k, dictionary, ngram,
                                  skip_levenshtein=True)
        dt = time.perf_counter() - t0
        walls.append(round(dt, 2))
        rates.append(round(len(domains) / dt))
    return {
        "protocol": f"Pool.map k={k} (effective cgroup cores), full "
                    f"train+test corpus ({len(domains)} domains), "
                    f"skip_levenshtein=True, {reps} reps, wall incl. pool "
                    "startup",
        "k": k,
        "wall_sec": walls,
        "domains_per_sec": rates,
        "mean_domains_per_sec": round(sum(rates) / len(rates)),
        "rss_after_mib": rss_mib(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["idle", "latency", "batch"])
    ap.add_argument("--model", default="models/dt12.onnx")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from src.features import get_kernel_mode
    model_path = (args.model if os.path.isabs(args.model)
                  else os.path.join(REPO, args.model))
    t0 = time.perf_counter()
    body = {"idle": part_idle, "latency": part_latency,
            "batch": part_batch}[args.part](model_path)
    result = {
        "part": args.part,
        "model": os.path.basename(model_path),
        "feature_kernel": get_kernel_mode(),
        "cgroup": cgroup_summary(),
        "wall_sec_total": round(time.perf_counter() - t0, 1),
        "result": body,
    }
    text = json.dumps(result, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
