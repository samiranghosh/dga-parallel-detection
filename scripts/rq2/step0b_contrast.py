"""B4 Step 0b: RF-100 n_jobs=-1 vs n_jobs=1 under the CANONICAL protocol.
(The full per-model canonical set is produced by step56_arms.py; this adds the
joblib-fan-out contrast that motivated locking n_jobs=1.)
Writes results/rq2/step0b_canonical_contrast.json.
"""
import sys, os, json, pickle
sys.path.insert(0, r"c:/Users/samir/source/repos/dga-parallel-detection")

ART = r"C:/Users/samir/AppData/Local/Temp/claude/c--Users-samir-source-repos-dga-parallel-detection/932315aa-8a56-47d8-95fa-59c19e89893b/scratchpad/artifacts"
REPO = r"c:/Users/samir/source/repos/dga-parallel-detection"

from src.shared_resources import initialize_shared_resources
from src.latency_harness import (measure_single_request_latency,
                                 CANONICAL_SAMPLE, CANONICAL_WARMUP, CANONICAL_REPS)
import pandas as pd

if __name__ == "__main__":
    dictionary, ngram = initialize_shared_resources(os.path.join(REPO, "data/"))
    domains = (pd.read_csv(os.path.join(REPO, "data/test.csv"))["domain"]
               .astype(str).tolist()[:CANONICAL_SAMPLE])
    with open(os.path.join(ART, "rf100_5feat.pkl"), "rb") as f:
        model = pickle.load(f)

    out = {}
    model.n_jobs = -1
    out["rf100_njobs_-1"] = measure_single_request_latency(
        domains, dictionary, ngram, model, reps=CANONICAL_REPS,
        warmup=CANONICAL_WARMUP, serving_n_jobs=None)
    out["rf100_njobs_1"] = measure_single_request_latency(
        domains, dictionary, ngram, model, reps=CANONICAL_REPS,
        warmup=CANONICAL_WARMUP, serving_n_jobs=1)
    for k, v in out.items():
        t = v["total"]
        print(f"[0b] {k}: total p50/p95/p99 = {t['p50_us']/1000:.2f} / "
              f"{t['p95_us']/1000:.2f} / {t['p99_us']/1000:.2f} ms", flush=True)
    with open(os.path.join(REPO, "results/rq2/step0b_canonical_contrast.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("[0b] DONE", flush=True)
