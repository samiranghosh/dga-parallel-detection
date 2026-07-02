"""B6 Step 4: tl2cgen compiled-.so bench (Windows-blocked since B3).

Runs in the `full` image (Linux): compiles RF-pruned to a native shared
library via treelite -> tl2cgen (gcc), then benches FOUR runtimes for the
SAME model under the canonical single-request protocol:

    sklearn (n_jobs=1)  |  treelite GTIL  |  tl2cgen .so  |  ONNX (ort 1.25.1)

Parity is asserted (labels array_equal vs sklearn on the canonical 2000-row
matrix) before any timing is recorded. DATA POINT ONLY - the B4 runtime
adjudication (ONNX) is not re-opened here.
"""
import os
import sys
import json
import time
import pickle
import argparse

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(REPO, "data")
RF_PKL = os.path.join(REPO, "results", "rq2", "models", "rf_pruned.pkl")
RF_ONNX = os.path.join(REPO, "results", "rq3", "models", "rf_pruned.onnx")
SO_PATH = "/tmp/rf_pruned.so"


class GtilModel:
    def __init__(self, tl_model):
        import treelite
        self._gtil = treelite.gtil
        self._model = tl_model

    def predict(self, X):
        proba = np.asarray(self._gtil.predict(self._model, X.astype(np.float32)))
        return proba.reshape(len(X), -1).argmax(axis=1)


class CompiledModel:
    def __init__(self, so_path):
        import tl2cgen
        self._tl2cgen = tl2cgen
        self._pred = tl2cgen.Predictor(so_path)

    def predict(self, X):
        dmat = self._tl2cgen.DMatrix(X.astype(np.float32))
        proba = np.asarray(self._pred.predict(dmat))
        return proba.reshape(len(X), -1).argmax(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import treelite
    import tl2cgen
    from cgroup_info import cgroup_summary
    from src.onnx_model import OnnxPredictor
    from src.features import extract_all_sequential
    from src.shared_resources import initialize_serving_resources
    from src.latency_harness import (measure_single_request_latency,
                                     CANONICAL_SAMPLE, CANONICAL_WARMUP,
                                     CANONICAL_REPS)
    from measure_serve import load_domains

    with open(RF_PKL, "rb") as f:
        rf = pickle.load(f)
    rf.n_jobs = 1

    tl_model = treelite.sklearn.import_model(rf)
    t0 = time.perf_counter()
    tl2cgen.export_lib(tl_model, toolchain="gcc", libpath=SO_PATH,
                       params={"parallel_comp": 8})
    compile_sec = time.perf_counter() - t0

    dictionary, ngram = initialize_serving_resources(DATA)
    domains = load_domains("test", limit=CANONICAL_SAMPLE)

    runtimes = {
        "sklearn_njobs1": rf,
        "treelite_gtil": GtilModel(tl_model),
        "tl2cgen_so": CompiledModel(SO_PATH),
        "onnx_ort": OnnxPredictor(RF_ONNX),
    }

    # parity BEFORE timing
    X = extract_all_sequential(domains, dictionary, ngram, skip_levenshtein=True)
    ref = np.asarray(rf.predict(X)).ravel()
    parity = {}
    for name, model in runtimes.items():
        got = np.asarray(model.predict(X)).ravel().astype(ref.dtype)
        parity[name] = bool(np.array_equal(ref, got))
    if not all(parity.values()):
        raise SystemExit(f"parity FAILED, refusing to time: {parity}")

    arms = {}
    for name, model in runtimes.items():
        res = measure_single_request_latency(
            domains, dictionary, ngram, model, skip_levenshtein=True,
            reps=CANONICAL_REPS, warmup=CANONICAL_WARMUP,
            serving_n_jobs=None)
        arms[name] = res
        t = res["total"]
        print(f"{name:16s} total p50 {t['p50_us']:8.1f}  p95 {t['p95_us']:8.1f}"
              f"  p99 {t['p99_us']:8.1f}  (predict p50 "
              f"{res['predict']['p50_us']:.1f})")

    result = {
        "protocol": "CANONICAL-B4 (2000 test.csv domains, 200 warmup, "
                    "1000 reps), model = RF-pruned (B4 fallback), "
                    "FEATURE_KERNEL=" + os.environ.get("FEATURE_KERNEL", "fast"),
        "note": "data point only - B4 runtime adjudication (ONNX) not "
                "re-opened. tl2cgen was import-blocked on Windows (B3).",
        "cgroup": cgroup_summary(),
        "compile_sec": round(compile_sec, 2),
        "so_bytes": os.path.getsize(SO_PATH),
        "parity_labels_equal_vs_sklearn": parity,
        "arms": arms,
        "versions": {"treelite": treelite.__version__,
                     "tl2cgen": tl2cgen.__version__},
    }
    text = json.dumps(result, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text[:1500])


if __name__ == "__main__":
    main()
