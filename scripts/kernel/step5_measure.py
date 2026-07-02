"""B5 Step 5: before/after measurement -> results/kernel/step5_*.json.

Three parts (run each on an idle box, sequentially - timing purity):
  --part single   canonical single-request protocol (latency_harness), DT-12
                  sklearn + DT-12 ONNX, FEATURE_KERNEL legacy vs fast
  --part batch    Pool.map k=8 over the full train+test corpus (999,927
                  domains), legacy vs fast, 3 reps each
  --part memory   staged serving-stack RSS + load times (fresh subprocess per
                  stack), sklearn-path and ONNX-path, legacy vs fast
  --part stack    (internal child for --part memory)

DT-12 = the B4-gated primary (results/rq2/models/dt12.pkl); its ONNX file is
rebuilt deterministically from the pickle into the scratch artifacts dir.
"""
import os
import sys
import json
import time
import pickle
import argparse
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

DATA = os.path.join(REPO, "data")
DT12_PKL = os.path.join(REPO, "results", "rq2", "models", "dt12.pkl")
ART = os.environ.get(
    "KERNEL_ART",
    os.path.join(os.environ.get("TEMP", "/tmp"), "b5_artifacts"))
DT12_ONNX = os.path.join(ART, "dt12.onnx")
AUTOMATON_BLOB = os.path.join(ART, "dict_automaton.pkl")
OUTDIR = os.path.join(REPO, "results", "kernel")
BATCH_REPS = 3
BATCH_K = 8


def ensure_artifacts():
    os.makedirs(ART, exist_ok=True)
    if not os.path.exists(DT12_ONNX):
        from src.onnx_model import convert_to_onnx
        with open(DT12_PKL, "rb") as f:
            dt12 = pickle.load(f)
        convert_to_onnx(dt12, 5, DT12_ONNX)
    if not os.path.exists(AUTOMATON_BLOB):
        from src import features
        from src.shared_resources import initialize_shared_resources
        dictionary, _ = initialize_shared_resources(DATA)
        with open(AUTOMATON_BLOB, "wb") as f:
            f.write(features.export_automaton_blob(dictionary))


def part_single():
    from src import features
    from src.shared_resources import initialize_shared_resources
    from src.latency_harness import measure_canonical
    from src.onnx_model import OnnxPredictor

    ensure_artifacts()
    dictionary, ngram_table = initialize_shared_resources(DATA)
    with open(DT12_PKL, "rb") as f:
        dt12 = pickle.load(f)
    onnx_pred = OnnxPredictor(DT12_ONNX)

    out = {"protocol": "CANONICAL-B4 (2000 test.csv domains, 200 warmup, "
                       "1000 reps, n_jobs=1, idle box)",
           "model": "DT-12 (B4 primary)", "arms": {}}
    for mode in ("legacy", "fast"):
        features.set_kernel_mode(mode)
        if mode == "fast":
            features.warm_kernel(dictionary)  # build outside the timed region
        for runtime, model in (("sklearn", dt12), ("onnx", onnx_pred)):
            res = measure_canonical(model, dictionary, ngram_table,
                                    data_path=DATA)
            out["arms"][f"dt12_{runtime}_{mode}"] = res
            t = res["total"]; ft = res["feature"]; pr = res["predict"]
            print(f"dt12 {runtime:7s} {mode:6s}: total p50 {t['p50_us']:7.1f}"
                  f" (feat {ft['p50_us']:6.1f} / predict {pr['p50_us']:6.1f})"
                  f"  p95 {t['p95_us']:7.1f}  p99 {t['p99_us']:7.1f}")
    _write("step5_single_request.json", out)


def part_batch():
    import pandas as pd
    from src import features
    from src.shared_resources import initialize_shared_resources
    from src.parallel_engine import parallel_extract_features

    dictionary, ngram_table = initialize_shared_resources(DATA)
    frames = [pd.read_csv(os.path.join(DATA, f"{s}.csv"))
              for s in ("train", "test")]
    domains = pd.concat(frames, ignore_index=True)["domain"].astype(str).tolist()

    # Workers are SPAWNED: they re-import features and read FEATURE_KERNEL
    # from the environment - set_kernel_mode() in the parent does not reach
    # them. Arms are therefore driven by os.environ (inherited by children).
    # Third arm isolates the Step-4 blob shipping (initargs) against lazy
    # per-worker build, since worker builds run in parallel during spawn.
    arms = [
        ("legacy", "legacy", True),
        ("fast_blob_attach", "fast", True),
        ("fast_worker_build", "fast", False),
    ]
    out = {"protocol": f"Pool.map k={BATCH_K}, full train+test corpus "
                       f"({len(domains)} domains), skip_levenshtein=True, "
                       f"{BATCH_REPS} reps, wall time incl. pool startup "
                       "(Windows spawn); kernel mode propagated to workers "
                       "via FEATURE_KERNEL env", "arms": {}}
    real_export = features.export_automaton_blob
    ref = None
    try:
        for name, mode, ship_blob in arms:
            os.environ["FEATURE_KERNEL"] = mode
            features.set_kernel_mode(mode)
            features.export_automaton_blob = (
                real_export if ship_blob else lambda d: None)
            times, rates = [], []
            for _ in range(BATCH_REPS):
                t0 = time.perf_counter()
                X = parallel_extract_features(domains, BATCH_K, dictionary,
                                              ngram_table,
                                              skip_levenshtein=True)
                dt = time.perf_counter() - t0
                times.append(dt)
                rates.append(len(domains) / dt)
            if ref is None:
                ref = X
            else:
                import numpy as np
                assert np.array_equal(ref, X), \
                    "kernel outputs diverged in batch path"
            out["arms"][name] = {
                "wall_sec": [round(t, 2) for t in times],
                "domains_per_sec": [round(r) for r in rates],
                "mean_domains_per_sec": round(sum(rates) / len(rates)),
            }
            print(f"batch {name:18s}: "
                  f"{out['arms'][name]['mean_domains_per_sec']:,} dom/s "
                  f"(walls {out['arms'][name]['wall_sec']})")
    finally:
        features.export_automaton_blob = real_export
        os.environ.pop("FEATURE_KERNEL", None)
    out["equal_outputs_across_kernels"] = True
    _write("step5_batch.json", out)


STACKS = {
    # runtime, kernel  (dict set is always loaded: the feature API keys on it)
    "sklearn_legacy": ("sklearn", False),
    "sklearn_fast": ("sklearn", True),
    "onnx_legacy": ("onnx", False),
    "onnx_fast": ("onnx", True),
}


def part_stack(name):
    """Child: build one serving stack, print staged RSS (MiB) as JSON."""
    import psutil
    runtime, fast = STACKS[name]
    p = psutil.Process()
    stages = {}
    t_all = time.perf_counter()

    def snap(label):
        stages[label] = round(p.memory_info().rss / 2**20, 1)

    snap("interpreter")
    if runtime == "sklearn":
        with open(DT12_PKL, "rb") as f:
            model = pickle.load(f)  # noqa: F841  (imports numpy+sklearn)
        snap("model_sklearn")
    else:
        sys.path.insert(0, REPO)
        from src.onnx_model import OnnxPredictor
        model = OnnxPredictor(DT12_ONNX)  # noqa: F841  (ort only, no sklearn)
        snap("model_onnx_session")

    t0 = time.perf_counter()
    dictionary = set(l.strip() for l in
                     open(os.path.join(DATA, "english_dictionary.txt"),
                          encoding="utf-8") if l.strip())
    t_dict = time.perf_counter() - t0
    snap("dict_set")

    t_auto = None
    if fast:
        t0 = time.perf_counter()
        with open(AUTOMATON_BLOB, "rb") as f:
            automaton = pickle.loads(f.read())  # noqa: F841
        t_auto = time.perf_counter() - t0
        snap("automaton_attach")

    with open(os.path.join(DATA, "ngram_table.pkl"), "rb") as f:
        ngram = pickle.load(f)  # noqa: F841
    snap("trigram")

    print(json.dumps({"stages_rss_mib": stages,
                      "total_rss_mib": stages["trigram"],
                      "load_sec": {"dict_set": round(t_dict, 3),
                                   "automaton_attach": (round(t_auto, 3)
                                                        if t_auto else None),
                                   "whole_stack": round(time.perf_counter()
                                                        - t_all, 3)}}))


def part_memory():
    ensure_artifacts()
    out = {"note": "fresh subprocess per stack; automaton attached from a "
                   "pre-serialized blob (the Step-4 path). Dict set stays "
                   "loaded in fast stacks: the feature API keys the automaton "
                   "cache on the dictionary object.",
           "stacks": {}}
    for name in STACKS:
        r = subprocess.run([sys.executable, __file__, "--part", "stack",
                            "--stack", name],
                           capture_output=True, text=True, cwd=REPO, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"{name}: {r.stderr[-1500:]}")
        out["stacks"][name] = json.loads(r.stdout.strip().splitlines()[-1])
        print(name, "->", out["stacks"][name]["total_rss_mib"], "MiB")
    _write("step5_memory.json", out)


def _write(fname, obj):
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["single", "batch", "memory", "stack"])
    ap.add_argument("--stack", default=None)
    args = ap.parse_args()
    if args.part == "single":
        part_single()
    elif args.part == "batch":
        part_batch()
    elif args.part == "memory":
        part_memory()
    else:
        part_stack(args.stack)
