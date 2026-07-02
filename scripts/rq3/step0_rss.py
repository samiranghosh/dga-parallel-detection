"""B6 Step 0: serving-stack RSS with the dictionary set released.

Contrast (fresh subprocess per stack, DT-12 ONNX serving path):
  onnx_fast_set_retained   - B5 status quo: set stays resident because the
                             matcher cache keys on it (118.9 MiB in
                             results/kernel/step5_memory.json)
  onnx_fast_kernel_only    - Step 0a: initialize_serving_resources builds the
                             AC automaton, releases the set
  onnx_marisa_kernel_only  - Step 0b: marisa trie mmapped from
                             data/english_dictionary.marisa (0.7 MiB)

Each child prints staged RSS as JSON; the parent aggregates into
results/rq3/step0_rss.json. A first inference runs before the final
snapshot so the working set is materialised, and values are read after
gc.collect() so a released set can actually leave the number.
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
OUT = os.path.join(REPO, "results", "rq3", "step0_rss.json")

STACKS = ("onnx_fast_set_retained", "onnx_fast_kernel_only",
          "onnx_marisa_kernel_only")


def ensure_onnx():
    os.makedirs(ART, exist_ok=True)
    if not os.path.exists(DT12_ONNX):
        from src.onnx_model import convert_to_onnx
        with open(DT12_PKL, "rb") as f:
            dt12 = pickle.load(f)
        convert_to_onnx(dt12, 5, DT12_ONNX)


def part_stack(name):
    import gc
    import psutil
    p = psutil.Process()
    stages = {}
    t_all = time.perf_counter()

    def snap(label):
        gc.collect()
        stages[label] = round(p.memory_info().rss / 2**20, 1)

    snap("interpreter")
    from src.onnx_model import OnnxPredictor
    model = OnnxPredictor(DT12_ONNX)
    snap("model_onnx_session")

    from src import features
    from src.shared_resources import (initialize_shared_resources,
                                      initialize_serving_resources)
    t0 = time.perf_counter()
    if name == "onnx_fast_set_retained":
        features.set_kernel_mode("fast")
        dictionary, ngram = initialize_shared_resources(DATA)
        features.warm_kernel(dictionary)  # cache keyed on the live set
    elif name == "onnx_fast_kernel_only":
        features.set_kernel_mode("fast")
        dictionary, ngram = initialize_serving_resources(DATA)
    else:
        features.set_kernel_mode("fast_marisa")
        dictionary, ngram = initialize_serving_resources(DATA)
    t_res = time.perf_counter() - t0
    snap("dict_resources_plus_trigram")

    feats = features.extract_features("googleadservices", "googleadservices",
                                      dictionary, ngram, skip_levenshtein=True)
    model.predict(feats.reshape(1, -1))
    snap("first_inference")

    print(json.dumps({
        "stages_rss_mib": stages,
        "total_rss_mib": stages["first_inference"],
        "load_sec": {"dict_resources": round(t_res, 3),
                     "whole_stack": round(time.perf_counter() - t_all, 3)},
    }))


def main():
    ensure_onnx()
    out = {"note": "fresh subprocess per stack; DT-12 ONNX; RSS after "
                   "gc.collect(); first inference included before the final "
                   "snapshot. Contrast vs results/kernel/step5_memory.json "
                   "onnx_fast (set retained).",
           "platform": sys.platform,
           "stacks": {}}
    for name in STACKS:
        r = subprocess.run([sys.executable, __file__, "--part", "stack",
                            "--stack", name],
                           capture_output=True, text=True, cwd=REPO,
                           timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"{name}: {r.stderr[-1500:]}")
        out["stacks"][name] = json.loads(r.stdout.strip().splitlines()[-1])
        print(name, "->", out["stacks"][name]["total_rss_mib"], "MiB")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="main", choices=["main", "stack"])
    ap.add_argument("--stack", default=None, choices=STACKS)
    args = ap.parse_args()
    if args.part == "stack":
        part_stack(args.stack)
    else:
        main()
