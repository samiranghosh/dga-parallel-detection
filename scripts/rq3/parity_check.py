"""B6 Step 5/6: cross-arch functional parity (T6/D) - CORRECTNESS ONLY.

Runs inside the serve-onnx image (numpy + ort + kernels; no pandas/sklearn):

  1. A2 golden-v2: all three FEATURE_KERNEL modes vs the committed oracle
     (tests/golden_features_v2.json), exact ==. This is the feature-parity
     gate on the new architecture.
  2. Canonical 2000-row feature matrix (test.csv file order): sha256 per
     mode + per-feature column means (localises any cross-arch drift).
  3. Model parity: dt12.onnx + rf_pruned.onnx labels + probabilities on
     that matrix, vs the x86 reference. Criteria = B3's hard requirement:
     labels array_equal, probas allclose rtol=1e-3 (atol=1e-6).

  --generate  write the x86 reference JSON (run on x86 first, commit it)
  --check     compare this machine against the reference (ARM / QEMU)

NO TIMING NUMBERS are recorded here by design: QEMU is functional-only
(Rejected: QEMU for latency/throughput).
"""
import os
import sys
import json
import hashlib
import argparse
import platform

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(REPO, "data")
GOLDEN = os.path.join(REPO, "tests", "golden_features_v2.json")
MODES = ("legacy", "fast", "fast_marisa")
MODELS = {"dt12": "dt12.onnx", "rf_pruned": "rf_pruned.onnx"}
DEFAULT_REF = os.path.join(REPO, "results", "rq3", "parity_expected.json")


def golden_a2():
    """Exact-match golden v2 under all three modes. Returns per-mode counts."""
    from src import features
    from src.shared_resources import initialize_shared_resources
    with open(GOLDEN, encoding="utf-8") as f:
        golden = json.load(f)
    dictionary, ngram = initialize_shared_resources(DATA)
    report = {}
    for mode in MODES:
        features.set_kernel_mode(mode)
        mismatches = 0
        for domain, expected in golden["vectors"].items():
            got = features.extract_features(domain, domain, dictionary, ngram,
                                            skip_levenshtein=True).tolist()
            if got != expected:
                mismatches += 1
        report[mode] = {"domains": len(golden["vectors"]),
                        "mismatches": mismatches,
                        "pass": mismatches == 0}
    features.set_kernel_mode("fast")
    return report, (dictionary, ngram)


def feature_matrix(resources):
    from src import features
    from measure_serve import load_domains
    dictionary, ngram = resources
    domains = load_domains("test", limit=2000)
    out = {}
    X_fast = None
    for mode in MODES:
        features.set_kernel_mode(mode)
        X = features.extract_all_sequential(domains, dictionary, ngram,
                                            skip_levenshtein=True)
        out[mode] = {
            "sha256": hashlib.sha256(X.tobytes()).hexdigest(),
            "column_means": [round(float(m), 12) for m in X.mean(axis=0)],
        }
        if mode == "fast":
            X_fast = X
    features.set_kernel_mode("fast")
    return out, X_fast


def model_outputs(X):
    from src.onnx_model import OnnxPredictor
    out = {}
    for name, fname in MODELS.items():
        model_path = os.path.join(REPO, "models", fname)
        if not os.path.exists(model_path):
            model_path = os.path.join(REPO, "results", "rq3", "models", fname)
        pred = OnnxPredictor(model_path)
        labels = np.asarray(pred.predict(X)).ravel().astype(int)
        probas = np.asarray(pred.predict_proba(X)).astype(np.float64)
        out[name] = {
            "labels": labels.tolist(),
            "probas_rounded_7dp": np.round(probas, 7).tolist(),
        }
    return out


def snapshot():
    report, resources = golden_a2()
    feats, X_fast = feature_matrix(resources)
    models = model_outputs(X_fast)
    return {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "golden_v2_a2": report,
        "feature_matrix_2000": feats,
        "models": models,
    }


def compare(ref, got):
    verdict = {"arch_ref": ref["machine"], "arch_here": got["machine"]}
    verdict["golden_v2_a2_pass_all_modes"] = all(
        got["golden_v2_a2"][m]["pass"] for m in MODES)
    verdict["feature_hash_equal"] = {
        m: ref["feature_matrix_2000"][m]["sha256"]
           == got["feature_matrix_2000"][m]["sha256"] for m in MODES}
    verdict["models"] = {}
    for name in MODELS:
        r_lab = np.asarray(ref["models"][name]["labels"])
        g_lab = np.asarray(got["models"][name]["labels"])
        r_p = np.asarray(ref["models"][name]["probas_rounded_7dp"])
        g_p = np.asarray(got["models"][name]["probas_rounded_7dp"])
        verdict["models"][name] = {
            "labels_array_equal": bool(np.array_equal(r_lab, g_lab)),
            "probas_allclose_rtol1e-3": bool(
                np.allclose(r_p, g_p, rtol=1e-3, atol=1e-6)),
            "probas_max_abs_delta": float(np.max(np.abs(r_p - g_p))),
        }
    verdict["PASS"] = (
        verdict["golden_v2_a2_pass_all_modes"]
        and all(verdict["feature_hash_equal"].values())
        and all(m["labels_array_equal"] and m["probas_allclose_rtol1e-3"]
                for m in verdict["models"].values()))
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true",
                    help="write the reference snapshot (x86)")
    ap.add_argument("--check", action="store_true",
                    help="compare this machine against the reference")
    ap.add_argument("--ref", default=DEFAULT_REF)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    snap = snapshot()
    if args.generate:
        path = args.out or args.ref
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snap, f)
        print(f"reference written: {path} ({snap['machine']})")
        print(json.dumps(snap["golden_v2_a2"], indent=2))
        return
    if args.check:
        ref_path = args.ref
        if not os.path.exists(ref_path):
            ref_path = os.path.join(REPO, "parity", "parity_expected.json")
        with open(ref_path, encoding="utf-8") as f:
            ref = json.load(f)
        verdict = compare(ref, snap)
        text = json.dumps(verdict, indent=2)
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text)
        print(text)
        sys.exit(0 if verdict["PASS"] else 1)
    ap.error("pass --generate or --check")


if __name__ == "__main__":
    main()
