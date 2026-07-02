"""B4 Steps 5-6: compression arms + runtime arms.

Step 5: float32 cast of RF tree arrays (RSS + accuracy delta) on RF-100 and
RF-pruned; int8 = INVESTIGATE-ONLY (feasibility analysis, no reliance).
Step 6: ONNX-export all four models (ort 1.25.1, zipmap=False), parity-assert
each (labels array_equal, probas rtol=1e-3), then per-variant under the
CANONICAL protocol: single-request p50/p95/p99 (sklearn path + ONNX path) and
staged RSS (sklearn-path and ONNX-only fresh subprocesses).
RF-100-ONNX cold-start rubric: try optimized .ort save; if re-init > 60 s ->
"disqualified on cold-start", record and move on.

Writes results/rq2/step5_compression.json + step6_runtime.json.
NOTE: latency sections require an otherwise-idle box.
"""
import sys, os, json, time, pickle, subprocess, textwrap
sys.path.insert(0, r"c:/Users/samir/source/repos/dga-parallel-detection")
import numpy as np

ART = r"C:/Users/samir/AppData/Local/Temp/claude/c--Users-samir-source-repos-dga-parallel-detection/932315aa-8a56-47d8-95fa-59c19e89893b/scratchpad/artifacts"
REPO = r"c:/Users/samir/source/repos/dga-parallel-detection"
RQ2 = os.path.join(REPO, "results/rq2")

from sklearn.metrics import accuracy_score

def stage_rss(pkl_or_onnx, kind):
    """Fresh-process staged RSS. kind: 'sklearn' or 'onnx'."""
    if kind == "sklearn":
        body = f"""
            import os, json, pickle, psutil, numpy as np
            P = psutil.Process(os.getpid()); mb = lambda: round(P.memory_info().rss/(1024**2),1)
            s = {{}}
            s['interpreter'] = mb()
            import sklearn.ensemble, sklearn.tree, sklearn.linear_model, sklearn.pipeline, sklearn.preprocessing
            s['runtime'] = mb()
            with open(r"{pkl_or_onnx}", "rb") as f: m = pickle.load(f)
            s['loaded'] = mb()
            m.predict(np.array([[9.0, 0.0, 0.44, 0.004, 0.55]]))
            s['first_predict'] = mb()
            print(json.dumps(s))
        """
    else:
        body = f"""
            import os, sys, json, psutil, numpy as np
            P = psutil.Process(os.getpid()); mb = lambda: round(P.memory_info().rss/(1024**2),1)
            s = {{}}
            s['interpreter'] = mb()
            import onnxruntime as rt
            s['runtime'] = mb()
            sess = rt.InferenceSession(r"{pkl_or_onnx}")
            s['loaded'] = mb()
            sess.run(None, {{'float_input': np.array([[9.0,0.0,0.44,0.004,0.55]], dtype=np.float32)}})
            s['first_predict'] = mb()
            assert 'sklearn' not in sys.modules
            print(json.dumps(s))
        """
    sp = os.path.join(ART, "_rss_tmp.py")
    with open(sp, "w") as f:
        f.write(textwrap.dedent(body))
    out = subprocess.run([sys.executable, sp], capture_output=True, text=True, timeout=900)
    return json.loads(out.stdout.strip().splitlines()[-1])

def bench_predict(fn, row, warm=200, reps=1000):
    for _ in range(warm): fn(row)
    s = []
    for _ in range(reps):
        t0 = time.perf_counter_ns(); fn(row)
        s.append(time.perf_counter_ns() - t0)
    a = np.asarray(s, dtype=np.float64) / 1000.0
    return {"p50_us": round(float(np.percentile(a, 50)), 1),
            "p95_us": round(float(np.percentile(a, 95)), 1),
            "p99_us": round(float(np.percentile(a, 99)), 1)}

if __name__ == "__main__":
    X_test = np.load(os.path.join(ART, "X_test5.npy"))
    y_test = np.load(os.path.join(ART, "y_test.npy"))
    row64 = X_test[7:8]; row32 = row64.astype(np.float32)
    sample = X_test[:2000]

    models = {}
    with open(os.path.join(ART, "rf100_5feat.pkl"), "rb") as f:
        models["rf100"] = pickle.load(f)
    for n in ("rf_pruned", "dt12", "logreg"):
        with open(os.path.join(RQ2, "models", f"{n}.pkl"), "rb") as f:
            models[n] = pickle.load(f)
    for m in models.values():
        if hasattr(m, "n_jobs"): m.n_jobs = 1

    # ================= Step 5: compression arms =================
    step5 = {}
    for name in ("rf100", "rf_pruned"):
        m = models[name]
        base_acc = accuracy_score(y_test, m.predict(X_test))
        # float32 cast: tree values + thresholds are float64 ndarrays inside
        # each estimator's tree_. sklearn's Tree struct is fixed-dtype, so a
        # TRUE in-place cast isn't supported by the public API - measure the
        # array-level saving instead and validate accuracy via float32 INPUT
        # (the traversal compares x[f] <= threshold; casting inputs is the
        # deployable half; array cast requires a custom serving struct).
        arrays = 0
        for est in m.estimators_:
            t = est.tree_
            arrays += t.value.nbytes + t.threshold.nbytes
        acc32 = accuracy_score(y_test, m.predict(X_test.astype(np.float32)))
        step5[name] = {
            "float64_value_threshold_bytes": int(arrays),
            "float32_savable_bytes": int(arrays // 2),
            "accuracy_float64_input_pct": round(base_acc * 100, 4),
            "accuracy_float32_input_pct": round(acc32 * 100, 4),
            "note": ("sklearn Tree arrays are fixed float64; true cast needs a custom "
                     "serving structure (or ONNX, which is float32 end-to-end - measured "
                     "in step 6). Savable-if-cast is exact arithmetic on value+threshold."),
        }
        print(f"[s5] {name}: arrays={arrays/1e6:.1f}MB savable~{arrays/2e6:.1f}MB "
              f"acc64={base_acc*100:.4f}% acc32-in={acc32*100:.4f}%", flush=True)
    step5["int8_investigation"] = {
        "verdict": "NOT APPLICABLE as weight quantization",
        "reason": ("trees store split thresholds + leaf class distributions, not "
                   "weights; int8-quantizing thresholds changes decision boundaries "
                   "directly (unlike NN weight rounding). Feature value ranges here "
                   "span [0,1] ratios and integer lengths - 8-bit threshold grids "
                   "would collapse distinct splits on the ratio features. ONNX "
                   "ai.onnx.ml TreeEnsemble has no int8 mode either. Reported as "
                   "investigate-only per plan; not pursued."),
    }
    with open(os.path.join(RQ2, "step5_compression.json"), "w") as f:
        json.dump(step5, f, indent=2)

    # ================= Step 6: runtime arms =================
    from src.onnx_model import convert_to_onnx, get_predictor
    step6 = {"parity": {}, "latency_sklearn_us": {}, "latency_onnx_us": {},
             "rss_sklearn_mb": {}, "rss_onnx_mb": {}}

    onnx_paths = {"rf100": os.path.join(ART, "rf100_5feat.onnx")}  # exists (B3)
    for name in ("rf_pruned", "dt12", "logreg"):
        p = os.path.join(ART, f"{name}.onnx")
        if not os.path.exists(p):
            print(f"[s6] converting {name} -> onnx...", flush=True)
            convert_to_onnx(models[name], 5, p)
        onnx_paths[name] = p

    for name, m in models.items():
        if name == "rf100":
            # parity already artifact-verified in B3 (0/2000, 2.24e-7); the
            # 30-min session init is handled in the dedicated cold-start block.
            continue
        ox = get_predictor("onnx", onnx_paths[name])
        skp, oxp = m.predict(sample), ox.predict(sample)
        sk_pr, ox_pr = m.predict_proba(sample), ox.predict_proba(sample)
        assert np.array_equal(skp, oxp), f"{name}: ONNX label parity FAILED"
        assert np.allclose(sk_pr, ox_pr, rtol=1e-3, atol=1e-4), f"{name}: proba parity FAILED"
        step6["parity"][name] = {"labels_equal": True,
                                 "max_abs_proba_diff": float(np.abs(sk_pr - ox_pr).max())}
        print(f"[s6] {name} parity OK ({step6['parity'][name]['max_abs_proba_diff']:.2e})", flush=True)

    # Per-variant latency under the CANONICAL protocol (B4 Step 0b): real-domain
    # extract+predict, 200 warmup / 1000 reps; total+feature+predict reported.
    from src.shared_resources import initialize_shared_resources
    from src.latency_harness import measure_canonical
    dictionary, ngram = initialize_shared_resources(os.path.join(REPO, "data/"))
    for name, m in models.items():
        step6["latency_sklearn_us"][name] = measure_canonical(m, dictionary, ngram,
                                                              os.path.join(REPO, "data/"))
        print(f"[s6] {name} sklearn CANONICAL total p50="
              f"{step6['latency_sklearn_us'][name]['total']['p50_us']:.0f}us", flush=True)
    for name in models:
        if name == "rf100":
            continue  # rf100-onnx handled in the cold-start block
        ox = get_predictor("onnx", onnx_paths[name])
        step6["latency_onnx_us"][name] = measure_canonical(ox, dictionary, ngram,
                                                           os.path.join(REPO, "data/"))
        print(f"[s6] {name} onnx CANONICAL total p50="
              f"{step6['latency_onnx_us'][name]['total']['p50_us']:.0f}us", flush=True)

    for name in models:
        pkl = (os.path.join(ART, "rf100_5feat.pkl") if name == "rf100"
               else os.path.join(RQ2, "models", f"{name}.pkl"))
        step6["rss_sklearn_mb"][name] = stage_rss(pkl, "sklearn")
        print(f"[s6] {name} sklearn RSS={step6['rss_sklearn_mb'][name]}", flush=True)
        if name != "rf100":  # rf100 onnx session init ~30 min: cold-start block
            step6["rss_onnx_mb"][name] = stage_rss(onnx_paths[name], "onnx")
            print(f"[s6] {name} onnx RSS={step6['rss_onnx_mb'][name]}", flush=True)

    step6["rf100_onnx"] = {
        "steady_state_note": ("latency 18 us p50 + session RSS 990.7 MiB measured in B3 "
                              "(results/spike/probe2_onnx_treelite.json); init 1932 s"),
        "cold_start": "adjudicated by the dedicated cold-start block (step6_rf100_coldstart.json)",
    }
    with open(os.path.join(RQ2, "step6_runtime.json"), "w") as f:
        json.dump(step6, f, indent=2)
    print("[s56] DONE", flush=True)
