"""Probe-2 finalization (resilient):
1. GTIL re-bench + parity (full percentiles) — first, on the idle box.
2. ONNX-only staged RSS subprocess, ORT_ENABLE_ALL (the config proven to
   complete), MEMORY-ONLY, stdout streamed to a stage log so a timeout still
   yields the runtime-floor stages.
3. Write the fully reconstructed results/spike/probe2_onnx_treelite.json —
   ALWAYS, even if the RSS subprocess times out (failure documented in-JSON).
"""
import sys, os, json, time, subprocess, textwrap

ART = r"C:/Users/samir/AppData/Local/Temp/claude/c--Users-samir-source-repos-dga-parallel-detection/932315aa-8a56-47d8-95fa-59c19e89893b/scratchpad/artifacts"
REPO = r"c:/Users/samir/source/repos/dga-parallel-detection"
ONNX_PATH = os.path.join(ART, "rf100_5feat.onnx")
OUT_JSON = os.path.join(REPO, "results/spike/probe2_onnx_treelite.json")
STAGE_LOG = os.path.join(ART, "_rss_stages.jsonl")

sys.path.insert(0, REPO)
import pickle
import numpy as np

# ---------- 1. GTIL re-bench ----------
print("[final] treelite import + GTIL bench...", flush=True)
import treelite
from treelite import gtil
import onnxruntime, skl2onnx, sklearn

with open(os.path.join(ART, "rf100_5feat.pkl"), "rb") as f:
    model = pickle.load(f)
X_test = np.load(os.path.join(ART, "X_test5.npy"))
row32 = X_test[7:8].astype(np.float32)
sample32 = X_test[:2000].astype(np.float32)

t0 = time.time()
tl_model = treelite.sklearn.import_model(model)
tl_import_sec = round(time.time() - t0, 1)
gp = np.asarray(gtil.predict(tl_model, sample32, nthread=1))
gp2 = gp[:, 0, :] if gp.ndim == 3 else gp
sk_pred = model.predict(X_test[:2000]); sk_prob = model.predict_proba(X_test[:2000])
for _ in range(200):
    gtil.predict(tl_model, row32, nthread=1)
s = []
for _ in range(1000):
    t = time.perf_counter_ns()
    gtil.predict(tl_model, row32, nthread=1)
    s.append(time.perf_counter_ns() - t)
a = np.asarray(s, dtype=np.float64) / 1000.0
gtil_bench = {"p50_us": float(np.percentile(a, 50)), "p95_us": float(np.percentile(a, 95)),
              "p99_us": float(np.percentile(a, 99)), "mean_us": float(a.mean()),
              "sd_us": float(a.std(ddof=1)), "n": 1000}
gtil_parity = {"labels_equal": bool(np.array_equal(sk_pred, gp2.argmax(1))),
               "label_mismatches": int((sk_pred != gp2.argmax(1)).sum()),
               "max_abs_proba_diff": float(np.abs(sk_prob[:, 1] - gp2[:, 1]).max())}
print(f"[final] gtil p50={gtil_bench['p50_us']:.0f}us import={tl_import_sec}s "
      f"parity={gtil_parity['labels_equal']}", flush=True)

# free the big objects before the RSS subprocess runs
del model, tl_model, gp, gp2, sk_prob
import gc; gc.collect()

# ---------- 2. ONNX-only staged RSS (memory-only, streamed stages) ----------
rss_script = textwrap.dedent(f"""
    import os, sys, json, time, psutil
    P = psutil.Process(os.getpid())
    def emit(k):
        print(json.dumps({{k: round(P.memory_info().rss / (1024**2), 1),
                           "t": round(time.time() - T0, 1)}}), flush=True)
    T0 = time.time()
    emit('interpreter')
    import numpy as np
    emit('import_numpy')
    import onnxruntime as rt
    emit('import_onnxruntime')
    sess = rt.InferenceSession(r"{ONNX_PATH}")   # default = ORT_ENABLE_ALL
    emit('load_model_session')
    x = np.array([[9.0, 0.0, 0.44, 0.004, 0.55]], dtype=np.float32)
    sess.run(None, {{'float_input': x}})
    emit('first_inference')
    assert 'sklearn' not in sys.modules
    print(json.dumps({{"done": True}}), flush=True)
""")
sp = os.path.join(ART, "_onnx_rss_stage3.py")
with open(sp, "w") as f:
    f.write(rss_script)

print("[final] RSS subprocess (ENABLE_ALL, memory-only, 45 min budget)...", flush=True)
rss_status = "ok"
with open(STAGE_LOG, "w") as logf:
    proc = subprocess.Popen([sys.executable, sp], stdout=logf,
                            stderr=subprocess.DEVNULL, text=True)
    try:
        proc.wait(timeout=2700)
        if proc.returncode != 0:
            rss_status = f"subprocess exit {proc.returncode}"
    except subprocess.TimeoutExpired:
        proc.kill()
        rss_status = "TIMEOUT >45min at session init"

stages, timings = {}, {}
with open(STAGE_LOG) as f:
    for line in f:
        d = json.loads(line)
        for k, v in d.items():
            if k == "t":
                continue
            if k == "done":
                continue
            stages[k] = v
            timings[k + "_at_sec"] = d.get("t")
print(f"[final] RSS status={rss_status} stages={stages}", flush=True)

# ---------- 3. Write the reconstructed probe-2 JSON ----------
result = {
    "versions": {"onnxruntime": onnxruntime.__version__, "skl2onnx": skl2onnx.__version__,
                 "treelite": treelite.__version__, "tl2cgen": "1.0.0 (import-blocked)",
                 "sklearn": sklearn.__version__},
    "onnx_file_mb": round(os.path.getsize(ONNX_PATH) / 1e6, 1),
    "onnx_parity": {"labels_equal": True, "label_mismatches": 0,
                    "max_abs_proba_diff": 2.2411346434214607e-07, "n_rows": 2000,
                    "provenance": "main probe-2 run log"},
    "onnx_single_request": {"p50_us": 18.0,
                            "provenance": "main probe-2 run log (ENABLE_ALL session; p50 only)"},
    "gtil_parity": gtil_parity,
    "gtil_single_request": gtil_bench,
    "treelite_import_sec": tl_import_sec,
    "tl2cgen_status": ("BLOCKED on Windows: import fails in libloader "
                       "(os.add_dll_directory on non-existent conda-style path); native "
                       "compile would additionally require MSVC. GTIL benched instead; "
                       "compiled-lib deferred to Batch 6 (Linux/Docker) per plan."),
    "onnx_only_rss_mb": stages,
    "onnx_only_rss_stage_seconds": timings,
    "onnx_only_rss_status": rss_status,
    "onnx_session_init_finding": (
        "Session init on the UNPRUNED 6.1M-node RF-100 is pathological: "
        "ORT_ENABLE_ALL took ~15-25 min (main run), ORT_DISABLE_ALL exceeded a "
        "30-min budget (so the cost is TreeEnsemble kernel construction, not the "
        "graph optimizer). Cold-start of the uncompressed model is infeasible on "
        "edge; structural pruning is a prerequisite for the ONNX path's LOAD TIME, "
        "not only its RSS."),
    "protocol": "single (1,5) float32 row, 200 warmup, 1000 timed reps",
}
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print("[final] wrote probe2_onnx_treelite.json DONE", flush=True)
