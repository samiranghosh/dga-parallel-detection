"""Probe-2 RSS stage, standalone: ONNX-only staged RSS with ORT_DISABLE_ALL
(fast session init) + in-process latency A/B vs the ENABLE_ALL number (18 us)
measured in the main probe-2 run. Merges results into probe2_onnx_treelite.json.
"""
import sys, os, json, time, subprocess, textwrap

ART = r"C:/Users/samir/AppData/Local/Temp/claude/c--Users-samir-source-repos-dga-parallel-detection/932315aa-8a56-47d8-95fa-59c19e89893b/scratchpad/artifacts"
REPO = r"c:/Users/samir/source/repos/dga-parallel-detection"
ONNX_PATH = os.path.join(ART, "rf100_5feat.onnx")
OUT_JSON = os.path.join(REPO, "results/spike/probe2_onnx_treelite.json")

rss_script = textwrap.dedent(f"""
    import os, sys, json, time, psutil
    P = psutil.Process(os.getpid())
    mb = lambda: round(P.memory_info().rss / (1024**2), 1)
    stages = {{}}
    stages['interpreter'] = mb()
    import numpy as np
    stages['import_numpy'] = mb()
    import onnxruntime as rt
    stages['import_onnxruntime'] = mb()
    so = rt.SessionOptions()
    so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL
    t0 = time.time()
    sess = rt.InferenceSession(r"{ONNX_PATH}", so)
    init_sec = round(time.time() - t0, 1)
    stages['load_model_session'] = mb()
    x = np.array([[9.0, 0.0, 0.44, 0.004, 0.55]], dtype=np.float32)
    sess.run(None, {{'float_input': x}})
    stages['first_inference'] = mb()
    # in-process single-request latency with DISABLE_ALL (A/B vs 18 us ENABLE_ALL)
    for _ in range(200):
        sess.run(None, {{'float_input': x}})
    s = []
    for _ in range(1000):
        t = time.perf_counter_ns()
        sess.run(None, {{'float_input': x}})
        s.append(time.perf_counter_ns() - t)
    a = np.asarray(s, dtype=np.float64) / 1000.0
    assert 'sklearn' not in sys.modules, "sklearn leaked into ONNX-only process"
    print(json.dumps({{
        'stages_mb': stages,
        'session_init_sec_disable_all': init_sec,
        'latency_disable_all_us': {{'p50': float(np.percentile(a, 50)),
                                    'p95': float(np.percentile(a, 95)),
                                    'p99': float(np.percentile(a, 99))}},
    }}))
""")
sp = os.path.join(ART, "_onnx_rss_stage2.py")
with open(sp, "w") as f:
    f.write(rss_script)
print("[rss] launching ONNX-only subprocess (ORT_DISABLE_ALL)...", flush=True)
out = subprocess.run([sys.executable, sp], capture_output=True, text=True, timeout=1800)
if out.returncode != 0:
    print(out.stderr[-2000:], flush=True)
    raise SystemExit(1)
payload = json.loads(out.stdout.strip().splitlines()[-1])
print("[rss]", json.dumps(payload, indent=2), flush=True)

# Rebuild the full probe-2 JSON (the main run crashed at the RSS stage before
# writing). Values measured in the main run but not re-runnable cheaply are
# embedded with provenance = "main-run log".
print("[rss] re-benching GTIL for full percentiles...", flush=True)
sys.path.insert(0, REPO)
import pickle
import numpy as np
import treelite
from treelite import gtil
import onnxruntime, skl2onnx, sklearn

with open(os.path.join(ART, "rf100_5feat.pkl"), "rb") as f:
    model = pickle.load(f)
X_test = np.load(os.path.join(ART, "X_test5.npy"))
row32 = X_test[7:8].astype(np.float32)
sample = X_test[:2000].astype(np.float32)

t0 = time.time()
tl_model = treelite.sklearn.import_model(model)
tl_import_sec = round(time.time() - t0, 1)
gp = np.asarray(gtil.predict(tl_model, sample, nthread=1))
gp2 = gp[:, 0, :] if gp.ndim == 3 else gp
sk_pred = model.predict(X_test[:2000]); sk_prob = model.predict_proba(X_test[:2000])
gtil_pred = gp2.argmax(1)
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

result = {
    "versions": {"onnxruntime": onnxruntime.__version__, "skl2onnx": skl2onnx.__version__,
                 "treelite": treelite.__version__, "tl2cgen": "1.0.0 (import-blocked)",
                 "sklearn": sklearn.__version__},
    "onnx_file_mb": round(os.path.getsize(ONNX_PATH) / 1e6, 1),
    "onnx_parity": {"labels_equal": True, "label_mismatches": 0,
                    "max_abs_proba_diff": 2.2411346434214607e-07, "n_rows": 2000,
                    "provenance": "main probe-2 run log"},
    "onnx_single_request_enable_all": {"p50_us": 18.0,
                                       "provenance": "main probe-2 run log (p50 only)"},
    "onnx_single_request_disable_all": payload["latency_disable_all_us"],
    "gtil_parity": {"labels_equal": bool(np.array_equal(sk_pred, gtil_pred)),
                    "label_mismatches": int((sk_pred != gtil_pred).sum()),
                    "max_abs_proba_diff": float(np.abs(sk_prob[:, 1] - gp2[:, 1]).max())},
    "gtil_single_request": gtil_bench,
    "treelite_import_sec": tl_import_sec,
    "tl2cgen_status": ("BLOCKED on Windows: import fails in libloader "
                       "(os.add_dll_directory on non-existent conda-style path "
                       "'C:\\Program Files\\Python311\\Library\\bin'); native compile "
                       "would additionally require MSVC. Per plan: GTIL benched now, "
                       "compiled-lib deferred to Batch 6 (Linux/Docker)."),
    "onnx_only_rss": payload,
    "onnx_session_init_note": (
        "ORT_ENABLE_ALL session init on the unpruned 6.1M-node RF-100 took >15 min "
        "with multi-GB transient RSS (observed in the main probe-2 run; it also "
        "timed out the first RSS subprocess at 600 s). ORT_DISABLE_ALL init is "
        "measured here (session_init_sec_disable_all) with a latency A/B so "
        "Batch 4 can pick the serving config."),
    "protocol": "single (1,5) float32 row, 200 warmup, 1000 timed reps",
}
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print(f"[rss] gtil re-bench p50={gtil_bench['p50_us']:.0f}us import={tl_import_sec}s", flush=True)
print("[rss] wrote probe2_onnx_treelite.json DONE", flush=True)
