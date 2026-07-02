"""Probe 2 (B3 Step 4): ONNX vs treelite single-request + ONNX-only staged RSS.

Produces results/spike/probe2_onnx_treelite.json. Protocol mirrors Batch 1:
single (1,5) float32 row, 200 warmup, 1000 timed reps, p50/p95/p99.
"""
import sys, os, json, pickle, time, subprocess, textwrap
sys.path.insert(0, r"c:/Users/samir/source/repos/dga-parallel-detection")
import numpy as np

ART = r"C:/Users/samir/AppData/Local/Temp/claude/c--Users-samir-source-repos-dga-parallel-detection/932315aa-8a56-47d8-95fa-59c19e89893b/scratchpad/artifacts"
REPO = r"c:/Users/samir/source/repos/dga-parallel-detection"
ONNX_PATH = os.path.join(ART, "rf100_5feat.onnx")

def pct(ns_list):
    a = np.asarray(ns_list, dtype=np.float64) / 1000.0
    return {"n": len(ns_list), "p50_us": float(np.percentile(a, 50)),
            "p95_us": float(np.percentile(a, 95)), "p99_us": float(np.percentile(a, 99)),
            "mean_us": float(a.mean()), "sd_us": float(a.std(ddof=1))}

def bench(fn, row, warm=200, reps=1000):
    for _ in range(warm): fn(row)
    out = []
    for _ in range(reps):
        t0 = time.perf_counter_ns(); fn(row)
        out.append(time.perf_counter_ns() - t0)
    return pct(out)

result = {"versions": {}}
import onnxruntime, skl2onnx, treelite, sklearn
result["versions"] = {"onnxruntime": onnxruntime.__version__, "skl2onnx": skl2onnx.__version__,
                      "treelite": treelite.__version__, "tl2cgen": "1.0.0 (import-blocked)",
                      "sklearn": sklearn.__version__}

print("[p2] loading model...", flush=True)
with open(os.path.join(ART, "rf100_5feat.pkl"), "rb") as f:
    model = pickle.load(f)
X_test = np.load(os.path.join(ART, "X_test5.npy"))
row64 = X_test[7:8]; row32 = row64.astype(np.float32)
sample = X_test[:2000]

# ---- 1. ONNX export ----
if not os.path.exists(ONNX_PATH):
    print("[p2] converting RF-100 -> ONNX (this can take minutes)...", flush=True)
    t0 = time.time()
    from src.onnx_model import convert_to_onnx
    convert_to_onnx(model, 5, ONNX_PATH)
    result["onnx_convert_sec"] = round(time.time() - t0, 1)
result["onnx_file_mb"] = round(os.path.getsize(ONNX_PATH) / 1e6, 1)
print(f"[p2] onnx file: {result['onnx_file_mb']} MB", flush=True)

# ---- 2. ONNX parity on real data + single-request bench ----
from src.onnx_model import get_predictor
ox = get_predictor("onnx", ONNX_PATH)
sk_pred = model.predict(sample); sk_prob = model.predict_proba(sample)
ox_pred = ox.predict(sample); ox_prob = ox.predict_proba(sample)
result["onnx_parity"] = {
    "labels_equal": bool(np.array_equal(sk_pred, ox_pred)),
    "label_mismatches": int((sk_pred != ox_pred).sum()),
    "max_abs_proba_diff": float(np.abs(sk_prob - ox_prob).max()),
    "n_rows": len(sample),
}
print(f"[p2] onnx parity: {result['onnx_parity']}", flush=True)
result["onnx_single_request"] = bench(lambda r: ox.predict(r), row32)
print(f"[p2] onnx p50 = {result['onnx_single_request']['p50_us']:.0f} us", flush=True)

# ---- 3. treelite GTIL ----
print("[p2] importing model into treelite...", flush=True)
t0 = time.time()
tl_model = treelite.sklearn.import_model(model)
result["treelite_import_sec"] = round(time.time() - t0, 1)
from treelite import gtil
gp = gtil.predict(tl_model, sample.astype(np.float32), nthread=1)
gp = np.asarray(gp)
if gp.ndim == 3:  # (n, n_target, n_class)
    gp2 = gp[:, 0, :]
else:
    gp2 = gp
gtil_pred = gp2.argmax(1) if gp2.ndim == 2 and gp2.shape[1] > 1 else (gp2.ravel() > 0.5).astype(int)
result["gtil_parity"] = {
    "labels_equal": bool(np.array_equal(sk_pred, gtil_pred)),
    "label_mismatches": int((sk_pred != gtil_pred).sum()),
    "max_abs_proba_diff": float(np.abs(sk_prob[:, 1] - (gp2[:, 1] if gp2.ndim == 2 and gp2.shape[1] > 1 else gp2.ravel())).max()),
}
print(f"[p2] gtil parity: {result['gtil_parity']}", flush=True)
result["gtil_single_request"] = bench(
    lambda r: gtil.predict(tl_model, r, nthread=1), row32)
print(f"[p2] gtil p50 = {result['gtil_single_request']['p50_us']:.0f} us", flush=True)

# ---- 4. tl2cgen: import-blocked on this box ----
result["tl2cgen_status"] = (
    "BLOCKED on Windows: import fails in libloader (os.add_dll_directory on "
    "non-existent 'C:\\Program Files\\Python311\\Library\\bin', a conda-style "
    "path); native compile would additionally require MSVC. Per plan: bench "
    "GTIL now, defer compiled-lib measurement to Batch 6 (Linux/Docker).")

# ---- 5. ONNX-only staged RSS (fresh subprocess, no sklearn import) ----
print("[p2] ONNX-only staged RSS subprocess...", flush=True)
rss_script = textwrap.dedent(f"""
    import os, json, psutil
    P = psutil.Process(os.getpid())
    mb = lambda: P.memory_info().rss / (1024**2)
    stages = {{}}
    stages['interpreter'] = round(mb(), 1)
    import numpy as np
    stages['import_numpy'] = round(mb(), 1)
    import onnxruntime as rt
    stages['import_onnxruntime'] = round(mb(), 1)
    sess = rt.InferenceSession(r"{ONNX_PATH}")
    stages['load_model_session'] = round(mb(), 1)
    x = np.array([[9.0, 0.0, 0.44, 0.004, 0.55]], dtype=np.float32)
    sess.run(None, {{'float_input': x}})
    stages['first_inference'] = round(mb(), 1)
    assert 'sklearn' not in str(list(__import__('sys').modules.keys()))
    print(json.dumps(stages))
""")
sp = os.path.join(ART, "_onnx_rss_stage.py")
with open(sp, "w") as f:
    f.write(rss_script)
out = subprocess.run([sys.executable, sp], capture_output=True, text=True, timeout=600)
result["onnx_only_rss_mb"] = json.loads(out.stdout.strip().splitlines()[-1])
print(f"[p2] onnx-only RSS: {result['onnx_only_rss_mb']}", flush=True)

with open(os.path.join(REPO, "results/spike/probe2_onnx_treelite.json"), "w") as f:
    json.dump(result, f, indent=2)
print("[p2] DONE", flush=True)
