"""Probes 3 + 4 (B3 Steps 5-6): 3-feature accuracy + distillation floor.

Probe 3: baseline RF config (n_estimators=100, rs=42) on {length,
numerical_ratio, pronounceability} (cols 0,1,3 - no dictionary features).
  - E7-protocol holdout (train->test) vs the 93.179% 5-feature baseline
  - stratified 5-fold CV (acc+F1, mean+-SD) on the combined 1M set,
    with a same-protocol 5-feature CV reference
Probe 4: on all 5 features - (a) LogisticRegression (scaled, in a Pipeline),
(b) DecisionTree depth in {6,8,10,12}. 5-fold CV acc+F1; best-of-each holdout
+ single-request latency (1000 reps) + pickle size + model-load RSS subprocess.

Writes results/spike/probe3_3feature.json and probe4_distillation.json.
"""
import sys, os, json, pickle, time, subprocess, textwrap
sys.path.insert(0, r"c:/Users/samir/source/repos/dga-parallel-detection")
import numpy as np

ART = r"C:/Users/samir/AppData/Local/Temp/claude/c--Users-samir-source-repos-dga-parallel-detection/932315aa-8a56-47d8-95fa-59c19e89893b/scratchpad/artifacts"
REPO = r"c:/Users/samir/source/repos/dga-parallel-detection"
BASELINE_ACC = 93.1790

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

X_train = np.load(os.path.join(ART, "X_train5.npy"))
X_test = np.load(os.path.join(ART, "X_test5.npy"))
y_train = np.load(os.path.join(ART, "y_train.npy"))
y_test = np.load(os.path.join(ART, "y_test.npy"))
X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])
COLS3 = [0, 1, 3]  # length, numerical_ratio, pronounceability

def rf100():
    return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

def cv5(make_model, X, y, label):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []
    for i, (tr, te) in enumerate(skf.split(X, y)):
        m = make_model()
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        accs.append(accuracy_score(y[te], p)); f1s.append(f1_score(y[te], p))
        print(f"  [{label}] fold {i+1}/5 acc={accs[-1]*100:.4f}", flush=True)
    return {"acc_mean": float(np.mean(accs)) * 100, "acc_sd": float(np.std(accs, ddof=1)) * 100,
            "f1_mean": float(np.mean(f1s)), "f1_sd": float(np.std(f1s, ddof=1)),
            "folds_acc": [a * 100 for a in accs]}

def bench_single(model, row, warm=200, reps=1000):
    for _ in range(warm): model.predict(row)
    s = []
    for _ in range(reps):
        t0 = time.perf_counter_ns(); model.predict(row)
        s.append(time.perf_counter_ns() - t0)
    a = np.asarray(s, dtype=np.float64) / 1000.0
    return {"p50_us": float(np.percentile(a, 50)), "p95_us": float(np.percentile(a, 95)),
            "p99_us": float(np.percentile(a, 99))}

def load_rss_mb(pkl_path):
    """Fresh-process staged RSS: interpreter+numpy+sklearn -> +model -> +predict."""
    script = textwrap.dedent(f"""
        import os, json, pickle, psutil
        import numpy as np
        P = psutil.Process(os.getpid()); mb = lambda: P.memory_info().rss/(1024**2)
        import sklearn.ensemble, sklearn.linear_model, sklearn.tree, sklearn.pipeline, sklearn.preprocessing
        base = mb()
        with open(r"{pkl_path}", "rb") as f: m = pickle.load(f)
        loaded = mb()
        m.predict(np.array([[9.0, 0.0, 0.44, 0.004, 0.55]]))
        peak = mb()
        print(json.dumps({{"runtime_base": round(base,1), "after_load": round(loaded,1),
                           "after_predict": round(peak,1), "model_delta": round(loaded-base,1)}}))
    """)
    sp = os.path.join(ART, "_load_rss.py")
    with open(sp, "w") as f: f.write(script)
    out = subprocess.run([sys.executable, sp], capture_output=True, text=True, timeout=600)
    return json.loads(out.stdout.strip().splitlines()[-1])

# ============================ PROBE 3 ============================
print("[p3] holdout (E7 protocol), 3-feature...", flush=True)
m3 = rf100(); m3.fit(X_train[:, COLS3], y_train)
p = m3.predict(X_test[:, COLS3])
hold3 = {"acc": accuracy_score(y_test, p) * 100, "f1": float(f1_score(y_test, p))}
print(f"[p3] 3-feature holdout acc={hold3['acc']:.4f}%  (baseline 5-feat {BASELINE_ACC}%)", flush=True)

print("[p3] 5-fold CV, 3-feature...", flush=True)
cv3 = cv5(rf100, X_all[:, COLS3], y_all, "3feat")
print("[p3] 5-fold CV, 5-feature reference...", flush=True)
cv5f = cv5(rf100, X_all, y_all, "5feat")

probe3 = {
    "features": ["length", "numerical_ratio", "pronounceability"],
    "baseline_5feat_holdout_acc": BASELINE_ACC,
    "holdout_3feat": hold3,
    "holdout_delta_pp": round(hold3["acc"] - BASELINE_ACC, 4),
    "cv5_3feat": cv3, "cv5_5feat_reference": cv5f,
    "cv_delta_pp": round(cv3["acc_mean"] - cv5f["acc_mean"], 4),
    "protocol": "RF-100 rs=42; holdout = E7 train->test; CV = stratified 5-fold on combined 1M",
}
with open(os.path.join(REPO, "results/spike/probe3_3feature.json"), "w") as f:
    json.dump(probe3, f, indent=2)
print("[p3] DONE", flush=True)

# ============================ PROBE 4 ============================
def make_lr():
    return Pipeline([("scaler", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=1000, random_state=42))])

probe4 = {"cv": {}, "best": {}}
print("[p4] 5-fold CV: LogisticRegression...", flush=True)
probe4["cv"]["logreg"] = cv5(make_lr, X_all, y_all, "LR")
for d in (6, 8, 10, 12):
    print(f"[p4] 5-fold CV: DecisionTree depth={d}...", flush=True)
    probe4["cv"][f"dt_depth{d}"] = cv5(
        lambda d=d: DecisionTreeClassifier(max_depth=d, random_state=42),
        X_all, y_all, f"DT{d}")

# best-of-each on holdout + latency + RSS
best_dt_key = max((k for k in probe4["cv"] if k.startswith("dt_")),
                  key=lambda k: probe4["cv"][k]["acc_mean"])
best_depth = int(best_dt_key.replace("dt_depth", ""))
print(f"[p4] best DT depth={best_depth}; fitting holdout models...", flush=True)

lr = make_lr(); lr.fit(X_train, y_train)
dt = DecisionTreeClassifier(max_depth=best_depth, random_state=42); dt.fit(X_train, y_train)
row = X_test[7:8]
for name, m in (("logreg", lr), (f"dt_depth{best_depth}", dt)):
    pred = m.predict(X_test)
    pkl = os.path.join(ART, f"probe4_{name}.pkl")
    with open(pkl, "wb") as f: pickle.dump(m, f, protocol=pickle.HIGHEST_PROTOCOL)
    probe4["best"][name] = {
        "holdout_acc": accuracy_score(y_test, pred) * 100,
        "holdout_f1": float(f1_score(y_test, pred)),
        "single_request_us": bench_single(m, row),
        "pickle_kb": round(os.path.getsize(pkl) / 1024, 1),
        "load_rss_mb": load_rss_mb(pkl),
    }
    print(f"[p4] {name}: acc={probe4['best'][name]['holdout_acc']:.4f}% "
          f"p50={probe4['best'][name]['single_request_us']['p50_us']:.0f}us "
          f"pickle={probe4['best'][name]['pickle_kb']}KB", flush=True)

probe4["baseline_5feat_holdout_acc"] = BASELINE_ACC
probe4["protocol"] = ("5 features; CV = stratified 5-fold on combined 1M; LR in a "
                      "StandardScaler pipeline; best-of-each refit on train, holdout on test; "
                      "latency = 1000 warmed single-row reps")
with open(os.path.join(REPO, "results/spike/probe4_distillation.json"), "w") as f:
    json.dump(probe4, f, indent=2)
print("[p4] DONE", flush=True)
