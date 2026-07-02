"""B4 Step 1: freeze the model grid {RF-100, RF-pruned, DT-12, LR} + manifest.

RF-pruned selection rule (per brief): smallest pickle within 1 pp of 93.179%.
Fixed seeds everywhere (random_state=42). Small artifacts + manifest written to
results/rq2/models/; RF-100 (491 MB) stays out of git - manifest records its
config + data hash for exact reproduction.
"""
import sys, os, json, time, pickle, hashlib
sys.path.insert(0, r"c:/Users/samir/source/repos/dga-parallel-detection")
import numpy as np

ART = r"C:/Users/samir/AppData/Local/Temp/claude/c--Users-samir-source-repos-dga-parallel-detection/932315aa-8a56-47d8-95fa-59c19e89893b/scratchpad/artifacts"
REPO = r"c:/Users/samir/source/repos/dga-parallel-detection"
OUT = os.path.join(REPO, "results/rq2/models")
BASELINE = 93.1790

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from src.classifier import run_pruning_sweep, train_pruned_random_forest

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    X_train = np.load(os.path.join(ART, "X_train5.npy"))
    X_test = np.load(os.path.join(ART, "X_test5.npy"))
    y_train = np.load(os.path.join(ART, "y_train.npy"))
    y_test = np.load(os.path.join(ART, "y_test.npy"))
    data_hash = hashlib.sha256(X_train.tobytes()).hexdigest()[:16]
    print(f"[grid] data: X_train{X_train.shape} sha256[:16]={data_hash}", flush=True)

    manifest = {"seed": 42, "train_matrix_sha256_16": data_hash,
                "baseline_acc_pct": BASELINE, "models": {}}

    def record(name, model, acc, f1, path, extra=None):
        e = {"accuracy_pct": round(acc * 100, 4), "f1": round(f1, 4),
             "pickle_bytes": os.path.getsize(path) if path and os.path.exists(path) else None,
             "path": os.path.relpath(path, REPO).replace("\\", "/") if path else "NOT COMMITTED (491 MB; reproduce: RF n_estimators=100 rs=42 on train matrix)"}
        if hasattr(model, "estimators_"):
            e["total_nodes"] = int(sum(t.tree_.node_count for t in model.estimators_))
        elif hasattr(model, "tree_"):
            e["total_nodes"] = int(model.tree_.node_count)
        if extra: e.update(extra)
        manifest["models"][name] = e
        print(f"[grid] {name}: acc={e['accuracy_pct']}% f1={e['f1']} "
              f"size={e['pickle_bytes']} nodes={e.get('total_nodes')}", flush=True)

    # ---- RF-100 (exists, bit-exact verified in B3) ----
    with open(os.path.join(ART, "rf100_5feat.pkl"), "rb") as f:
        rf100 = pickle.load(f)
    p = rf100.predict(X_test)
    record("rf100", rf100, accuracy_score(y_test, p), f1_score(y_test, p), None,
           {"config": "RandomForestClassifier(n_estimators=100, random_state=42)"})

    # ---- RF-pruned sweep ----
    print("[grid] pruning sweep (5 depths x 4 alphas)...", flush=True)
    t0 = time.time()
    sweep = run_pruning_sweep(X_train, y_train, X_test, y_test, n_estimators=100)
    print(f"[grid] sweep done [{time.time()-t0:.0f}s]", flush=True)
    with open(os.path.join(REPO, "results/rq2/pruning_sweep.json"), "w") as f:
        json.dump(sweep, f, indent=2)
    ok = [r for r in sweep if r["accuracy"] * 100 >= BASELINE - 1.0]
    best = min(ok, key=lambda r: r["size_bytes"])
    print(f"[grid] selected pruned config: depth={best['max_depth']} alpha={best['ccp_alpha']} "
          f"acc={best['accuracy']*100:.4f}% size={best['size_bytes']/1e6:.1f}MB", flush=True)
    d = None if best["max_depth"] == -1 else best["max_depth"]
    rf_pruned = train_pruned_random_forest(X_train, y_train, n_estimators=100,
                                           max_depth=d, ccp_alpha=best["ccp_alpha"])
    p = rf_pruned.predict(X_test)
    path = os.path.join(OUT, "rf_pruned.pkl")
    with open(path, "wb") as f:
        pickle.dump(rf_pruned, f, protocol=pickle.HIGHEST_PROTOCOL)
    record("rf_pruned", rf_pruned, accuracy_score(y_test, p), f1_score(y_test, p), path,
           {"config": f"RF(n_estimators=100, max_depth={d}, ccp_alpha={best['ccp_alpha']}, rs=42)"})

    # ---- DT-12 (frozen probe-4 config) ----
    dt = DecisionTreeClassifier(max_depth=12, random_state=42)
    dt.fit(X_train, y_train)
    p = dt.predict(X_test)
    path = os.path.join(OUT, "dt12.pkl")
    with open(path, "wb") as f:
        pickle.dump(dt, f, protocol=pickle.HIGHEST_PROTOCOL)
    record("dt12", dt, accuracy_score(y_test, p), f1_score(y_test, p), path,
           {"config": "DecisionTreeClassifier(max_depth=12, random_state=42)"})

    # ---- LR pipeline ----
    lr = Pipeline([("scaler", StandardScaler()),
                   ("lr", LogisticRegression(max_iter=1000, random_state=42))])
    lr.fit(X_train, y_train)
    p = lr.predict(X_test)
    path = os.path.join(OUT, "logreg.pkl")
    with open(path, "wb") as f:
        pickle.dump(lr, f, protocol=pickle.HIGHEST_PROTOCOL)
    record("logreg", lr, accuracy_score(y_test, p), f1_score(y_test, p), path,
           {"config": "StandardScaler + LogisticRegression(max_iter=1000, rs=42)"})

    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("[grid] manifest written. DONE", flush=True)
