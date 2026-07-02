"""B4 Step 7: feature rigor on the gate-winning model (arg --model dt12|rf100).

- Exhaustive 31-subset sweep of the 5 features (extends E7's prefix ablation;
  supersedes probe 3's single 3-feature point). E7 protocol: train on
  ExtraHop train, evaluate on holdout, fixed seed.
- SHAP TreeExplainer summary (mean |SHAP| per feature) on a 20k holdout sample.
- Verdict: is the full 5-feature set on the accuracy-vs-cost Pareto front?

Writes results/rq2/step7_subsets.json (+ shap values inside).
"""
import sys, os, json, time, itertools, argparse
sys.path.insert(0, r"c:/Users/samir/source/repos/dga-parallel-detection")
import numpy as np

ART = r"C:/Users/samir/AppData/Local/Temp/claude/c--Users-samir-source-repos-dga-parallel-detection/932315aa-8a56-47d8-95fa-59c19e89893b/scratchpad/artifacts"
REPO = r"c:/Users/samir/source/repos/dga-parallel-detection"
RQ2 = os.path.join(REPO, "results/rq2")
FEATS = ["length", "numerical_ratio", "meaningful_word_ratio",
         "pronounceability", "lms_percentage"]

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def make_model(kind):
    if kind == "dt12":
        return DecisionTreeClassifier(max_depth=12, random_state=42)
    return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dt12", choices=["dt12", "rf100"])
    args = ap.parse_args()

    X_train = np.load(os.path.join(ART, "X_train5.npy"))
    X_test = np.load(os.path.join(ART, "X_test5.npy"))
    y_train = np.load(os.path.join(ART, "y_train.npy"))
    y_test = np.load(os.path.join(ART, "y_test.npy"))

    results = []
    t0 = time.time()
    for r in range(1, 6):
        for combo in itertools.combinations(range(5), r):
            m = make_model(args.model)
            m.fit(X_train[:, combo], y_train)
            p = m.predict(X_test[:, combo])
            results.append({
                "features": [FEATS[i] for i in combo],
                "n_features": r,
                "uses_dictionary": bool({2, 4} & set(combo)),
                "accuracy_pct": round(accuracy_score(y_test, p) * 100, 4),
                "f1": round(f1_score(y_test, p), 4),
            })
            print(f"[s7] {results[-1]['features']}: {results[-1]['accuracy_pct']}%", flush=True)
    print(f"[s7] 31 subsets done [{time.time()-t0:.0f}s]", flush=True)

    best_per_n = {r: max((x for x in results if x["n_features"] == r),
                         key=lambda x: x["accuracy_pct"]) for r in range(1, 6)}
    full5 = next(x for x in results if x["n_features"] == 5)
    pareto_holds = all(full5["accuracy_pct"] >= best_per_n[r]["accuracy_pct"]
                       for r in range(1, 5))

    # SHAP on the full-5 gate-winning model
    import shap
    m5 = make_model(args.model); m5.fit(X_train, y_train)
    expl = shap.TreeExplainer(m5)
    sample = X_test[:20000]
    sv = expl.shap_values(sample)
    sv = np.asarray(sv)
    # normalize shapes across shap versions: want (n, 5) for the positive class
    if sv.ndim == 3:
        sv2 = sv[:, :, 1] if sv.shape[2] == 2 else sv[1]
    else:
        sv2 = sv
    mean_abs = np.abs(sv2).mean(axis=0)
    shap_summary = {FEATS[i]: round(float(mean_abs[i]), 5) for i in range(5)}

    out = {"model": args.model, "protocol": "E7 holdout (train->test), rs=42",
           "subsets": results, "best_per_n": best_per_n,
           "full5_accuracy_pct": full5["accuracy_pct"],
           "pareto_front_holds_for_5": bool(pareto_holds),
           "shap_mean_abs_20k": shap_summary,
           "shap_ranking": sorted(shap_summary, key=shap_summary.get, reverse=True)}
    with open(os.path.join(RQ2, "step7_subsets.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[s7] full5={full5['accuracy_pct']}% pareto_holds={pareto_holds}", flush=True)
    print(f"[s7] shap ranking: {out['shap_ranking']}", flush=True)
    print("[s7] DONE", flush=True)
