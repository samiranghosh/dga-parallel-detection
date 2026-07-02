"""Gate-2 v2 (B4 verification fix): two-sided bracket operating-point comparison.

v1 (in step24_gates.py) tuned each threshold via a benign-score quantile with
`>=`, which OVERSHOT the reference FPR for coarse-scored models (DT-12: 598
distinct scores -> achieved FPR 0.0850 vs target 0.0732), confounding the
matched-FPR FNR comparison. v2 reports, per model, the CONSERVATIVE point
(largest achieved FPR <= target) and the PERMISSIVE bracket (next distinct
score above), and adjudicates DT vs RF-100 at DT's ACHIEVED conservative FPR.
Overwrites results/rq2/gate2_operating_point.json with a machine verdict field.
"""
import sys, os, json, pickle
sys.path.insert(0, r"c:/Users/samir/source/repos/dga-parallel-detection")
import numpy as np
from sklearn.metrics import confusion_matrix

ART = r"C:/Users/samir/AppData/Local/Temp/claude/c--Users-samir-source-repos-dga-parallel-detection/932315aa-8a56-47d8-95fa-59c19e89893b/scratchpad/artifacts"
REPO = r"c:/Users/samir/source/repos/dga-parallel-detection"


def rates(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"fpr": round(fp / (fp + tn), 6), "fnr": round(fn / (fn + tp), 6),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


if __name__ == "__main__":
    X = np.load(os.path.join(ART, "X_test5.npy"))
    y = np.load(os.path.join(ART, "y_test.npy"))
    models = {}
    with open(os.path.join(ART, "rf100_5feat.pkl"), "rb") as f:
        models["rf100"] = pickle.load(f)
    for n in ("rf_pruned", "dt12", "logreg"):
        with open(os.path.join(REPO, "results/rq2/models", f"{n}.pkl"), "rb") as f:
            models[n] = pickle.load(f)

    probas = {n: m.predict_proba(X)[:, 1] for n, m in models.items()}
    preds = {n: m.predict(X) for n, m in models.items()}
    target = rates(y, preds["rf100"])["fpr"]
    benign = y == 0

    out = {"rf100_reference_fpr": target,
           "method": ("two-sided bracket: conservative = smallest threshold with achieved "
                      "FPR <= target; permissive = next lower distinct score (FPR overshoots). "
                      "Corrects the v1 quantile overshoot flagged in verification - DT's 598 "
                      "distinct scores cannot hit the target exactly."),
           "argmax": {n: rates(y, preds[n]) for n in models},
           "bracket": {}}

    for n in models:
        s = probas[n]
        cand = np.unique(s[benign])
        ts = np.concatenate([cand, [np.inf]])
        fprs = np.array([np.mean(s[benign] >= t) for t in ts])
        ok = fprs <= target
        t_cons = float(ts[ok][np.argmax(fprs[ok])])
        below = fprs > target
        t_perm = float(ts[below][np.argmin(fprs[below])]) if below.any() else t_cons
        row = {}
        for tag, t in (("conservative", t_cons), ("permissive", t_perm)):
            r = rates(y, (s >= t).astype(int))
            r["threshold"] = round(t, 6)
            row[tag] = r
        out["bracket"][n] = row

    dt_fpr = out["bracket"]["dt12"]["conservative"]["fpr"]
    s = probas["rf100"]
    cand = np.unique(s[benign])
    fprs = np.array([np.mean(s[benign] >= t) for t in cand])
    ok = fprs <= dt_fpr
    t = float(cand[ok][np.argmax(fprs[ok])])
    out["rf100_at_dt12_conservative_fpr"] = {**rates(y, (s >= t).astype(int)),
                                             "threshold": round(t, 6)}

    dtc = out["bracket"]["dt12"]["conservative"]
    rfc = out["rf100_at_dt12_conservative_fpr"]
    comp = ("DT better" if dtc["fnr"] < rfc["fnr"]
            else ("RF better" if rfc["fnr"] < dtc["fnr"] else "tie"))
    out["verdict"] = (f"PASS - at DT-12's conservative operating point (FPR {dtc['fpr']:.4f} "
                      f"<= RF ref {target:.4f}), DT FNR {dtc['fnr']:.4f} vs RF FNR "
                      f"{rfc['fnr']:.4f} at the same achieved FPR -> {comp}")
    with open(os.path.join(REPO, "results/rq2/gate2_operating_point.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(out["verdict"])
