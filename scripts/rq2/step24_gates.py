"""B4 Steps 2-4: the four DT-12 promotion gates.

Gate 1  McNemar DT-12 vs RF-100, full ExtraHop holdout (exact binomial).
Gate 2  FPR/FNR at argmax for all four models + matched-FPR operating point
        (each threshold tuned to RF-100's FPR; compare FNR). Confusions.
Gate 3  ExtraHop has NO family labels -> per-family analysis runs on chrmor
        (folded into Gate 4, as the brief's contingency prescribes).
Gate 4  Train ExtraHop -> test chrmor (no fine-tuning). Accuracy/F1/FPR/FNR
        + per-family recall; overlap-with-training reported and also excluded.

Writes results/rq2/gate1_mcnemar.json, gate2_operating_point.json,
gate34_chrmor.json.
"""
import sys, os, json, time, pickle
sys.path.insert(0, r"c:/Users/samir/source/repos/dga-parallel-detection")
import numpy as np
import pandas as pd

ART = r"C:/Users/samir/AppData/Local/Temp/claude/c--Users-samir-source-repos-dga-parallel-detection/932315aa-8a56-47d8-95fa-59c19e89893b/scratchpad/artifacts"
REPO = r"c:/Users/samir/source/repos/dga-parallel-detection"
RQ2 = os.path.join(REPO, "results/rq2")
CHRMOR = os.path.join(REPO, "data/DGA_domains_dataset/dga_domains_full.csv")

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.stats import binomtest

def load_models():
    with open(os.path.join(ART, "rf100_5feat.pkl"), "rb") as f:
        rf100 = pickle.load(f)
    models = {"rf100": rf100}
    for name in ("rf_pruned", "dt12", "logreg"):
        with open(os.path.join(RQ2, "models", f"{name}.pkl"), "rb") as f:
            models[name] = pickle.load(f)
    return models

def rates(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "fpr": round(fp / (fp + tn), 6), "fnr": round(fn / (fn + tp), 6),
            "accuracy_pct": round(accuracy_score(y, p) * 100, 4),
            "f1": round(f1_score(y, p), 4)}

if __name__ == "__main__":
    models = load_models()
    X_test = np.load(os.path.join(ART, "X_test5.npy"))
    y_test = np.load(os.path.join(ART, "y_test.npy"))

    preds = {n: m.predict(X_test) for n, m in models.items()}
    probas = {n: m.predict_proba(X_test)[:, 1] for n, m in models.items()}

    # ---------------- Gate 1: McNemar (exact binomial) ----------------
    rf_ok = preds["rf100"] == y_test
    dt_ok = preds["dt12"] == y_test
    b = int((~dt_ok & rf_ok).sum())   # DT wrong, RF right
    c = int((dt_ok & ~rf_ok).sum())   # DT right, RF wrong
    test = binomtest(min(b, c), b + c, 0.5)  # two-sided exact
    p_exact = float(test.pvalue)
    gate1 = {
        "n_holdout": int(len(y_test)),
        "dt_wrong_rf_right_b": b, "dt_right_rf_wrong_c": c,
        "discordant_pairs": b + c, "exact_binomial_p_two_sided": p_exact,
        "direction": "DT better" if c > b else ("RF better" if b > c else "tie"),
        "acc_dt12_pct": round(float(dt_ok.mean()) * 100, 4),
        "acc_rf100_pct": round(float(rf_ok.mean()) * 100, 4),
        "verdict": ("BEATS (significant, DT favour)" if p_exact < 0.05 and c > b
                    else ("WORSE (significant, RF favour)" if p_exact < 0.05 and b > c
                          else "NOT MEANINGFULLY WORSE (non-significant)")),
    }
    with open(os.path.join(RQ2, "gate1_mcnemar.json"), "w") as f:
        json.dump(gate1, f, indent=2)
    print(f"[gate1] b={b} c={c} p={p_exact:.3e} -> {gate1['verdict']}", flush=True)

    # ---------------- Gate 2: operating point ----------------
    rf_fpr = rates(y_test, preds["rf100"])["fpr"]
    gate2 = {"rf100_reference_fpr": rf_fpr, "argmax": {}, "matched_fpr": {}}
    benign_mask = y_test == 0
    for n in models:
        gate2["argmax"][n] = rates(y_test, preds[n])
        # threshold s.t. FPR == RF-100's: (1 - rf_fpr) quantile of benign scores
        t = float(np.quantile(probas[n][benign_mask], 1 - rf_fpr))
        p_at = (probas[n] >= t).astype(int)
        r = rates(y_test, p_at)
        r["threshold"] = round(t, 6)
        r["distinct_scores"] = int(len(np.unique(np.round(probas[n], 8))))
        gate2["matched_fpr"][n] = r
    with open(os.path.join(RQ2, "gate2_operating_point.json"), "w") as f:
        json.dump(gate2, f, indent=2)
    for n in models:
        a, m = gate2["argmax"][n], gate2["matched_fpr"][n]
        print(f"[gate2] {n:9s} argmax FPR={a['fpr']:.4f} FNR={a['fnr']:.4f} | "
              f"matched-FPR thr={m['threshold']:.4f} FNR={m['fnr']:.4f} "
              f"(distinct scores={m['distinct_scores']})", flush=True)

    # ---------------- Gates 3+4: chrmor cross-dataset ----------------
    print("[gate34] loading + cleaning chrmor (ExtraHop-identical pipeline)...", flush=True)
    ch = pd.read_csv(CHRMOR, header=None, names=["cls", "family", "host"])
    from src.preprocess import strip_tld
    ch["domain"] = ch["host"].astype(str).str.lower().apply(strip_tld)
    ch = ch[ch["domain"] != ""].drop_duplicates(subset=["domain"])
    ch["label"] = (ch["cls"] == "dga").astype(int)
    ch = ch.sort_values("domain").reset_index(drop=True)
    print(f"[gate34] cleaned: {len(ch)} rows, {ch['label'].mean()*100:.1f}% dga, "
          f"{ch[ch.label==1]['family'].nunique()} families", flush=True)

    # overlap with ExtraHop TRAINING domains (leakage check)
    train_domains = set(pd.read_csv(os.path.join(REPO, "data/train.csv"))["domain"].astype(str))
    ch["in_train"] = ch["domain"].isin(train_domains)
    print(f"[gate34] overlap with ExtraHop train: {int(ch['in_train'].sum())} "
          f"({ch['in_train'].mean()*100:.2f}%)", flush=True)

    from src.shared_resources import initialize_shared_resources
    from src.parallel_engine import parallel_extract_features
    dictionary, ngram = initialize_shared_resources(os.path.join(REPO, "data/"))
    t0 = time.time()
    Xc = parallel_extract_features(ch["domain"].tolist(), 8, dictionary, ngram,
                                   skip_levenshtein=True)
    print(f"[gate34] features {Xc.shape} [{time.time()-t0:.0f}s]", flush=True)
    yc = ch["label"].values

    gate34 = {"provenance": {
        "url": "https://github.com/chrmor/DGA_domains_dataset",
        "commit": "9dcc29e5cc644fdfd99d82a3abec993c15bbc7bc",
        "licence": "free for research purposes; cite Cucchiarelli et al. 2021, ESWA 170, doi:10.1016/j.eswa.2020.114551",
        "rows_cleaned": int(len(ch)),
        "overlap_with_train": int(ch["in_train"].sum()),
        "preprocessing": "identical to ExtraHop pipeline: lowercase -> tldextract strip_tld (subdomain kept) -> dedupe -> sort",
    }, "overall": {}, "overall_excl_overlap": {}, "per_family_recall": {}}

    keep = ~ch["in_train"].values
    for n, m in models.items():
        pc = m.predict(Xc)
        gate34["overall"][n] = rates(yc, pc)
        gate34["overall_excl_overlap"][n] = rates(yc[keep], pc[keep])
        if n in ("rf100", "rf_pruned", "dt12"):
            fam_recall = {}
            dga_mask = yc == 1
            fams = ch["family"].values
            for fam in sorted(ch[ch.label == 1]["family"].unique()):
                fm = dga_mask & (fams == fam)
                fam_recall[fam] = round(float(pc[fm].mean()), 4)  # recall = P(pred=1|family)
            gate34["per_family_recall"][n] = fam_recall
        print(f"[gate34] {n:9s} acc={gate34['overall'][n]['accuracy_pct']}% "
              f"f1={gate34['overall'][n]['f1']} fpr={gate34['overall'][n]['fpr']} "
              f"fnr={gate34['overall'][n]['fnr']}", flush=True)

    # disproportionality: family-level DT drop vs RF
    fr = gate34["per_family_recall"]
    flags = {f: {"rf100": fr["rf100"][f], "dt12": fr["dt12"][f],
                 "gap_pp": round((fr["rf100"][f] - fr["dt12"][f]) * 100, 2)}
             for f in fr["dt12"] if (fr["rf100"][f] - fr["dt12"][f]) > 0.05}
    gate34["families_dt_drops_gt5pp_vs_rf"] = flags
    with open(os.path.join(RQ2, "gate34_chrmor.json"), "w") as f:
        json.dump(gate34, f, indent=2)
    print(f"[gate34] families where DT drops >5 pp vs RF: {len(flags)}", flush=True)
    print("[gates] DONE", flush=True)
