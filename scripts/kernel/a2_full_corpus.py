"""B5 Step 3b: full-corpus A2 sweep -> results/kernel/a2_full_corpus.json.

Every domain in train.csv + test.csv, the two dictionary features computed
under FEATURE_KERNEL=legacy and =fast; reports max|delta| per feature and
the count of non-bit-identical values (required: 0.0 / 0).
"""
import os
import sys
import json
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

OUT = os.path.join(REPO, "results", "kernel", "a2_full_corpus.json")

if __name__ == "__main__":
    import pandas as pd
    from src.shared_resources import initialize_shared_resources
    from src import features

    dictionary, _ = initialize_shared_resources(os.path.join(REPO, "data"))
    frames = [pd.read_csv(os.path.join(REPO, "data", f"{s}.csv"))
              for s in ("train", "test")]
    domains = pd.concat(frames, ignore_index=True)["domain"].astype(str).tolist()
    print(f"{len(domains)} domains")

    def sweep(mode):
        features.set_kernel_mode(mode)
        t0 = time.perf_counter()
        out = [(features.calc_meaningful_word_ratio(d, dictionary),
                features.calc_lms_percentage(d, dictionary))
               for d in domains]
        return out, time.perf_counter() - t0

    ref, t_legacy = sweep("legacy")
    print(f"legacy sweep {t_legacy:.1f}s")
    got, t_fast = sweep("fast")
    print(f"fast sweep {t_fast:.1f}s")

    max_d_mwr = max_d_lms = 0.0
    n_diff_mwr = n_diff_lms = 0
    for (rm, rl), (gm, gl) in zip(ref, got):
        if rm != gm:
            n_diff_mwr += 1
            max_d_mwr = max(max_d_mwr, abs(rm - gm))
        if rl != gl:
            n_diff_lms += 1
            max_d_lms = max(max_d_lms, abs(rl - gl))

    result = {
        "n_domains": len(domains),
        "corpus": "train.csv + test.csv (full)",
        "max_abs_delta": {"meaningful_word_ratio": max_d_mwr,
                          "lms_percentage": max_d_lms},
        "n_not_bit_identical": {"meaningful_word_ratio": n_diff_mwr,
                                "lms_percentage": n_diff_lms},
        "sweep_seconds": {"legacy": round(t_legacy, 1), "fast": round(t_fast, 1)},
        "verdict": ("PASS - bit-identical on the full corpus"
                    if n_diff_mwr == n_diff_lms == 0 else "FAIL"),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
