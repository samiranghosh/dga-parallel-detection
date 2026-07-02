"""B5 Step 0: golden oracle v2 generator -> tests/golden_features_v2.json.

Regenerates the A2 oracle FROM THE LEGACY IMPLEMENTATION (src/features.py as of
branch point eb7758d) before any kernel code is touched. Composition:
  - the 20 v1 curated domains (src/golden_snapshot.SAMPLE_DOMAINS), unchanged;
  - A4 edge cases: empty, single-char, max-length (real + synthetic 63-char),
    digit-only, hyphenated, punycode/IDN (xn--... and raw unicode), unicode
    digits (isdigit() is True for Arabic-Indic numerals), repeated-word
    compounds, zero-dictionary-hit strings, case-sensitivity probes, dotted;
  - a 1,000-domain stratified random sample across train+test x label
    (proportional allocation, seed 42, pandas sample(random_state)).

Crash policy (per brief): if legacy crashes on an edge case, the exception is
RECORDED under "quarantined" and the case carries no vector - legacy semantics
are never "fixed" in this batch.

Dictionary semantics that make this oracle non-trivial (measured, nltk words):
all 26 single letters are dictionary words -> meaningful_word_ratio degenerates
to alpha-coverage under THIS dictionary, and lms >= 1/n for any string with a
letter. Matching is case-sensitive (dictionary is lowercased; no folding in the
feature code). The kernel must reproduce these outcomes via the same rules, not
special-case them.

Run:  python scripts/kernel/make_golden_v2.py            (writes the JSON)
      python scripts/kernel/make_golden_v2.py --check    (recompute + compare)
"""
import os
import sys
import json
import argparse

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

SEED = 42
SAMPLE_N = 1000
OUT_DEFAULT = os.path.join(REPO, "tests", "golden_features_v2.json")

EDGE_CASES = [
    # empty / single char
    "", "a", "q", "7", "-", ".",
    # synthetic max-length (63 = DNS label max; dataset max is also 63)
    "a" * 63,
    "9" * 63,
    # digit-only
    "0123456789",
    # hyphenated
    "-leading", "trailing-", "a-b-c-d", "co-operate-now",
    # punycode / internationalised
    "xn--mnchen-3ya", "xn--fiqs8s", "xn--80ak6aa92e",
    "münchen", "bücher",
    # unicode digits: '٣'.isdigit() is True -> numerical_ratio counts them
    "٣٤٥",
    # repeated-word compounds
    "paypalpaypal", "googlegoogle", "thethethe", "catcatcat",
    # zero dictionary hits (must contain NO a-z: single letters are all words)
    "345", "0-0-0", "---", "12-34",
    # case sensitivity (dictionary is lowercase; no folding in legacy)
    "GOOGLE", "PayPal",
    # dotted FQDN-style input (preprocessing normally strips this)
    "www.google.com",
]


def stratified_sample(n: int, seed: int):
    """n domains across train+test x label, proportional, deterministic."""
    import pandas as pd
    frames = []
    for split in ("train", "test"):
        df = pd.read_csv(os.path.join(REPO, "data", f"{split}.csv"))
        df["domain"] = df["domain"].astype(str)
        df["split"] = split
        frames.append(df[["domain", "label", "split"]])
    alldf = pd.concat(frames, ignore_index=True)
    total = len(alldf)
    picked = []
    groups = sorted(alldf.groupby(["split", "label"]).groups.keys())
    for key in groups:
        g = alldf[(alldf["split"] == key[0]) & (alldf["label"] == key[1])]
        k = round(n * len(g) / total)
        picked.append(g.sample(n=k, random_state=seed))
    out = pd.concat(picked, ignore_index=True)
    # top up / trim rounding drift deterministically
    if len(out) < n:
        rest = alldf.drop(out.index, errors="ignore")
        out = pd.concat([out, rest.sample(n=n - len(out), random_state=seed)],
                        ignore_index=True)
    out = out.iloc[:n]
    return [(r.domain, f"sample:{r.split}:{int(r.label)}")
            for r in out.itertuples()]


def real_max_length_domain():
    import pandas as pd
    d = pd.concat([pd.read_csv(os.path.join(REPO, "data", f"{s}.csv"))
                   for s in ("train", "test")])["domain"].astype(str)
    return d.loc[d.str.len().idxmax()]


def build(data_path: str):
    from src.shared_resources import initialize_shared_resources
    from src.features import extract_features, FEATURE_NAMES_5
    from src.golden_snapshot import SAMPLE_DOMAINS

    dictionary, ngram_table = initialize_shared_resources(data_path)

    cases = []  # (domain, source) - insertion order, dedupe keep-first
    for d in SAMPLE_DOMAINS:
        cases.append((d, "v1"))
    for d in EDGE_CASES:
        cases.append((d, "edge"))
    cases.append((real_max_length_domain(), "edge:real-max-length"))
    cases.extend(stratified_sample(SAMPLE_N, SEED))

    vectors, sources, quarantined = {}, {}, {}
    for domain, src in cases:
        if domain in vectors or domain in quarantined:
            continue
        try:
            feats = extract_features(domain, domain, dictionary, ngram_table,
                                     skip_levenshtein=True)
            vectors[domain] = [float(x) for x in feats.tolist()]
            sources[domain] = src
        except Exception as e:  # record, quarantine, never fix legacy here
            quarantined[domain] = f"{type(e).__name__}: {e}"

    return {
        "version": 2,
        "feature_names": FEATURE_NAMES_5,
        "skip_levenshtein": True,
        "provenance": {
            "generator": "scripts/kernel/make_golden_v2.py",
            "oracle": "legacy src/features.py at feature-kernel branch point (eb7758d)",
            "n_dictionary_words": len(dictionary),
            "n_trigrams": len(ngram_table),
            "seed": SEED,
            "sample_spec": f"{SAMPLE_N} stratified across train+test x label, "
                           f"proportional, pandas random_state={SEED}",
        },
        "quarantined": quarantined,
        "sources": sources,
        "vectors": vectors,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default=os.path.join(REPO, "data"))
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    snap = build(args.data_path)
    if args.check:
        with open(args.out, "r", encoding="utf-8") as f:
            golden = json.load(f)
        bad = []
        for dom, exp in golden["vectors"].items():
            got = snap["vectors"].get(dom)
            if got is None or any(abs(e - g) > 1e-12 for e, g in zip(exp, got)):
                bad.append(dom)
        print(json.dumps({"ok": not bad, "n": len(golden["vectors"]),
                          "mismatches": bad[:10]}))
        raise SystemExit(0 if not bad else 1)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2, ensure_ascii=True)
    print(f"wrote {args.out}: {len(snap['vectors'])} vectors, "
          f"{len(snap['quarantined'])} quarantined")
    for k, v in snap["quarantined"].items():
        print("  QUARANTINED", ascii(k), "->", v)
