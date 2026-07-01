"""
Golden-Output Feature Snapshot  (Batch 1.5)
===========================================
Dumps the 5-feature production vectors for a FIXED, curated sample of domains to
a committed JSON file. Later refactors that must not change behaviour — above all
the Aho-Corasick / DAWG rewrite of the two O(m^2) dictionary features (Batch 3) —
are regression-checked bit-for-bit against this snapshot.

Serves acceptance IDs A2 (dictionary-feature parity) and A3 (model parity).

The feature VALUES depend on the current dictionary + trigram table (both produced
by preprocess), so the snapshot records their sizes as provenance. A regression is
only meaningful when compared under the same dictionary/trigram artifacts.

Regenerate (only when the sample or resources intentionally change):
    python -m src.golden_snapshot --data-path data/ --out tests/golden_features.json
"""

import os
import json
import argparse
import logging
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Fixed, curated sample. Deliberately spans feature edge cases so a refactor that
# breaks any single extractor shows up:
#   - short (<3 chars, pronounceability guard), all-digits, mixed alnum
#   - real dictionary words / compounds (meaningful_word_ratio, lms)
#   - DGA-style high-entropy gibberish, hyphens, long strings, empty string
SAMPLE_DOMAINS: List[str] = [
    "",                       # empty guard
    "a",                      # length 1 (<3)
    "ab",                     # length 2 (<3)
    "google",                 # single dictionary word
    "googlebot",              # compound of two words
    "microsoftonline",        # long compound
    "read-fx",                # hyphenated, partial word
    "columbia-ca",            # hyphenated place + code
    "12345",                  # all digits
    "abc123",                 # 50% digits
    "shonan-sinkyu",          # real-data benign sample
    "stephensconstruction",   # long real compound
    "ocymmekqogkw",           # real-data DGA gibberish
    "vaqemxhdhekfwu",         # real-data DGA gibberish
    "xkcdqwertzuiop",         # keyboard-ish gibberish
    "facebooklogin",          # compound brand + word
    "the",                    # very common short word
    "supercalifragilistic",   # long, partially meaningful
    "zzzzzzzz",               # repeated char
    "news24",                 # word + digits
]


def build_snapshot(data_path: str) -> Dict[str, Any]:
    """Compute the 5-feature vectors for SAMPLE_DOMAINS with current resources."""
    import numpy as np  # noqa: F401  (imported for determinism note only)
    from src.shared_resources import initialize_shared_resources
    from src.features import extract_features, FEATURE_NAMES_5

    dictionary, ngram_table = initialize_shared_resources(data_path)

    vectors: Dict[str, List[float]] = {}
    for i, domain in enumerate(SAMPLE_DOMAINS):
        prev = SAMPLE_DOMAINS[i - 1] if i > 0 else domain
        feats = extract_features(domain, prev, dictionary, ngram_table,
                                 skip_levenshtein=True)
        # repr() of Python floats round-trips to the exact IEEE-754 double.
        vectors[domain] = [float(x) for x in feats.tolist()]

    return {
        "feature_names": FEATURE_NAMES_5,
        "skip_levenshtein": True,
        "provenance": {
            "n_dictionary_words": len(dictionary),
            "n_trigrams": len(ngram_table),
        },
        "vectors": vectors,
    }


def compare_against(snapshot_path: str, data_path: str,
                    rtol: float = 0.0, atol: float = 0.0) -> Dict[str, Any]:
    """Recompute and compare to a committed snapshot. rtol/atol=0 => bit-identical.

    Returns a dict with 'ok' and a list of mismatches (domain, index, expected, got).
    """
    import numpy as np

    with open(snapshot_path, "r", encoding="utf-8") as f:
        golden = json.load(f)
    fresh = build_snapshot(data_path)

    mismatches = []
    for domain, expected in golden["vectors"].items():
        got = fresh["vectors"].get(domain)
        if got is None:
            mismatches.append((domain, -1, expected, None))
            continue
        for j, (e, g) in enumerate(zip(expected, got)):
            if not np.isclose(e, g, rtol=rtol, atol=atol, equal_nan=True):
                mismatches.append((domain, j, e, g))
    return {"ok": len(mismatches) == 0, "mismatches": mismatches,
            "provenance_match": golden.get("provenance") == fresh.get("provenance")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Golden feature snapshot (Batch 1.5)")
    ap.add_argument("--data-path", default="data/")
    ap.add_argument("--out", default="tests/golden_features.json")
    ap.add_argument("--check", action="store_true",
                    help="Compare against an existing snapshot instead of writing.")
    args = ap.parse_args()

    if args.check:
        res = compare_against(args.out, args.data_path)
        print(json.dumps({"ok": res["ok"],
                          "provenance_match": res["provenance_match"],
                          "n_mismatches": len(res["mismatches"])}, indent=2))
        raise SystemExit(0 if res["ok"] else 1)

    snap = build_snapshot(args.data_path)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2)
    logger.info("Wrote golden snapshot: %s (%d domains, dict=%d, trigrams=%d)",
                args.out, len(snap["vectors"]),
                snap["provenance"]["n_dictionary_words"],
                snap["provenance"]["n_trigrams"])
