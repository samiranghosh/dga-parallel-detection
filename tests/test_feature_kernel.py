"""A2 equivalence gate (B5 Step 3): FEATURE_KERNEL=fast == legacy, bit-identical.

Three layers:
  1. golden oracle v2 (1,049 domains incl. A4 edge cases) under BOTH kernels;
  2. legacy == fast on a fresh random 10k train+test sample (seed 43 - a
     different draw than the Step-1 bench, seed 42);
  3. hand-built small dictionaries exercising rule corners (overlaps,
     abutting words, case sensitivity, unicode, empty).

Equality asserted with == (bitwise), not a tolerance: both kernels produce
int/int quotients from identical integers (rule R5 in src/features.py).
The brief's <=1e-12 ceiling is therefore trivially met; we hold the stricter
line and any nonzero delta fails.
"""
import os
import json

import numpy as np
import pytest

from src.features import (
    calc_meaningful_word_ratio,
    calc_lms_percentage,
    extract_features,
    get_kernel_mode,
    set_kernel_mode,
)
from src.shared_resources import initialize_shared_resources

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
GOLDEN_V2 = os.path.join(REPO, "tests", "golden_features_v2.json")

needs_data = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA, "english_dictionary.txt")),
    reason="run `python main.py --mode preprocess` first",
)


@pytest.fixture(autouse=True)
def _restore_kernel_mode():
    prev = get_kernel_mode()
    yield
    set_kernel_mode(prev)


@pytest.fixture(scope="module")
def resources():
    if not os.path.exists(os.path.join(DATA, "english_dictionary.txt")):
        pytest.skip("preprocess artifacts missing")
    return initialize_shared_resources(DATA)


@pytest.fixture(scope="module")
def golden():
    with open(GOLDEN_V2, "r", encoding="utf-8") as f:
        return json.load(f)


@needs_data
@pytest.mark.parametrize("mode", ["legacy", "fast"])
def test_golden_v2(mode, resources, golden):
    """Both kernels reproduce the committed oracle exactly."""
    dictionary, ngram_table = resources
    set_kernel_mode(mode)
    bad = []
    for domain, expected in golden["vectors"].items():
        got = extract_features(domain, domain, dictionary, ngram_table,
                               skip_levenshtein=True)
        for j, (e, g) in enumerate(zip(expected, got.tolist())):
            if e != g:
                bad.append((domain, golden["feature_names"][j], e, g))
    assert not bad, f"{len(bad)} mismatches vs golden v2 under {mode}: {bad[:5]}"


@needs_data
def test_legacy_equals_fast_random_10k(resources):
    """Fresh 10k random sample: the two dictionary features agree bitwise."""
    import pandas as pd
    dictionary, _ = resources
    frames = [pd.read_csv(os.path.join(DATA, f"{s}.csv")) for s in ("train", "test")]
    domains = (pd.concat(frames, ignore_index=True)["domain"].astype(str)
               .sample(n=10_000, random_state=43).tolist())

    set_kernel_mode("legacy")
    ref = [(calc_meaningful_word_ratio(d, dictionary),
            calc_lms_percentage(d, dictionary)) for d in domains]
    set_kernel_mode("fast")
    got = [(calc_meaningful_word_ratio(d, dictionary),
            calc_lms_percentage(d, dictionary)) for d in domains]

    mismatches = [(d, r, g) for d, r, g in zip(domains, ref, got) if r != g]
    assert not mismatches, f"{len(mismatches)} mismatches: {mismatches[:5]}"


RULE_CORNERS = [
    "", "a", "ab", "aa", "abab",          # empty / single / repeated
    "catdog", "cat-dog", "catsdogs",      # abutting words, separator
    "dogcatdog", "acatb",                 # overlap / single-letter fill
    "CATDOG", "CatDog",                   # case sensitivity (no folding)
    "münchen", "xn--mnchen-3ya",          # unicode + punycode
    "0-0-0", "123", "catcatcat",          # zero-hit + repeated compound
]


@pytest.mark.parametrize("dictionary", [
    {"cat", "dog", "cats", "at", "a"},
    {"ab", "b", "aba"},
    {"the"},
    set(),
])
def test_rule_corners_small_dicts(dictionary):
    """legacy == fast on corner strings for several tiny dictionaries."""
    for domain in RULE_CORNERS:
        set_kernel_mode("legacy")
        ref = (calc_meaningful_word_ratio(domain, dictionary),
               calc_lms_percentage(domain, dictionary))
        set_kernel_mode("fast")
        got = (calc_meaningful_word_ratio(domain, dictionary),
               calc_lms_percentage(domain, dictionary))
        assert ref == got, f"{domain!r} with {sorted(dictionary)}: {ref} != {got}"


@needs_data
def test_extract_features_vector_parity(resources):
    """Full 5-feature vectors identical across kernels (dtype and values)."""
    dictionary, ngram_table = resources
    domains = ["facebooklogin", "ocymmekqogkw", "news24", "", "a" * 63]
    set_kernel_mode("legacy")
    ref = [extract_features(d, d, dictionary, ngram_table, skip_levenshtein=True)
           for d in domains]
    set_kernel_mode("fast")
    got = [extract_features(d, d, dictionary, ngram_table, skip_levenshtein=True)
           for d in domains]
    for d, r, g in zip(domains, ref, got):
        assert r.dtype == g.dtype == np.float64
        assert np.array_equal(r, g), f"{d!r}: {r} != {g}"
