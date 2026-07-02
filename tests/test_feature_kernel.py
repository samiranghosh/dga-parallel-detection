"""A2 equivalence gate (B5 Step 3; extended B6 Step 0): all FEATURE_KERNEL
modes (legacy / fast / fast_marisa) bit-identical, raw set or DictKernel.

Three layers:
  1. golden oracle v2 (1,049 domains incl. A4 edge cases) under ALL kernels;
  2. legacy == fast == fast_marisa on a fresh random 10k train+test sample
     (seed 43 - a different draw than the Step-1 bench, seed 42);
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
@pytest.mark.parametrize("mode", ["legacy", "fast", "fast_marisa"])
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
@pytest.mark.parametrize("mode", ["fast", "fast_marisa"])
def test_legacy_equals_fast_random_10k(mode, resources):
    """Fresh 10k random sample: the two dictionary features agree bitwise."""
    import pandas as pd
    dictionary, _ = resources
    frames = [pd.read_csv(os.path.join(DATA, f"{s}.csv")) for s in ("train", "test")]
    domains = (pd.concat(frames, ignore_index=True)["domain"].astype(str)
               .sample(n=10_000, random_state=43).tolist())

    set_kernel_mode("legacy")
    ref = [(calc_meaningful_word_ratio(d, dictionary),
            calc_lms_percentage(d, dictionary)) for d in domains]
    set_kernel_mode(mode)
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


@pytest.mark.parametrize("mode", ["fast", "fast_marisa"])
@pytest.mark.parametrize("dictionary", [
    {"cat", "dog", "cats", "at", "a"},
    {"ab", "b", "aba"},
    {"the"},
    set(),
])
def test_rule_corners_small_dicts(dictionary, mode):
    """legacy == fast kernels on corner strings for several tiny dictionaries."""
    for domain in RULE_CORNERS:
        set_kernel_mode("legacy")
        ref = (calc_meaningful_word_ratio(domain, dictionary),
               calc_lms_percentage(domain, dictionary))
        set_kernel_mode(mode)
        got = (calc_meaningful_word_ratio(domain, dictionary),
               calc_lms_percentage(domain, dictionary))
        assert ref == got, f"{domain!r} with {sorted(dictionary)}: {ref} != {got}"


# ── B6 Step 0: DictKernel handle (set released) + serving-resources path ──

@needs_data
@pytest.mark.parametrize("mode", ["fast", "fast_marisa"])
def test_dict_kernel_handle_parity(mode, resources, golden):
    """A DictKernel built from the set reproduces golden v2 exactly - the
    handle path (Step 0a: set releasable) computes identical values."""
    from src.features import build_kernel
    dictionary, ngram_table = resources
    set_kernel_mode(mode)
    kernel = build_kernel(dictionary, mode=mode)
    bad = []
    for domain, expected in golden["vectors"].items():
        got = extract_features(domain, domain, kernel, ngram_table,
                               skip_levenshtein=True)
        for j, (e, g) in enumerate(zip(expected, got.tolist())):
            if e != g:
                bad.append((domain, golden["feature_names"][j], e, g))
    assert not bad, f"{len(bad)} mismatches via DictKernel/{mode}: {bad[:5]}"


@needs_data
def test_marisa_mmap_matches_set_build(resources, golden):
    """The mmapped data/english_dictionary.marisa (the serving artifact)
    agrees with a trie built from the in-memory set on golden v2."""
    from src.features import build_kernel
    marisa_path = os.path.join(DATA, "english_dictionary.marisa")
    if not os.path.exists(marisa_path):
        pytest.skip("run `python main.py --mode preprocess` first")
    dictionary, ngram_table = resources
    set_kernel_mode("fast_marisa")
    mmap_kernel = build_kernel(marisa_path=marisa_path)
    assert len(mmap_kernel) == len(dictionary), \
        "stale english_dictionary.marisa: word count differs from the set"
    for domain, expected in golden["vectors"].items():
        got = extract_features(domain, domain, mmap_kernel, ngram_table,
                               skip_levenshtein=True)
        assert got.tolist() == expected, f"mmap trie diverges on {domain!r}"


@needs_data
def test_serving_resources_returns_kernel(resources):
    """initialize_serving_resources hands back a DictKernel (not the set)
    under the fast modes, and its output matches the set path."""
    from src.features import DictKernel, build_kernel  # noqa: F401
    from src.shared_resources import initialize_serving_resources
    dictionary, ngram_table = resources
    domains = ["facebooklogin", "ocymmekqogkw", "news24", "", "a" * 63]
    for mode in ("fast", "fast_marisa"):
        set_kernel_mode(mode)
        served, _ = initialize_serving_resources(DATA)
        assert isinstance(served, DictKernel) and served.mode == mode
        for d in domains:
            ref = extract_features(d, d, dictionary, ngram_table,
                                   skip_levenshtein=True)
            got = extract_features(d, d, served, ngram_table,
                                   skip_levenshtein=True)
            assert np.array_equal(ref, got), f"{mode}/{d!r}"


def test_dict_kernel_mode_mismatch_raises():
    """Using a handle under a different active mode is a hard error, not a
    silent wrong-kernel measurement."""
    from src.features import build_kernel
    kernel = build_kernel({"cat", "dog"}, mode="fast")
    set_kernel_mode("fast_marisa")
    with pytest.raises(ValueError, match="DictKernel built for"):
        calc_lms_percentage("catdog", kernel)


@needs_data
def test_shm_path_with_automaton_block(resources):
    """B5 Step 4: shm initializer path (one shared automaton copy) ==
    sequential output; proves the extended _init_worker_shm API end-to-end."""
    import pandas as pd
    from src import features
    from src.shared_resources import SharedMemoryResources
    from src.parallel_engine import parallel_extract_features
    from src.features import extract_all_sequential

    dictionary, ngram_table = resources
    domains = (pd.read_csv(os.path.join(DATA, "test.csv"))["domain"]
               .astype(str).head(200).tolist())

    set_kernel_mode("fast")
    shm = SharedMemoryResources()
    try:
        names = shm.create(dictionary, ngram_table,
                           automaton_blob=features.export_automaton_blob(dictionary))
        assert "automaton_name" in names
        par = parallel_extract_features(domains, 2, dictionary, ngram_table,
                                        skip_levenshtein=True,
                                        use_shared_memory=True, shm_names=names)
    finally:
        shm.cleanup()
    seq = extract_all_sequential(domains, dictionary, ngram_table,
                                 skip_levenshtein=True)
    assert np.array_equal(par, seq)


@needs_data
@pytest.mark.parametrize("mode", ["fast", "fast_marisa"])
def test_extract_features_vector_parity(mode, resources):
    """Full 5-feature vectors identical across kernels (dtype and values)."""
    dictionary, ngram_table = resources
    domains = ["facebooklogin", "ocymmekqogkw", "news24", "", "a" * 63]
    set_kernel_mode("legacy")
    ref = [extract_features(d, d, dictionary, ngram_table, skip_levenshtein=True)
           for d in domains]
    set_kernel_mode(mode)
    got = [extract_features(d, d, dictionary, ngram_table, skip_levenshtein=True)
           for d in domains]
    for d, r, g in zip(domains, ref, got):
        assert r.dtype == g.dtype == np.float64
        assert np.array_equal(r, g), f"{d!r}: {r} != {g}"
