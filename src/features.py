"""
Linguistic Feature Extraction Module
=====================================
Owner: Member 2 (Feature Extraction)

Implements the 6 linguistic features from the base paper (Li et al., 2019):
  1. Length                    - O(1)
  2. Numerical Character %     - O(m)
  3. Meaningful Word Ratio     - O(m²)
  4. Pronounceability Score    - O(m)
  5. LMS Percentage            - O(m² × D)
  6. Levenshtein Edit Distance - O(m²)

Each function takes a domain string and returns a float.

Feature subsets:
  FEATURE_NAMES_6: All 6 features (original configuration)
  FEATURE_NAMES_5: Without Levenshtein (optimal configuration per E7 ablation)
"""

import os

import numpy as np


# ── Feature kernel switch (Batch 5) ──
#
# FEATURE_KERNEL selects the implementation of the two dictionary features
# (meaningful_word_ratio, lms_percentage):
#   'legacy' - original O(m^2) substring scans against the set
#   'fast'   - Aho-Corasick kernel (pyahocorasick, C automaton): one pass per
#              domain enumerates every dictionary-word occurrence, then an
#              O(m + matches) DP reproduces the legacy values exactly.
# Default 'fast' - flipped after the Step-3 A2 gate passed: golden v2
# (1,049 domains incl. A4 edge cases) bit-identical under both kernels, and
# the full train+test corpus (999,927 domains) swept with max|delta| = 0.0
# on both features (results/kernel/a2_full_corpus.json). 'legacy' stays
# selectable for A/B and rollback via the environment variable or
# set_kernel_mode().

_kernel_mode = os.environ.get("FEATURE_KERNEL", "fast").strip().lower()
if _kernel_mode not in ("legacy", "fast"):
    raise ValueError(f"FEATURE_KERNEL must be 'legacy' or 'fast', got {_kernel_mode!r}")


def get_kernel_mode() -> str:
    return _kernel_mode


def set_kernel_mode(mode: str):
    """Switch feature-kernel implementation at runtime (tests / A-B).

    Process-local: spawned Pool workers re-import this module and read
    FEATURE_KERNEL from the environment, so for parallel A/B runs set
    os.environ["FEATURE_KERNEL"] before creating the pool (children
    inherit it); this setter alone does not reach them.
    """
    global _kernel_mode
    if mode not in ("legacy", "fast"):
        raise ValueError(f"kernel mode must be 'legacy' or 'fast', got {mode!r}")
    _kernel_mode = mode


# Automata are built once per dictionary object and memoised. Keyed by
# (id, len): the id alone could be reused after a dictionary is garbage
# collected (sets are not weakref-able); len makes a stale hit implausible.
# Production processes hold exactly one dictionary for their lifetime.
_AUTOMATON_CACHE = {}
_AUTOMATON_CACHE_MAX = 8  # small test dictionaries; production uses one entry


def warm_kernel(dictionary, automaton=None):
    """Pre-build or adopt the AC automaton for `dictionary` (no-op in legacy).

    Called from Pool initializers (B5 Step 4) so the build/attach cost lands
    in pool startup, not in the first chunk. When `automaton` is given (a
    deserialized copy shipped by the parent - 0.06 s attach vs 0.50 s
    per-worker rebuild, results/kernel/attach_paths.json), it is adopted
    into the cache for this dictionary object.
    """
    if _kernel_mode != "fast":
        return
    if automaton is not None:
        key = (id(dictionary), len(dictionary))
        if len(_AUTOMATON_CACHE) >= _AUTOMATON_CACHE_MAX:
            _AUTOMATON_CACHE.pop(next(iter(_AUTOMATON_CACHE)))
        _AUTOMATON_CACHE[key] = automaton
    else:
        _get_automaton(dictionary)


def export_automaton_blob(dictionary) -> bytes:
    """Serialize the automaton for `dictionary` once, for worker attach."""
    import pickle
    return pickle.dumps(_get_automaton(dictionary), pickle.HIGHEST_PROTOCOL)


def _get_automaton(dictionary):
    if len(dictionary) == 0:
        # pyahocorasick cannot finalise a zero-word automaton; legacy
        # semantics for an empty dictionary are simply "no matches".
        return None
    key = (id(dictionary), len(dictionary))
    automaton = _AUTOMATON_CACHE.get(key)
    if automaton is None:
        import ahocorasick
        automaton = ahocorasick.Automaton()
        for word in dictionary:
            automaton.add_word(word, len(word))
        automaton.make_automaton()
        if len(_AUTOMATON_CACHE) >= _AUTOMATON_CACHE_MAX:
            _AUTOMATON_CACHE.pop(next(iter(_AUTOMATON_CACHE)))
        _AUTOMATON_CACHE[key] = automaton
    return automaton


# Semantic rules extracted from the legacy implementations (the parity
# contract; each is replicated, not reinterpreted):
#   R1 Candidate matches = every substring domain[i:j] that is a dictionary
#      member: every occurrence of every dictionary word, case-sensitive,
#      no normalisation, no minimum length (the nltk dictionary contains all
#      26 single letters, so mwr degenerates to alpha-coverage under it).
#      Aho-Corasick's all-occurrence iteration yields exactly this set.
#   R2 lms_percentage = len(longest match) / len(domain); 0.0 when the domain
#      is empty or has no match.
#   R3 meaningful_word_ratio = (max characters coverable by NON-overlapping
#      matches) / len(domain); abutting matches allowed, overlaps not.
#      The legacy scan is a weighted-interval DP over match end positions:
#      covered[j] = max(covered[j-1], max over matches (i,j) of
#      covered[i] + (j-i)); its start-indexed carry-forward form is
#      equivalent (covered[i] is final before it propagates to i+1).
#   R4 Denominator = raw len(domain), non-alpha characters included.
#   R5 Both values are exact int/int quotients; identical integer numerators
#      and denominators make the IEEE-754 result bit-identical to legacy.
def _dict_features_fast(domain: str, automaton) -> tuple:
    """(meaningful_word_ratio, lms_percentage) in ONE automaton pass."""
    n = len(domain)
    if n == 0 or automaton is None:
        return 0.0, 0.0
    max_len = 0
    by_end = [None] * (n + 1)  # match lengths bucketed by end position
    for end_idx, length in automaton.iter(domain):
        if length > max_len:
            max_len = length
        j = end_idx + 1
        if by_end[j] is None:
            by_end[j] = [length]
        else:
            by_end[j].append(length)
    if max_len == 0:
        return 0.0, 0.0
    covered = [0] * (n + 1)
    for j in range(1, n + 1):
        best = covered[j - 1]
        lengths = by_end[j]
        if lengths is not None:
            for length in lengths:
                c = covered[j - length] + length
                if c > best:
                    best = c
        covered[j] = best
    return covered[n] / n, max_len / n


# ── Feature Set Configurations ──

FEATURE_NAMES_6 = [
    'length', 'numerical_ratio', 'meaningful_word_ratio',
    'pronounceability', 'lms_percentage', 'levenshtein',
]

FEATURE_NAMES_5 = [
    'length', 'numerical_ratio', 'meaningful_word_ratio',
    'pronounceability', 'lms_percentage',
]


# ── Feature 1: Length ──

def calc_length(domain: str) -> float:
    """Return the character length of the domain string."""
    return float(len(domain))


# ── Feature 2: Numerical Character Percentage ──

def calc_numerical_ratio(domain: str) -> float:
    """Return the ratio of digit characters to total length.

    Example: 'abc123' -> 0.5
    """
    if not domain:
        return 0.0
    return sum(c.isdigit() for c in domain) / len(domain)


# ── Feature 3: Meaningful Word Ratio ──

def calc_meaningful_word_ratio(domain: str, dictionary: set) -> float:
    """Return the fraction of the domain covered by English dictionary words.

    Uses greedy longest-match scanning across all substrings.
    Example: 'googlebot' with dictionary {'google', 'bot'} -> 1.0
    """
    if _kernel_mode == "fast":
        return _dict_features_fast(domain, _get_automaton(dictionary))[0]

    if not domain:
        return 0.0

    n = len(domain)
    # dp[i] = max characters covered from position 0 to i
    covered = [0] * (n + 1)

    for i in range(n):
        # Carry forward previous coverage
        covered[i + 1] = max(covered[i + 1], covered[i])
        # Try all substrings starting at i
        for j in range(i + 1, n + 1):
            substring = domain[i:j]
            if substring in dictionary:
                covered[j] = max(covered[j], covered[i] + len(substring))

    return covered[n] / n


# ── Feature 4: Pronounceability Score ──

def calc_pronounceability(domain: str, ngram_table: dict) -> float:
    """Return the average trigram probability across all character trigrams.

    Higher score = more pronounceable = more likely benign.
    """
    if len(domain) < 3:
        return 0.0

    trigrams = [domain[i:i+3] for i in range(len(domain) - 2)]
    probs = [ngram_table.get(t, 1e-10) for t in trigrams]
    return float(np.mean(probs))


# ── Feature 5: Longest Meaningful Substring (LMS) Percentage ──

def calc_lms_percentage(domain: str, dictionary: set) -> float:
    """Return len(longest meaningful substring) / len(domain).

    Scans all substrings of domain against the dictionary.
    """
    if _kernel_mode == "fast":
        return _dict_features_fast(domain, _get_automaton(dictionary))[1]

    if not domain:
        return 0.0

    n = len(domain)
    max_len = 0
    for i in range(n):
        for j in range(i + 1, n + 1):
            sub = domain[i:j]
            if sub in dictionary:
                max_len = max(max_len, len(sub))

    return max_len / n


# ── Feature 6: Levenshtein Edit Distance ──

def calc_levenshtein(domain: str, prev_domain: str) -> float:
    """Return the Levenshtein edit distance between domain and prev_domain.

    Uses dynamic programming. O(m × n) where m, n are string lengths.
    If python-Levenshtein is installed, uses the C extension for speed.
    """
    # Try fast C version first, fall back to pure Python DP
    try:
        from Levenshtein import distance
        return float(distance(domain, prev_domain))
    except ImportError:
        pass

    # Pure Python DP fallback
    m, n = len(domain), len(prev_domain)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev_row = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if domain[i - 1] == prev_domain[j - 1]:
                dp[j] = prev_row[j - 1]
            else:
                dp[j] = 1 + min(prev_row[j], dp[j - 1], prev_row[j - 1])
    return float(dp[n])


# ── Combined Extraction ──

def extract_features(domain: str, prev_domain: str,
                     dictionary: set, ngram_table: dict,
                     skip_levenshtein: bool = False) -> np.ndarray:
    """Extract linguistic features for a single domain.

    Args:
        domain: Domain string to extract features from.
        prev_domain: Previous domain (for Levenshtein distance).
        dictionary: English dictionary set.
        ngram_table: Trigram frequency table.
        skip_levenshtein: If True, return 5 features (without Levenshtein).
            The 5-feature configuration achieves higher accuracy (93.18%)
            than the 6-feature configuration (92.60%) because Levenshtein
            distance between adjacent domains in shuffled datasets is noise.

    Returns:
        np.ndarray of shape (5,) or (6,) with dtype float64.
    """
    if _kernel_mode == "fast":
        # one automaton pass yields both dictionary features
        mwr, lms = _dict_features_fast(domain, _get_automaton(dictionary))
    else:
        mwr = calc_meaningful_word_ratio(domain, dictionary)
        lms = calc_lms_percentage(domain, dictionary)
    feats = [
        calc_length(domain),
        calc_numerical_ratio(domain),
        mwr,
        calc_pronounceability(domain, ngram_table),
        lms,
    ]
    if not skip_levenshtein:
        feats.append(calc_levenshtein(domain, prev_domain))
    return np.array(feats, dtype=np.float64)


def extract_all_sequential(domain_list: list, dictionary: set,
                           ngram_table: dict,
                           skip_levenshtein: bool = False) -> np.ndarray:
    """Extract features for all domains sequentially (baseline).

    Args:
        domain_list: List of domain strings.
        dictionary: English dictionary set.
        ngram_table: Trigram frequency table.
        skip_levenshtein: If True, extract 5 features only.

    Returns:
        np.ndarray of shape (N, 5) or (N, 6).
    """
    N = len(domain_list)
    n_feats = 5 if skip_levenshtein else 6
    features = np.zeros((N, n_feats), dtype=np.float64)
    for i, domain in enumerate(domain_list):
        prev = domain_list[i - 1] if i > 0 else domain
        features[i] = extract_features(domain, prev, dictionary, ngram_table,
                                       skip_levenshtein=skip_levenshtein)
    return features
