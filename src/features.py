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
"""

import numpy as np


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
                     dictionary: set, ngram_table: dict) -> np.ndarray:
    """Extract all 6 linguistic features for a single domain.

    Returns:
        np.ndarray of shape (6,) with dtype float64.
    """
    return np.array([
        calc_length(domain),
        calc_numerical_ratio(domain),
        calc_meaningful_word_ratio(domain, dictionary),
        calc_pronounceability(domain, ngram_table),
        calc_lms_percentage(domain, dictionary),
        calc_levenshtein(domain, prev_domain),
    ], dtype=np.float64)


def extract_all_sequential(domain_list: list, dictionary: set,
                           ngram_table: dict) -> np.ndarray:
    """Extract features for all domains sequentially (baseline).

    Returns:
        np.ndarray of shape (N, 6).
    """
    N = len(domain_list)
    features = np.zeros((N, 6), dtype=np.float64)
    for i, domain in enumerate(domain_list):
        prev = domain_list[i - 1] if i > 0 else domain
        features[i] = extract_features(domain, prev, dictionary, ngram_table)
    return features
