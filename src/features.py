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
    # TODO: Implement
    raise NotImplementedError


# ── Feature 2: Numerical Character Percentage ──

def calc_numerical_ratio(domain: str) -> float:
    """Return the ratio of digit characters to total length.

    Example: 'abc123' -> 0.5
    """
    # TODO: Implement
    raise NotImplementedError


# ── Feature 3: Meaningful Word Ratio ──

def calc_meaningful_word_ratio(domain: str, dictionary: set) -> float:
    """Return the fraction of the domain covered by English dictionary words.

    Uses greedy longest-match scanning across all substrings.
    Example: 'googlebot' with dictionary {'google', 'bot'} -> 1.0
    """
    # TODO: Implement
    raise NotImplementedError


# ── Feature 4: Pronounceability Score ──

def calc_pronounceability(domain: str, ngram_table: dict) -> float:
    """Return the average trigram probability across all character trigrams.

    Higher score = more pronounceable = more likely benign.
    """
    # TODO: Implement
    raise NotImplementedError


# ── Feature 5: Longest Meaningful Substring (LMS) Percentage ──

def calc_lms_percentage(domain: str, dictionary: set) -> float:
    """Return len(longest meaningful substring) / len(domain).

    Scans all substrings of domain against the dictionary.
    """
    # TODO: Implement
    raise NotImplementedError


# ── Feature 6: Levenshtein Edit Distance ──

def calc_levenshtein(domain: str, prev_domain: str) -> float:
    """Return the Levenshtein edit distance between domain and prev_domain.

    Uses dynamic programming. O(m × n) where m, n are string lengths.
    If python-Levenshtein is installed, uses the C extension for speed.
    """
    # TODO: Implement
    # Try fast C version first, fall back to pure Python DP
    # try:
    #     from Levenshtein import distance
    #     return float(distance(domain, prev_domain))
    # except ImportError:
    #     pass  # Pure Python fallback below
    raise NotImplementedError


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
