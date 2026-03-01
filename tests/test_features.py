"""
Unit Tests for Feature Extraction
===================================
Tests each of the 6 linguistic features against known expected values.

Run: pytest tests/test_features.py -v
"""

import pytest
import numpy as np
from src.features import (
    calc_length,
    calc_numerical_ratio,
    calc_meaningful_word_ratio,
    calc_pronounceability,
    calc_lms_percentage,
    calc_levenshtein,
    extract_features,
)


# ── Fixtures ──

@pytest.fixture
def dictionary():
    """Minimal English dictionary for testing."""
    return {"google", "bot", "mail", "face", "book", "the", "test", "hello"}


@pytest.fixture
def ngram_table():
    """Minimal trigram frequency table for testing."""
    # TODO: Populate with a few known trigrams and their probabilities
    return {"goo": 0.01, "oog": 0.008, "ogl": 0.005, "gle": 0.012,
            "xyz": 0.0001, "qqq": 0.00001}


# ── Feature 1: Length ──

class TestLength:
    def test_simple(self):
        assert calc_length("google") == 6

    def test_empty(self):
        assert calc_length("") == 0

    def test_long_domain(self):
        assert calc_length("a" * 100) == 100


# ── Feature 2: Numerical Character % ──

class TestNumericalRatio:
    def test_no_digits(self):
        assert calc_numerical_ratio("google") == 0.0

    def test_all_digits(self):
        assert calc_numerical_ratio("12345") == 1.0

    def test_mixed(self):
        assert abs(calc_numerical_ratio("abc123") - 0.5) < 1e-9

    def test_empty(self):
        # Edge case: define behavior for empty string
        result = calc_numerical_ratio("")
        assert result == 0.0 or np.isnan(result)  # Either convention is fine


# ── Feature 3: Meaningful Word Ratio ──

class TestMeaningfulWordRatio:
    def test_fully_meaningful(self, dictionary):
        # 'googlebot' = 'google' + 'bot' -> 9/9 = 1.0
        # TODO: Adjust expected value based on your implementation
        pass

    def test_no_meaningful(self, dictionary):
        result = calc_meaningful_word_ratio("xyzqqq", dictionary)
        assert result == 0.0

    def test_partial(self, dictionary):
        # 'googlexyz' = 'google' covers 6/9 = 0.667
        # TODO: Implement test based on your matching strategy
        pass


# ── Feature 4: Pronounceability ──

class TestPronounceability:
    def test_pronounceable(self, ngram_table):
        # 'google' has known trigrams -> higher score
        # TODO: Implement with expected value
        pass

    def test_unpronounceable(self, ngram_table):
        # 'xyzqqq' has rare trigrams -> lower score
        # TODO: Implement
        pass


# ── Feature 5: LMS Percentage ──

class TestLMSPercentage:
    def test_full_match(self, dictionary):
        result = calc_lms_percentage("google", dictionary)
        assert abs(result - 1.0) < 1e-9

    def test_no_match(self, dictionary):
        result = calc_lms_percentage("xyzqqq", dictionary)
        assert result == 0.0

    def test_partial_match(self, dictionary):
        # 'googlexyz' -> LMS is 'google' (6/9 = 0.667)
        # TODO: Implement
        pass


# ── Feature 6: Levenshtein Distance ──

class TestLevenshtein:
    def test_identical(self):
        assert calc_levenshtein("google", "google") == 0.0

    def test_one_edit(self):
        assert calc_levenshtein("google", "googla") == 1.0

    def test_completely_different(self):
        assert calc_levenshtein("abc", "xyz") == 3.0

    def test_empty_strings(self):
        assert calc_levenshtein("", "") == 0.0
        assert calc_levenshtein("abc", "") == 3.0
        assert calc_levenshtein("", "xyz") == 3.0


# ── Combined Extraction ──

class TestExtractFeatures:
    def test_output_shape(self, dictionary, ngram_table):
        result = extract_features("google", "yahoo", dictionary, ngram_table)
        assert result.shape == (6,)
        assert result.dtype == np.float64

    def test_no_nan(self, dictionary, ngram_table):
        result = extract_features("testdomain", "prevdomain", dictionary, ngram_table)
        assert not np.any(np.isnan(result))
