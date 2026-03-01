"""
Fault Handler Tests
=====================
Tests validation logic and robust parallel extraction.

Run: pytest tests/test_fault_handler.py -v
"""

import pytest
import numpy as np
from src.fault_handler import validate_features, N_FEATURES


class TestValidateFeatures:
    """Test the per-chunk validation logic."""

    def test_valid_features(self):
        """Well-formed features should pass validation."""
        result = np.array([
            [6.0, 0.0, 0.667, 0.005, 1.0, 3.0],
            [10.0, 0.3, 0.5, 0.002, 0.4, 5.0],
        ])
        assert validate_features(result, expected_rows=2) is True

    def test_wrong_shape_rows(self):
        """Fewer rows than expected should fail."""
        result = np.array([[6.0, 0.0, 0.667, 0.005, 1.0, 3.0]])
        assert validate_features(result, expected_rows=2) is False

    def test_wrong_shape_cols(self):
        """Wrong number of columns should fail."""
        result = np.array([[6.0, 0.0, 0.667, 0.005, 1.0]])  # only 5 cols
        assert validate_features(result, expected_rows=1) is False

    def test_nan_detection(self):
        """NaN in features should fail."""
        result = np.array([[6.0, np.nan, 0.667, 0.005, 1.0, 3.0]])
        assert validate_features(result, expected_rows=1) is False

    def test_inf_detection(self):
        """Inf in features should fail."""
        result = np.array([[6.0, 0.0, 0.667, np.inf, 1.0, 3.0]])
        assert validate_features(result, expected_rows=1) is False

    def test_negative_length(self):
        """Negative length (col 0) should fail."""
        result = np.array([[-1.0, 0.0, 0.5, 0.005, 0.5, 3.0]])
        assert validate_features(result, expected_rows=1) is False

    def test_ratio_out_of_range(self):
        """Ratio > 1 (col 1) should fail."""
        result = np.array([[6.0, 1.5, 0.5, 0.005, 0.5, 3.0]])
        assert validate_features(result, expected_rows=1) is False

    def test_negative_levenshtein(self):
        """Negative Levenshtein (col 5) should fail."""
        result = np.array([[6.0, 0.0, 0.5, 0.005, 0.5, -1.0]])
        assert validate_features(result, expected_rows=1) is False

    def test_zero_length_domain(self):
        """Zero-length domain features should still be valid."""
        result = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        assert validate_features(result, expected_rows=1) is True

    def test_boundary_ratio_values(self):
        """Exact 0.0 and 1.0 should pass for ratio columns."""
        result = np.array([[5.0, 0.0, 1.0, 0.01, 0.0, 10.0]])
        assert validate_features(result, expected_rows=1) is True


class TestRobustParallelExtract:
    """Test the fault-tolerant extraction pipeline."""

    DOMAINS = sorted([
        "google", "facebook", "amazon", "xyzabc", "qqqqwww",
        "microsoft", "apple", "netflix", "twitter", "linkedin",
    ])
    DICT = {"google", "face", "book", "amazon", "micro", "soft",
            "apple", "net", "test", "link"}
    NGRAMS = {"goo": 0.01, "oog": 0.008, "ogl": 0.005, "gle": 0.012,
              "fac": 0.009, "ace": 0.011, "ama": 0.006, "maz": 0.002}

    def test_robust_matches_standard(self):
        """Robust extraction should produce same results as standard."""
        from src.parallel_engine import parallel_extract_features

        standard = parallel_extract_features(
            self.DOMAINS, 2, self.DICT, self.NGRAMS, robust=False
        )
        robust = parallel_extract_features(
            self.DOMAINS, 2, self.DICT, self.NGRAMS, robust=True
        )
        assert np.allclose(standard, robust, rtol=1e-10, atol=1e-10)

    def test_robust_with_timeout(self):
        """Robust extraction with generous timeout should succeed."""
        from src.parallel_engine import parallel_extract_features

        result = parallel_extract_features(
            self.DOMAINS, 2, self.DICT, self.NGRAMS,
            robust=True, chunk_timeout=60.0,
        )
        assert result.shape == (len(self.DOMAINS), N_FEATURES)
