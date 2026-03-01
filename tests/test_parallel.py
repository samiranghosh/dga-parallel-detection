"""
Parallel Correctness Tests
============================
The most critical test: verifies that parallel extraction
produces IDENTICAL results to the sequential baseline.

Run: pytest tests/test_parallel.py -v
"""

import pytest
import numpy as np

# Minimal shared fixtures  — avoids loading 1M domains in tests
SAMPLE_DOMAINS = sorted([
    "google", "facebook", "amazon", "xyzabc", "qqqqwww",
    "microsoft", "apple", "netflix", "twitter", "linkedin",
    "randomdga1", "randomdga2", "aabbccdd", "testdomain", "example",
    "malicioussiteabc", "cryptolockerdga", "lockymalware", "botnetcnc", "evildomain",
])

DICTIONARY = {"google", "face", "book", "amazon", "micro", "soft", "apple", "net", "test", "example", "link", "mail"}

NGRAM_TABLE = {
    "goo": 0.01, "oog": 0.008, "ogl": 0.005, "gle": 0.012,
    "fac": 0.009, "ace": 0.011, "ceb": 0.003, "ebo": 0.004, "boo": 0.007,
    "ama": 0.006, "maz": 0.002, "azo": 0.003, "zon": 0.005,
    "mic": 0.008, "icr": 0.004, "cro": 0.009, "ros": 0.006, "oso": 0.003,
}


class TestParallelCorrectness:
    """Verify parallel output matches sequential output exactly."""

    def test_parallel_equals_sequential(self):
        """CRITICAL TEST: parallel features == sequential features."""
        from src.features import extract_all_sequential
        from src.parallel_engine import parallel_extract_features

        sequential = extract_all_sequential(SAMPLE_DOMAINS, DICTIONARY, NGRAM_TABLE)
        parallel = parallel_extract_features(SAMPLE_DOMAINS, 4, DICTIONARY, NGRAM_TABLE)

        assert sequential.shape == parallel.shape, (
            f"Shape mismatch: sequential={sequential.shape}, parallel={parallel.shape}"
        )
        assert np.allclose(sequential, parallel, rtol=1e-10, atol=1e-10), (
            f"Feature mismatch:\nseq:\n{sequential}\npar:\n{parallel}"
        )

    def test_parallel_different_k_values(self):
        """Results should be identical regardless of K."""
        from src.parallel_engine import parallel_extract_features

        results = {}
        for k in [2, 4]:  # skip K=8 as we only have 20 domains
            results[k] = parallel_extract_features(SAMPLE_DOMAINS, k, DICTIONARY, NGRAM_TABLE)

        assert np.allclose(results[2], results[4], rtol=1e-10, atol=1e-10), (
            "K=2 and K=4 produced different results!"
        )

    def test_parallel_deterministic(self):
        """Two runs with same K should produce identical results."""
        from src.parallel_engine import parallel_extract_features

        r1 = parallel_extract_features(SAMPLE_DOMAINS, 2, DICTIONARY, NGRAM_TABLE)
        r2 = parallel_extract_features(SAMPLE_DOMAINS, 2, DICTIONARY, NGRAM_TABLE)

        assert np.allclose(r1, r2, rtol=1e-10, atol=1e-10), (
            "Non-deterministic parallel results detected!"
        )
