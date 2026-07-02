"""
Fault Handler Tests
=====================
Tests validation logic and robust parallel extraction.

Run: pytest tests/test_fault_handler.py -v
"""

import pytest
import numpy as np
from src.fault_handler import validate_features, VALID_FEATURE_COUNTS


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
        """Wrong number of columns (not 5 or 6) should fail."""
        result = np.array([[6.0, 0.0, 0.667]])  # only 3 cols — invalid
        assert validate_features(result, expected_rows=1) is False

    def test_valid_5_features(self):
        """5-feature production output (no Levenshtein) should pass."""
        result = np.array([
            [6.0, 0.0, 0.667, 0.005, 1.0],
            [10.0, 0.3, 0.5, 0.002, 0.4],
        ])
        assert validate_features(result, expected_rows=2) is True

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
        assert result.shape[1] in VALID_FEATURE_COUNTS
        assert result.shape[0] == len(self.DOMAINS)


class TestAdaptiveWorkerRecovery:
    """RQ1 A5: the adaptive controller reaps and replaces dead workers.

    Deterministic (no process-killing) — uses fake worker handles so it cannot
    flake on timing. The full end-to-end worker-kill recovery (controller
    replacement + collector re-queue -> output identical to sequential) is
    exercised by scripts/fault injection; this locks the decision logic in CI.
    """

    class _FakeWorker:
        def __init__(self, alive):
            self._alive = alive
        def is_alive(self):
            return self._alive
        def join(self, timeout=None):
            pass

    def _bare_controller(self, target):
        from src.parallel_engine import AdaptiveController
        c = AdaptiveController.__new__(AdaptiveController)  # skip __init__/processes
        c.target_workers = target
        c.workers_replaced = 0
        return c

    def test_replaces_dead_worker_up_to_target(self):
        c = self._bare_controller(target=3)
        c.workers = [self._FakeWorker(True), self._FakeWorker(False),
                     self._FakeWorker(True)]  # one dead
        spawned = []
        c._start_worker = lambda: (c.workers.append(self._FakeWorker(True)),
                                   spawned.append(1))

        replaced = c._reap_and_replace()
        assert replaced == 1                              # one death -> one replacement
        assert c.workers_replaced == 1
        assert len(c.workers) == 3                        # live count restored to target
        assert all(w.is_alive() for w in c.workers)
        assert len(spawned) == 1

    def test_no_replacement_when_all_alive(self):
        c = self._bare_controller(target=2)
        c.workers = [self._FakeWorker(True), self._FakeWorker(True)]
        c._start_worker = lambda: (_ for _ in ()).throw(
            AssertionError("must not spawn when healthy"))
        assert c._reap_and_replace() == 0
        assert c.workers_replaced == 0

    def test_pending_scale_down_is_not_a_deficit(self):
        # target already lowered (e.g. mid scale-down); live count still higher.
        c = self._bare_controller(target=1)
        c.workers = [self._FakeWorker(True), self._FakeWorker(True)]
        c._start_worker = lambda: (_ for _ in ()).throw(
            AssertionError("must not spawn above target"))
        assert c._reap_and_replace() == 0                 # surplus, not deficit
        assert len(c.workers) == 2
