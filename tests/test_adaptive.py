import time
import numpy as np
import pytest

from src.parallel_engine import static_extract_features, AdaptiveController
from src.benchmark_adaptive import run_streaming_benchmark
from src.load_profiles import uniform_load

# Basic fixtures for testing
@pytest.fixture
def test_domains():
    return ["google", "facebook", "amazon", "microsoft", "apple", "netflix", 
            "twitter", "linkedin", "test1", "test2", "test3", "test4"] * 5

@pytest.fixture
def dictionary():
    return {"google", "face", "book", "amazon", "micro", "soft", "apple", "net", "link", "test"}

@pytest.fixture
def ngram_table():
    return {"goo": 0.01, "oog": 0.008, "ogl": 0.005, "gle": 0.012,
            "fac": 0.009, "ace": 0.011, "ama": 0.006, "maz": 0.002}

def test_static_extract_features(test_domains, dictionary, ngram_table):
    """Test the static pool extraction baseline."""
    # 5 features
    res = static_extract_features(test_domains, k=2, dictionary=dictionary, 
                                  ngram_table=ngram_table, skip_levenshtein=True)
    assert res.shape == (len(test_domains), 5)
    
def test_streaming_benchmark_correctness(test_domains, dictionary, ngram_table):
    """Test that the streaming benchmark runs without error and returns reasonable stats."""
    gen = uniform_load(test_domains, rate=100.0, batch_size=10)
    stats = run_streaming_benchmark(
        domain_list=test_domains,
        dictionary=dictionary,
        ngram_table=ngram_table,
        load_profile_generator=gen,
        min_workers=1,
        max_workers=2,
        skip_levenshtein=True
    )
    
    assert stats["throughput_domains_per_sec"] > 0
    assert stats["mean_latency_ms"] > 0
    assert stats["workers_spawned"] >= 1
    
def test_adaptive_bounds(test_domains, dictionary, ngram_table):
    """Test that the controller respects min and max worker bounds."""
    # Force a slow rate so queue is empty (should stay at min_workers)
    gen = uniform_load(test_domains, rate=1.0, batch_size=5)
    
    stats = run_streaming_benchmark(
        domain_list=test_domains,
        dictionary=dictionary,
        ngram_table=ngram_table,
        load_profile_generator=gen,
        min_workers=2,
        max_workers=4,
        skip_levenshtein=True
    )
    
    # It should have started with 2 workers and never exceeded 4.
    # Since load is very light, it shouldn't spawn more than 2.
    assert stats["workers_spawned"] == 2
    assert stats["mean_active_workers"] >= 2.0
    assert stats["mean_active_workers"] <= 4.0


def test_proportional_delta_scaling():
    """RQ1: scaling magnitude is PROPORTIONAL to the queue-depth error (not a
    fixed +/-1), holds inside the hysteresis dead-band, respects the min/max
    band, and CPU headroom gates scale-up only.

    Exercises the pure decision method directly (no worker processes), so it is
    fast and deterministic.
    """
    from src.parallel_engine import AdaptiveController

    c = AdaptiveController.__new__(AdaptiveController)  # bare instance, no __init__
    c.min_workers = 1
    c.max_workers = 8
    c.target_queue_per_worker = 2
    c.kp = 0.5
    c.hysteresis = 1.0

    # Large backlog with 1 worker -> add many at once (proportional), clamped to
    # max. The old bang-bang controller could only ever add 1.
    assert c._proportional_delta(qsize=500, num_workers=1) == 7   # ramps 1 -> 8
    # Moderate backlog -> proportional step > 1, still under max.
    d = c._proportional_delta(qsize=20, num_workers=2)
    assert d > 1 and 2 + d <= 8
    # Shallow queue at setpoint -> hold (hysteresis dead-band).
    assert c._proportional_delta(qsize=4, num_workers=2) == 0
    # Slack (near-empty queue, many workers) -> scale DOWN proportionally.
    assert c._proportional_delta(qsize=0, num_workers=8) < -1
    # Band respected at the edges.
    assert c._proportional_delta(qsize=10_000, num_workers=8) == 0   # already max
    assert c._proportional_delta(qsize=0, num_workers=1) == 0        # already min
    # CPU saturated -> no scale-up despite heavy backlog.
    assert c._proportional_delta(qsize=500, num_workers=1, cpu_ok_for_scale=False) == 0
