import time
import queue
import logging
import multiprocessing
import threading
import numpy as np
from typing import List, Tuple, Dict, Any, Callable

from src.parallel_engine import AdaptiveController
from src.chunker import Chunk

logger = logging.getLogger(__name__)


def run_streaming_benchmark(
    domain_list: List[str], 
    dictionary: set, 
    ngram_table: dict,
    load_profile_generator: Callable,
    min_workers: int,
    max_workers: int,
    skip_levenshtein: bool = True
) -> Dict[str, Any]:
    """Runs a streaming benchmark using the AdaptiveController.
    
    Args:
        domain_list: List of domains to process.
        dictionary: The reference dictionary.
        ngram_table: The reference ngram table.
        load_profile_generator: An iterator yielding (scheduled_time, batch_of_domains).
        min_workers: Minimum worker count (set equal to max_workers for static baseline).
        max_workers: Maximum worker count.
        skip_levenshtein: Whether to skip the Levenshtein feature.
        
    Returns:
        A dictionary containing latency and worker metrics.
    """
    # Setup shared memory for workers
    from src.shared_resources import SharedMemoryResources
    shm = SharedMemoryResources()
    shm_names = shm.create(dictionary, ngram_table)
    
    in_queue = multiprocessing.Queue()
    out_queue = multiprocessing.Queue()
    
    # Initialize controller
    controller = AdaptiveController(in_queue, out_queue, min_workers, max_workers, shm_names, skip_levenshtein)
    
    # Tracking
    total_domains = 0
    latencies = []
    active_worker_history = []
    
    # Start time reference
    start_time_ref = time.time()
    
    # Start a collector thread to gather results as they finish
    results_received = 0
    expected_results = 0
    
    def collector():
        nonlocal results_received
        while results_received < expected_results or expected_results == 0:
            try:
                # We expect to get (chunk_idx, features, enqueue_time) but the worker 
                # only returns (chunk_idx, features). We can measure end-to-end latency 
                # if we track when chunks were queued.
                # To do this correctly, we'll wrap the out_queue reads.
                res = out_queue.get(timeout=1.0)
                if isinstance(res, tuple) and len(res) == 2:
                    chunk_idx, features = res
                    results_received += 1
            except queue.Empty:
                if not controller.running and results_received >= expected_results:
                    break
    
    # We will measure latency manually in the main thread by timing the feed and collect
    # Actually, a better way to measure per-batch latency without altering the worker
    # is to time it at the collector. Let's build a dictionary of chunk_idx -> enqueue_time
    enqueue_times = {}
    
    collector_thread = threading.Thread(target=collector, daemon=True)
    # We can't start collector yet because expected_results is 0. 
    # We'll run the collector in the main thread AFTER feeding, or concurrently.
    
    try:
        # Feeder loop
        chunk_idx = 0
        prev_domain = None
        
        for scheduled_time, batch in load_profile_generator:
            # Wait until it's time to send this batch
            now = time.time()
            if scheduled_time > now:
                time.sleep(scheduled_time - now)
                
            # Create chunk
            chunk = (prev_domain, batch)
            prev_domain = batch[-1]
            
            # Record tracking info
            enqueue_times[chunk_idx] = time.time()
            total_domains += len(batch)
            expected_results += 1
            
            # Send to workers
            in_queue.put((chunk_idx, chunk))
            chunk_idx += 1
            
            # Record worker count over time
            active_worker_history.append(len(controller.workers))
            
            # Drain out_queue periodically to avoid blocking
            while True:
                try:
                    res = out_queue.get_nowait()
                    if isinstance(res, tuple) and len(res) == 2:
                        idx, _ = res
                        if idx in enqueue_times:
                            latencies.append(time.time() - enqueue_times[idx])
                        results_received += 1
                except queue.Empty:
                    break

        # Wait for remaining results
        while results_received < expected_results:
            try:
                res = out_queue.get(timeout=1.0)
                if isinstance(res, tuple) and len(res) == 2:
                    idx, _ = res
                    if idx in enqueue_times:
                        latencies.append(time.time() - enqueue_times[idx])
                    results_received += 1
                active_worker_history.append(len(controller.workers))
            except queue.Empty:
                if not controller.running:
                    break
                
    finally:
        controller.shutdown()
        shm.cleanup()
        
    duration = time.time() - start_time_ref
    
    stats = {
        "throughput_domains_per_sec": total_domains / duration if duration > 0 else 0,
        "mean_latency_ms": np.mean(latencies) * 1000 if latencies else 0,
        "p50_latency_ms": np.percentile(latencies, 50) * 1000 if latencies else 0,
        "p95_latency_ms": np.percentile(latencies, 95) * 1000 if latencies else 0,
        "p99_latency_ms": np.percentile(latencies, 99) * 1000 if latencies else 0,
        "mean_active_workers": np.mean(active_worker_history) if active_worker_history else min_workers,
        "workers_spawned": controller.workers_spawned,
        "workers_retired": controller.workers_retired,
        "total_monitor_cpu_ms": controller.total_monitor_cpu_ms
    }
    
    return stats


def measure_saturation_throughput(domain_list: List[str], dictionary: set, ngram_table: dict, k: int = 8) -> float:
    """Measure the streaming saturation throughput (μ) of the engine.
    
    Feeds the queue as fast as possible (no delays) to find maximum processing rate.
    """
    logger.info("Measuring saturation throughput (μ)...")
    
    # We use a simple generator that yields everything immediately
    def burst_generator():
        # Batch size of 100 for streaming emulation
        batch_size = 100
        start = time.time()
        for i in range(0, len(domain_list), batch_size):
            yield start, domain_list[i:i+batch_size]
            
    stats = run_streaming_benchmark(
        domain_list=domain_list,
        dictionary=dictionary,
        ngram_table=ngram_table,
        load_profile_generator=burst_generator(),
        min_workers=k,
        max_workers=k,
        skip_levenshtein=True
    )
    
    mu = stats["throughput_domains_per_sec"]
    logger.info(f"Measured saturation throughput: {mu:.2f} domains/s")
    return mu


def compare_adaptive_vs_static(
    domain_list: List[str], 
    dictionary: set, 
    ngram_table: dict,
    reps: int = 5
) -> Dict[str, Any]:
    """Run the full RQ1 utilization sweep protocol.
    
    Sweeps ρ ∈ {0.1, 0.3, 0.5, 0.7, 0.9, 1.05} using uniform load.
    Also tests bursty and ramp profiles at ρ = 0.6.
    Compares Adaptive (min=1, max=8) vs Static (min=8, max=8).
    """
    from src.load_profiles import uniform_load, poisson_bursty_load, ramp_load
    
    # 1. Measure saturation throughput (μ)
    mu = measure_saturation_throughput(domain_list, dictionary, ngram_table, k=8)
    
    rhos = [0.1, 0.3, 0.5, 0.7, 0.9, 1.05]
    results = {
        "mu": mu,
        "uniform_sweep": {"rhos": rhos, "static": [], "adaptive": []},
        "bursty_0.6": {"static": [], "adaptive": []},
        "ramp_0.6": {"static": [], "adaptive": []}
    }
    
    # We use a small subset of domains for the sweep to keep runtime manageable
    # e.g. 50,000 domains per test
    test_domains = domain_list[:50000]
    
    for rho in rhos:
        rate = rho * mu
        print(f"Testing uniform load at ρ={rho} (rate={rate:.2f} dom/s)")
        
        static_stats_reps = []
        adaptive_stats_reps = []
        
        for rep in range(reps):
            print(f"  Rep {rep+1}/{reps}...")
            
            # Static K=8
            gen_static = uniform_load(test_domains, rate=rate, batch_size=100)
            st_stats = run_streaming_benchmark(
                test_domains, dictionary, ngram_table, gen_static, 
                min_workers=8, max_workers=8
            )
            static_stats_reps.append(st_stats)
            
            # Adaptive 1-8
            gen_adaptive = uniform_load(test_domains, rate=rate, batch_size=100)
            ad_stats = run_streaming_benchmark(
                test_domains, dictionary, ngram_table, gen_adaptive, 
                min_workers=1, max_workers=8
            )
            adaptive_stats_reps.append(ad_stats)
            
        results["uniform_sweep"]["static"].append(static_stats_reps)
        results["uniform_sweep"]["adaptive"].append(adaptive_stats_reps)
        
    # Bursty at ρ = 0.6
    rate = 0.6 * mu
    print(f"Testing bursty load at ρ=0.6 (mean_rate={rate:.2f} dom/s)")
    for rep in range(reps):
        gen = poisson_bursty_load(test_domains, mean_rate=rate, burst_factor=3.0)
        st = run_streaming_benchmark(test_domains, dictionary, ngram_table, gen, 8, 8)
        results["bursty_0.6"]["static"].append(st)
        
        gen = poisson_bursty_load(test_domains, mean_rate=rate, burst_factor=3.0)
        ad = run_streaming_benchmark(test_domains, dictionary, ngram_table, gen, 1, 8)
        results["bursty_0.6"]["adaptive"].append(ad)
        
    # Ramp at ρ = 0.6 (average)
    # Ramp from 0.2*mu to 1.0*mu (avg 0.6*mu)
    print(f"Testing ramp load (0.2μ -> 1.0μ)")
    for rep in range(reps):
        gen = ramp_load(test_domains, start_rate=0.2*mu, end_rate=1.0*mu)
        st = run_streaming_benchmark(test_domains, dictionary, ngram_table, gen, 8, 8)
        results["ramp_0.6"]["static"].append(st)
        
        gen = ramp_load(test_domains, start_rate=0.2*mu, end_rate=1.0*mu)
        ad = run_streaming_benchmark(test_domains, dictionary, ngram_table, gen, 1, 8)
        results["ramp_0.6"]["adaptive"].append(ad)

    return results
