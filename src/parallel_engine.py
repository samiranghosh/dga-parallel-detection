"""
Parallel Feature Extraction Engine
====================================
Owner: Member 3 (Parallel Engine)

Orchestrates data-parallel feature extraction using multiprocessing.Pool
and an adaptive queue-fed model.
This is Layer 1 parallelism (manual data parallelism).

Key design decisions:
- Replaced Pool.map() with AdaptiveController and queue-fed workers.
- Windows-compatible: uses 'spawn' start method (no fork)
- Workers attach to shared memory resources (avoids pickle overhead).
- Overlapping chunks for Levenshtein boundary correctness
- Pool size capped at 61 on Windows (WaitForMultipleObjects limit = 63 handles)
"""

import os
import numpy as np
import multiprocessing
import threading
import time
import queue
import logging
from typing import List, Tuple, Optional, Dict

from src.chunker import Chunk, create_overlapping_chunks

logger = logging.getLogger(__name__)

# Windows WaitForMultipleObjects supports at most 63 handles.
# Pool uses handles = n_workers + internal sentinels, so cap at 61
# to leave headroom.
_WIN_MAX_POOL = 61


def _safe_pool_size(requested: int) -> int:
    """Return a pool size that is safe for the current OS.

    On Windows, caps at _WIN_MAX_POOL to avoid the
    WaitForMultipleObjects 63-handle limit.
    """
    if os.name == 'nt':
        return min(requested, _WIN_MAX_POOL)
    return requested


# ── Global worker state (set by initializer, avoids pickling) ──

_dictionary = None
_ngram_table = None
_skip_levenshtein = False


def _init_worker(dictionary, ngram_table, skip_levenshtein=False):
    """Pool initializer: store shared resources in worker globals."""
    global _dictionary, _ngram_table, _skip_levenshtein
    _dictionary = dictionary
    _ngram_table = ngram_table
    _skip_levenshtein = skip_levenshtein


def _init_worker_shm(shm_names, skip_levenshtein=False):
    """Pool initializer: attach to shared memory resources.

    Priority 4 enhancement — avoids per-worker pickle serialization
    of dictionary and n-gram table. Workers attach to pre-created
    shared memory blocks by name.
    """
    global _dictionary, _ngram_table, _skip_levenshtein
    from src.shared_resources import SharedMemoryResources
    _dictionary, _ngram_table = SharedMemoryResources.attach(shm_names)
    _skip_levenshtein = skip_levenshtein


def extract_chunk_features(chunk: Chunk) -> np.ndarray:
    """Worker function: extract features for one chunk.

    Args:
        chunk: Tuple of (context_domain_or_None, domain_list).

    Returns:
        np.ndarray of shape (len(domain_list), 5 or 6).
    """
    from src.features import extract_features

    context, domains = chunk
    n = len(domains)
    n_feats = 5 if _skip_levenshtein else 6
    features = np.zeros((n, n_feats), dtype=np.float64)

    prev_domain = context if context is not None else domains[0]

    for i, domain in enumerate(domains):
        features[i] = extract_features(domain, prev_domain, _dictionary,
                                       _ngram_table, skip_levenshtein=_skip_levenshtein)
        prev_domain = domain

    return features


def _adaptive_worker_loop(in_queue: multiprocessing.Queue, 
                          out_queue: multiprocessing.Queue, 
                          shm_names: dict, 
                          skip_levenshtein: bool):
    """Queue-fed worker loop that attaches to shared memory and processes chunks."""
    global _dictionary, _ngram_table, _skip_levenshtein
    
    # Attach to shared memory (done once per worker)
    from src.shared_resources import SharedMemoryResources
    _dictionary, _ngram_table = SharedMemoryResources.attach(shm_names)
    _skip_levenshtein = skip_levenshtein
    
    while True:
        try:
            task = in_queue.get()
            if task is None:  # Sentinel to exit
                break
            
            chunk_idx, chunk = task
            features = extract_chunk_features(chunk)
            out_queue.put((chunk_idx, features))
            
        except Exception as e:
            # Send error back to prevent deadlock
            # Extract chunk_idx if possible, else return -1
            idx = task[0] if 'task' in locals() and task is not None and isinstance(task, tuple) else -1
            out_queue.put((idx, e))
            break


class AdaptiveController:
    """Dynamically scales workers based on queue depth and CPU utilization."""
    
    def __init__(self, in_queue: multiprocessing.Queue, 
                 out_queue: multiprocessing.Queue, 
                 min_workers: int, 
                 max_workers: int, 
                 shm_names: dict, 
                 skip_levenshtein: bool):
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.shm_names = shm_names
        self.skip_levenshtein = skip_levenshtein
        
        self.workers = []
        self.running = True
        self.lock = threading.Lock()
        
        self.scale_cooldown = 0.5  # Seconds between scaling actions
        self.last_scale_time = time.time()
        
        # Instrumentation metrics (RQ1)
        self.workers_spawned = 0
        self.workers_retired = 0
        self.monitor_loop_iterations = 0
        self.total_monitor_cpu_ms = 0.0
        
        try:
            import psutil
            self.has_psutil = True
        except ImportError:
            self.has_psutil = False

        # Start initial workers
        for _ in range(self.min_workers):
            self._start_worker()
            
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
            
    def _start_worker(self):
        p = multiprocessing.Process(
            target=_adaptive_worker_loop,
            args=(self.in_queue, self.out_queue, self.shm_names, self.skip_levenshtein)
        )
        p.start()
        self.workers.append(p)
        self.workers_spawned += 1
        logger.debug(f"Started worker. Total workers: {len(self.workers)}")
        
    def _stop_worker(self):
        """Signal one worker to stop."""
        self.in_queue.put(None)
        
    def _monitor_loop(self):
        """Background thread to monitor and scale workers."""
        while self.running:
            time.sleep(0.1)
            with self.lock:
                if not self.running:
                    break
                
                t_start = time.perf_counter()
                self.monitor_loop_iterations += 1
                
                now = time.time()
                if now - self.last_scale_time < self.scale_cooldown:
                    self.total_monitor_cpu_ms += (time.perf_counter() - t_start) * 1000.0
                    continue
                    
                try:
                    qsize = self.in_queue.qsize()
                except NotImplementedError:
                    qsize = 1  
                    
                num_workers = len(self.workers)
                
                # Check CPU if psutil available
                cpu_ok_for_scale = True
                if self.has_psutil:
                    import psutil
                    if psutil.cpu_percent() > 85.0:
                        cpu_ok_for_scale = False
                
                if qsize > num_workers * 2 and num_workers < self.max_workers and cpu_ok_for_scale:
                    self._start_worker()
                    self.last_scale_time = now
                elif qsize == 0 and num_workers > self.min_workers:
                    # Scale down gracefully
                    self._stop_worker()
                    self.workers.pop()
                    self.workers_retired += 1
                    self.last_scale_time = now
                    logger.debug(f"Signaled worker to stop. Active tracked workers: {len(self.workers)}")
                    
                self.total_monitor_cpu_ms += (time.perf_counter() - t_start) * 1000.0
                    
    def shutdown(self):
        """Signal all remaining workers to stop and wait for them."""
        with self.lock:
            self.running = False
            for _ in self.workers:
                self.in_queue.put(None)
                
        for p in self.workers:
            p.join()


def static_extract_features(domain_list: list, k: int,
                            dictionary,
                            ngram_table,
                            pool_size: int = None,
                            skip_levenshtein: bool = False,
                            use_shared_memory: bool = False,
                            shm_names: dict = None) -> np.ndarray:
    """Fixed-pool fallback (the baseline for RQ1).
    
    Uses standard Pool.map over chunks with a static pool size.
    """
    chunks = create_overlapping_chunks(domain_list, k)

    if pool_size is None:
        n_pool = k
    else:
        n_pool = pool_size
    n_pool = _safe_pool_size(n_pool)

    # Note: If use_shared_memory is true, we need to handle shm_names.
    if use_shared_memory and shm_names is None:
        from src.shared_resources import SharedMemoryResources
        shm = SharedMemoryResources()
        shm_names = shm.create(dictionary, ngram_table)
        initargs = (shm_names, skip_levenshtein)
        initializer = _init_worker_shm
    elif use_shared_memory and shm_names is not None:
        initargs = (shm_names, skip_levenshtein)
        initializer = _init_worker_shm
    else:
        initargs = (dictionary, ngram_table, skip_levenshtein)
        initializer = _init_worker
        
    try:
        with multiprocessing.Pool(
            processes=n_pool,
            initializer=initializer,
            initargs=initargs
        ) as pool:
            results = pool.map(extract_chunk_features, chunks)
    finally:
        if use_shared_memory and shm_names is None and 'shm' in locals():
            shm.cleanup()

    return np.vstack(results)


def parallel_extract_features(domain_list: list, k: int,
                              dictionary,
                              ngram_table,
                              pool_size: int = None,
                              skip_levenshtein: bool = False,
                              use_shared_memory: bool = False,
                              shm_names: dict = None,
                              robust: bool = False,
                              max_retries: int = 2,
                              chunk_timeout: float = None,
                              return_stats: bool = False):
    """Orchestrate parallel feature extraction across K chunks.

    The data is split into K overlapping chunks. The original implementation
    used Pool.map(). Now it uses an AdaptiveController with multiprocessing.Queue
    to scale workers dynamically (RQ1).

    Args:
        domain_list: Sorted list of domain strings.
        k: Number of chunks to split the data into.
        dictionary: English dictionary (used if not in shared memory).
        ngram_table: N-gram frequency table (used if not in shared memory).
        pool_size: Maximum number of worker processes. Defaults to k.
                   Automatically capped at 61 on Windows.
        skip_levenshtein: If True, extract 5 features only.
        use_shared_memory: True (always used for adaptive model implicitly if shm_names are provided).
        shm_names: Dict from SharedMemoryResources.get_names().
        robust: If True, use fault-tolerant extraction with validation
                and retry logic (from fault_handler.py).
        max_retries: Max retries per failed chunk (only if robust=True).
        chunk_timeout: Per-chunk timeout in seconds (only if robust=True).
        return_stats: If True, returns a tuple (feature_matrix, stats_dict) containing AdaptiveController telemetry.

    Returns:
        np.ndarray of shape (N, 5 or 6) — merged feature matrix,
        or tuple (matrix, stats) if return_stats=True.
    """
    chunks = create_overlapping_chunks(domain_list, k)

    # Determine actual pool size
    if pool_size is None:
        n_pool = k
    else:
        n_pool = pool_size
    n_pool = _safe_pool_size(n_pool)

    if robust:
        from src.fault_handler import robust_parallel_extract
        res = robust_parallel_extract(
            chunks, n_pool, dictionary, ngram_table,
            max_retries=max_retries,
            chunk_timeout=chunk_timeout,
        )
        return (res, {}) if return_stats else res

    # Adaptive Queue-Fed Model (RQ1)
    
    # We must have shared memory names for the queue-fed model to avoid RAM bloat.
    cleanup_shm = False
    if shm_names is None:
        from src.shared_resources import SharedMemoryResources
        shm = SharedMemoryResources()
        shm_names = shm.create(dictionary, ngram_table)
        cleanup_shm = True
        
    try:
        in_queue = multiprocessing.Queue()
        out_queue = multiprocessing.Queue()
        
        # Load chunks into input queue
        for idx, chunk in enumerate(chunks):
            in_queue.put((idx, chunk))
            
        # Initialize controller
        # We start with a minimum of 2 workers (or 1 if n_pool is 1)
        min_workers = max(1, min(2, n_pool))
        max_workers = n_pool
        
        controller = AdaptiveController(in_queue, out_queue, min_workers, max_workers, shm_names, skip_levenshtein)
        
        # Collect results
        results = [None] * len(chunks)
        for _ in range(len(chunks)):
            idx, res = out_queue.get()
            if isinstance(res, Exception):
                controller.shutdown()
                raise RuntimeError(f"Worker failed on chunk {idx}: {res}") from res
            results[idx] = res
            
        controller.shutdown()
        
        stats = {
            "workers_spawned": controller.workers_spawned,
            "workers_retired": controller.workers_retired,
            "monitor_loop_iterations": controller.monitor_loop_iterations,
            "total_monitor_cpu_ms": controller.total_monitor_cpu_ms
        }
        
    finally:
        if cleanup_shm:
            shm.cleanup()

    matrix = np.vstack(results)
    if return_stats:
        return matrix, stats
    return matrix
