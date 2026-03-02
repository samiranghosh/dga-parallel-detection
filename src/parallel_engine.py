"""
Parallel Feature Extraction Engine
====================================
Owner: Member 3 (Parallel Engine)

Orchestrates data-parallel feature extraction using multiprocessing.Pool.
This is Layer 1 parallelism (manual data parallelism).

Key design decisions:
- Uses Pool.map() for simplicity and automatic load balancing
- Windows-compatible: uses 'spawn' start method (no fork)
- Worker initialization pattern for shared resources (avoids pickle overhead)
- Overlapping chunks for Levenshtein boundary correctness
- Pool size capped at 61 on Windows (WaitForMultipleObjects limit = 63 handles)
"""

import os
import numpy as np
import multiprocessing
import time
from typing import List, Tuple, Optional

from src.chunker import Chunk, create_overlapping_chunks

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


def parallel_extract_features(domain_list: list, k: int,
                              dictionary,
                              ngram_table,
                              pool_size: int = None,
                              skip_levenshtein: bool = False,
                              use_shared_memory: bool = False,
                              shm_names: dict = None,
                              robust: bool = False,
                              max_retries: int = 2,
                              chunk_timeout: float = None) -> np.ndarray:
    """Orchestrate parallel feature extraction across K chunks.

    The data is split into K overlapping chunks. The actual number of
    worker processes in the Pool can be smaller than K (controlled by
    pool_size); Pool.map() queues excess chunks automatically.

    This decoupling is important for:
      - E4 chunk-size sweep (64 chunks, 8 workers)
      - Windows safety (max 61 pool workers due to handle limit)

    Args:
        domain_list: Sorted list of domain strings.
        k: Number of chunks to split the data into.
        dictionary: English dictionary (will be shared via initializer).
        ngram_table: N-gram frequency table (will be shared via initializer).
        pool_size: Number of worker processes. Defaults to min(k, cores).
                   Automatically capped at 61 on Windows.
        skip_levenshtein: If True, extract 5 features only (without
            Levenshtein distance). Faster extraction and higher accuracy.
        use_shared_memory: If True, use multiprocessing.shared_memory
            instead of pickle-based Pool initializer. Requires shm_names.
        shm_names: Dict from SharedMemoryResources.get_names(). Required
            when use_shared_memory=True.
        robust: If True, use fault-tolerant extraction with validation
                and retry logic (from fault_handler.py).
        max_retries: Max retries per failed chunk (only if robust=True).
        chunk_timeout: Per-chunk timeout in seconds (only if robust=True).

    Returns:
        np.ndarray of shape (N, 5 or 6) — merged feature matrix.
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
        return robust_parallel_extract(
            chunks, n_pool, dictionary, ngram_table,
            max_retries=max_retries,
            chunk_timeout=chunk_timeout,
        )

    # Choose initializer based on shared memory flag
    if use_shared_memory and shm_names is not None:
        initializer = _init_worker_shm
        initargs = (shm_names, skip_levenshtein)
    else:
        initializer = _init_worker
        initargs = (dictionary, ngram_table, skip_levenshtein)

    with multiprocessing.Pool(
        processes=n_pool,
        initializer=initializer,
        initargs=initargs,
    ) as pool:
        results = pool.map(extract_chunk_features, chunks)

    return np.vstack(results)
