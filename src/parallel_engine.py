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
"""

import numpy as np
import multiprocessing
import time
from typing import List, Tuple, Optional

from src.chunker import Chunk, create_overlapping_chunks


# ── Global worker state (set by initializer, avoids pickling) ──

_dictionary = None
_ngram_table = None


def _init_worker(dictionary, ngram_table):
    """Pool initializer: store shared resources in worker globals."""
    global _dictionary, _ngram_table
    _dictionary = dictionary
    _ngram_table = ngram_table


def extract_chunk_features(chunk: Chunk) -> np.ndarray:
    """Worker function: extract features for one chunk.

    Args:
        chunk: Tuple of (context_domain_or_None, domain_list).

    Returns:
        np.ndarray of shape (len(domain_list), 6).
    """
    from src.features import extract_features

    context, domains = chunk
    n = len(domains)
    features = np.zeros((n, 6), dtype=np.float64)

    prev_domain = context if context is not None else domains[0]

    for i, domain in enumerate(domains):
        features[i] = extract_features(domain, prev_domain, _dictionary, _ngram_table)
        prev_domain = domain

    return features


def parallel_extract_features(domain_list: list, k: int,
                              dictionary,
                              ngram_table,
                              robust: bool = False,
                              max_retries: int = 2,
                              chunk_timeout: float = None) -> np.ndarray:
    """Orchestrate parallel feature extraction across K workers.

    Args:
        domain_list: Sorted list of domain strings.
        k: Number of worker processes.
        dictionary: English dictionary (will be shared via initializer).
        ngram_table: N-gram frequency table (will be shared via initializer).
        robust: If True, use fault-tolerant extraction with validation
                and retry logic (from fault_handler.py).
        max_retries: Max retries per failed chunk (only if robust=True).
        chunk_timeout: Per-chunk timeout in seconds (only if robust=True).

    Returns:
        np.ndarray of shape (N, 6) — merged feature matrix.
    """
    chunks = create_overlapping_chunks(domain_list, k)

    if robust:
        from src.fault_handler import robust_parallel_extract
        return robust_parallel_extract(
            chunks, k, dictionary, ngram_table,
            max_retries=max_retries,
            chunk_timeout=chunk_timeout,
        )

    with multiprocessing.Pool(
        processes=k,
        initializer=_init_worker,
        initargs=(dictionary, ngram_table)
    ) as pool:
        results = pool.map(extract_chunk_features, chunks)

    return np.vstack(results)
