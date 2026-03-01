"""
Overlapping Chunk Partitioner
==============================
Owner: Member 3 (Parallel Engine)

Creates overlapping chunks for parallel feature extraction.
Each chunk (except the first) receives 1 context domain from the
preceding chunk to correctly compute Levenshtein distance at boundaries.
Includes adaptive K selection via micro-benchmark.
"""

import time
import numpy as np
from typing import List, Tuple, Optional


# Type alias: (context_domain_or_None, list_of_domains_in_chunk)
Chunk = Tuple[Optional[str], List[str]]


def create_overlapping_chunks(domain_list: list, k: int) -> List[Chunk]:
    """Split domain_list into K overlapping chunks.

    Chunk 0: (None, domains[0 : N/K])
    Chunk i: (domains[i*C - 1], domains[i*C : (i+1)*C])

    The context domain is used only for computing the Levenshtein distance
    of the first domain in each chunk. It is NOT included in the output
    feature matrix.

    Args:
        domain_list: Sorted list of domain strings.
        k: Number of chunks (= number of worker processes).

    Returns:
        List of K Chunk tuples.
    """
    n = len(domain_list)
    chunk_size = n // k
    remainder = n % k

    chunks: List[Chunk] = []
    start = 0
    for i in range(k):
        # Distribute remainder domains across the first `remainder` chunks
        end = start + chunk_size + (1 if i < remainder else 0)
        chunk_domains = domain_list[start:end]
        context = domain_list[start - 1] if start > 0 else None
        chunks.append((context, chunk_domains))
        start = end

    return chunks


def auto_tune_k(domain_list: list, dictionary: set,
                ngram_table: dict, candidates: list = None) -> int:
    """Run a micro-benchmark to select the optimal K.

    Tests K = 2, 4, 8 on a 10K-domain subset and returns the K
    with the lowest wall-clock time per domain.

    Args:
        domain_list: Full domain list (will be subsampled to 10K).
        dictionary: English dictionary for feature extraction.
        ngram_table: Trigram frequency table.
        candidates: List of K values to test (default: [2, 4, 8]).

    Returns:
        Optimal K value.
    """
    from src.parallel_engine import parallel_extract_features

    if candidates is None:
        candidates = [2, 4, 8]

    subset_size = min(10000, len(domain_list))
    subset = domain_list[:subset_size]

    best_k = candidates[0]
    best_time = float('inf')

    for k in candidates:
        t0 = time.perf_counter()
        parallel_extract_features(subset, k, dictionary, ngram_table)
        elapsed = time.perf_counter() - t0
        if elapsed < best_time:
            best_time = elapsed
            best_k = k

    return best_k
