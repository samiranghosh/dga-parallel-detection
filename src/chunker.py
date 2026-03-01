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
    # TODO: Implement
    raise NotImplementedError


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
    # TODO: Implement
    # 1. Sample 10K domains from domain_list
    # 2. For each K in candidates:
    #    a. Create chunks
    #    b. Run parallel_extract_features
    #    c. Record wall-clock time
    # 3. Return K with minimum time
    raise NotImplementedError
