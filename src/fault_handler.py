"""
Fault Tolerance Handler
========================
Owner: Member 3 (Parallel Engine)

Provides retry logic, validation, and timeout handling for the
parallel worker pool. Ensures transient failures don't abort
the entire pipeline.

Design (from P1 Section 12):
  - Per-chunk validation after every worker return
  - Up to max_retries re-submissions for failed chunks
  - Optional per-chunk timeout via apply_async
  - Detailed logging of successes, retries, and failures
"""

import numpy as np
import logging
import multiprocessing
import time
from typing import List, Optional

logger = logging.getLogger(__name__)

# Expected feature count
N_FEATURES = 6


def validate_features(result: np.ndarray, expected_rows: int) -> bool:
    """Validate a worker's feature extraction output.

    Checks:
    - Shape is (expected_rows, 6)
    - No NaN values
    - No Inf values
    - Length feature (col 0) >= 0
    - Ratio features (cols 1, 2, 4) in [0, 1]
    - Pronounceability (col 3) >= 0
    - Levenshtein distance (col 5) >= 0

    Returns:
        True if valid, False otherwise.
    """
    # Shape check
    if result.shape != (expected_rows, N_FEATURES):
        logger.warning(
            f"Shape mismatch: expected ({expected_rows}, {N_FEATURES}), "
            f"got {result.shape}"
        )
        return False

    # NaN check
    nan_count = int(np.isnan(result).sum())
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values in feature matrix")
        return False

    # Inf check
    inf_count = int(np.isinf(result).sum())
    if inf_count > 0:
        logger.warning(f"Found {inf_count} Inf values in feature matrix")
        return False

    # Col 0: Length — must be >= 0
    if np.any(result[:, 0] < 0):
        logger.warning("Negative length values detected")
        return False

    # Col 1: Numerical ratio — must be in [0, 1]
    if np.any(result[:, 1] < -1e-9) or np.any(result[:, 1] > 1.0 + 1e-9):
        logger.warning("Numerical ratio outside [0, 1]")
        return False

    # Col 2: Meaningful word ratio — must be in [0, 1]
    if np.any(result[:, 2] < -1e-9) or np.any(result[:, 2] > 1.0 + 1e-9):
        logger.warning("Meaningful word ratio outside [0, 1]")
        return False

    # Col 3: Pronounceability — must be >= 0
    if np.any(result[:, 3] < -1e-9):
        logger.warning("Negative pronounceability values detected")
        return False

    # Col 4: LMS percentage — must be in [0, 1]
    if np.any(result[:, 4] < -1e-9) or np.any(result[:, 4] > 1.0 + 1e-9):
        logger.warning("LMS percentage outside [0, 1]")
        return False

    # Col 5: Levenshtein distance — must be >= 0
    if np.any(result[:, 5] < -1e-9):
        logger.warning("Negative Levenshtein distance detected")
        return False

    return True


def robust_parallel_extract(chunks: list, k: int,
                            dictionary, ngram_table,
                            max_retries: int = 2,
                            chunk_timeout: float = None) -> np.ndarray:
    """Execute parallel extraction with retry and timeout logic.

    Strategy:
      1. First attempt: pool.map() all chunks
      2. Validate each result via validate_features()
      3. Collect failed chunk indices
      4. Retry loop for failed chunks (up to max_retries)
      5. np.vstack() all valid results in original order
      6. Log statistics: total chunks, retries, failures

    Args:
        chunks: List of Chunk tuples from chunker.
        k: Number of worker processes.
        dictionary: English dictionary for worker initializer.
        ngram_table: N-gram table for worker initializer.
        max_retries: Maximum retry attempts per failed chunk.
        chunk_timeout: Timeout in seconds per chunk (None = no timeout).

    Returns:
        Merged feature matrix np.ndarray of shape (N, 6).

    Raises:
        RuntimeError: If any chunk fails after all retries.
    """
    from src.parallel_engine import _init_worker, extract_chunk_features, _safe_pool_size

    n_chunks = len(chunks)
    results = [None] * n_chunks
    pending = list(range(n_chunks))
    total_retried = 0

    for attempt in range(1 + max_retries):
        if not pending:
            break

        if attempt > 0:
            total_retried += len(pending)
            logger.warning(
                f"Retry {attempt}/{max_retries}: re-processing "
                f"{len(pending)} chunk(s)"
            )

        items = [(i, chunks[i]) for i in pending]

        try:
            pool_workers = _safe_pool_size(min(k, len(items)))
            pool = multiprocessing.Pool(
                processes=pool_workers,
                initializer=_init_worker,
                initargs=(dictionary, ngram_table),
            )

            if chunk_timeout is not None:
                # Per-chunk timeout via apply_async
                async_handles = []
                for idx, chunk in items:
                    handle = pool.apply_async(extract_chunk_features, (chunk,))
                    async_handles.append((idx, chunk, handle))

                pool.close()

                still_failed = []
                for idx, chunk, handle in async_handles:
                    try:
                        result = handle.get(timeout=chunk_timeout)
                        _, domains = chunk
                        if validate_features(result, len(domains)):
                            results[idx] = result
                        else:
                            still_failed.append(idx)
                    except multiprocessing.TimeoutError:
                        logger.warning(f"Chunk {idx}: timed out ({chunk_timeout}s)")
                        still_failed.append(idx)
                    except Exception as exc:
                        logger.warning(f"Chunk {idx}: worker error — {exc}")
                        still_failed.append(idx)

                pool.join()
            else:
                # Fast path: pool.map (no per-chunk timeout)
                batch = [chunk for _, chunk in items]
                batch_results = pool.map(extract_chunk_features, batch)
                pool.close()
                pool.join()

                still_failed = []
                for (idx, chunk), result in zip(items, batch_results):
                    _, domains = chunk
                    if validate_features(result, len(domains)):
                        results[idx] = result
                    else:
                        still_failed.append(idx)

        except Exception as exc:
            logger.error(f"Pool failure on attempt {attempt}: {exc}")
            still_failed = [idx for idx, _ in items]
            try:
                pool.terminate()
                pool.join()
            except Exception:
                pass

        pending = still_failed

    # Final report
    succeeded = sum(1 for r in results if r is not None)
    logger.info(
        f"Fault handler summary: {succeeded}/{n_chunks} chunks OK, "
        f"{len(pending)} failed, {total_retried} chunk-retries issued"
    )

    if pending:
        raise RuntimeError(
            f"{len(pending)} chunk(s) failed after {max_retries} retries: "
            f"indices {pending}"
        )

    return np.vstack(results)
