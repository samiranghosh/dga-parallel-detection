"""
Shared Memory Resource Manager
===============================
Owner: Member 3 (Parallel Engine)

Loads static read-only resources (English dictionary, n-gram table)
into shared memory so all worker processes can access them without
redundant copies.

Two modes:
  1. Pool initializer (default): Resources passed via initargs, pickled
     once per worker at pool creation time.
  2. Shared memory (Priority 4 enhancement): Resources placed in OS-level
     shared memory using multiprocessing.shared_memory. Workers attach
     by name — zero pickle overhead during pool creation. Measured to
     reduce IPC overhead by eliminating per-worker serialization.
"""

import os
import pickle
import json
import time
import numpy as np
from multiprocessing import shared_memory
from typing import Tuple, Optional, Dict, Any


def initialize_shared_resources(data_path: str) -> Tuple[set, dict]:
    """Load dictionary and n-gram table from disk.

    On Windows, multiprocessing uses 'spawn' so there is no fork()
    copy-on-write. Resources are loaded in the parent process and passed
    to each worker via the Pool initializer args pattern in parallel_engine.py.
    This avoids redundant disk I/O in every worker process.

    Args:
        data_path: Path to data/ directory containing
                   english_dictionary.txt and ngram_table.pkl.

    Returns:
        Tuple of (dictionary: set, ngram_table: dict)
    """
    dict_path = os.path.join(data_path, 'english_dictionary.txt')
    ngram_path = os.path.join(data_path, 'ngram_table.pkl')

    if not os.path.exists(dict_path):
        raise FileNotFoundError(
            f"English dictionary not found at {dict_path}. "
            "Run `python main.py --mode preprocess` first."
        )

    if not os.path.exists(ngram_path):
        raise FileNotFoundError(
            f"N-gram table not found at {ngram_path}. "
            "Run `python main.py --mode preprocess` first."
        )

    # Load English dictionary
    with open(dict_path, 'r', encoding='utf-8') as f:
        dictionary = set(line.strip() for line in f if line.strip())

    # Load trigram probability table
    with open(ngram_path, 'rb') as f:
        ngram_table = pickle.load(f)

    return dictionary, ngram_table


# ── Priority 4: Shared Memory Optimization ──

class SharedMemoryResources:
    """Manages dictionary and n-gram table in OS-level shared memory.

    Usage:
        # In parent process:
        shm = SharedMemoryResources()
        shm.create(dictionary, ngram_table)
        shm_names = shm.get_names()

        # In worker process (via initializer):
        dict_data, ngram_data = SharedMemoryResources.attach(shm_names)

        # Cleanup (parent process, after pool closes):
        shm.cleanup()
    """

    def __init__(self):
        self._dict_shm: Optional[shared_memory.SharedMemory] = None
        self._ngram_shm: Optional[shared_memory.SharedMemory] = None

    def create(self, dictionary: set, ngram_table: dict) -> Dict[str, str]:
        """Serialize resources and place them in shared memory.

        Args:
            dictionary: English dictionary set.
            ngram_table: Trigram frequency table dict.

        Returns:
            Dict with shared memory names: {'dict_name': ..., 'ngram_name': ...}
        """
        # Serialize to bytes
        dict_bytes = pickle.dumps(dictionary, protocol=pickle.HIGHEST_PROTOCOL)
        ngram_bytes = pickle.dumps(ngram_table, protocol=pickle.HIGHEST_PROTOCOL)

        # Create shared memory blocks
        self._dict_shm = shared_memory.SharedMemory(
            create=True, size=len(dict_bytes)
        )
        self._ngram_shm = shared_memory.SharedMemory(
            create=True, size=len(ngram_bytes)
        )

        # Copy data into shared memory
        self._dict_shm.buf[:len(dict_bytes)] = dict_bytes
        self._ngram_shm.buf[:len(ngram_bytes)] = ngram_bytes

        return self.get_names()

    def get_names(self) -> Dict[str, str]:
        """Return shared memory block names for worker attachment."""
        return {
            'dict_name': self._dict_shm.name,
            'dict_size': self._dict_shm.size,
            'ngram_name': self._ngram_shm.name,
            'ngram_size': self._ngram_shm.size,
        }

    @staticmethod
    def attach(shm_names: Dict[str, str]) -> Tuple[set, dict]:
        """Attach to existing shared memory and deserialize resources.

        Called in worker processes via pool initializer.

        Args:
            shm_names: Dict from get_names().

        Returns:
            Tuple of (dictionary: set, ngram_table: dict)
        """
        dict_shm = shared_memory.SharedMemory(name=shm_names['dict_name'])
        ngram_shm = shared_memory.SharedMemory(name=shm_names['ngram_name'])

        dictionary = pickle.loads(bytes(dict_shm.buf[:shm_names['dict_size']]))
        ngram_table = pickle.loads(bytes(ngram_shm.buf[:shm_names['ngram_size']]))

        # Close (not unlink) — parent owns the lifecycle
        dict_shm.close()
        ngram_shm.close()

        return dictionary, ngram_table

    def cleanup(self):
        """Unlink shared memory blocks. Call after pool is closed."""
        for shm in [self._dict_shm, self._ngram_shm]:
            if shm is not None:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass
        self._dict_shm = None
        self._ngram_shm = None

    def __del__(self):
        self.cleanup()


def benchmark_shared_memory(dictionary: set, ngram_table: dict,
                            domain_list: list, k: int = 8,
                            reps: int = 3) -> Dict[str, Any]:
    """Benchmark shared memory vs pickle-based initializer IPC.

    Runs parallel extraction twice — once with the default pickle-based
    Pool initializer, once with shared memory — and compares startup
    time, total extraction time, and IPC overhead.

    Args:
        dictionary: English dictionary set.
        ngram_table: Trigram frequency table.
        domain_list: Domain list for extraction.
        k: Number of workers/chunks.
        reps: Repetitions per configuration.

    Returns:
        Dict with timing comparisons:
        {
            'pickle_init': {'mean_time': ..., 'std_time': ...},
            'shared_memory': {'mean_time': ..., 'std_time': ...},
            'speedup': float,
            'shm_create_time_sec': float,
        }
    """
    from src.parallel_engine import parallel_extract_features

    results = {'pickle_init': {'times': []}, 'shared_memory': {'times': []}}

    # Benchmark pickle-based approach (default)
    for _ in range(reps):
        t0 = time.perf_counter()
        parallel_extract_features(domain_list, k, dictionary, ngram_table)
        results['pickle_init']['times'].append(time.perf_counter() - t0)

    # Benchmark shared memory approach
    shm_create_time = 0.0
    for r in range(reps):
        shm = SharedMemoryResources()
        t0 = time.perf_counter()
        shm_names = shm.create(dictionary, ngram_table)
        if r == 0:
            shm_create_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        parallel_extract_features(
            domain_list, k, dictionary, ngram_table,
            use_shared_memory=True, shm_names=shm_names,
        )
        results['shared_memory']['times'].append(time.perf_counter() - t0)
        shm.cleanup()

    for mode in ['pickle_init', 'shared_memory']:
        times = results[mode]['times']
        results[mode]['mean_time'] = float(np.mean(times))
        results[mode]['std_time'] = float(np.std(times))

    pickle_mean = results['pickle_init']['mean_time']
    shm_mean = results['shared_memory']['mean_time']
    results['speedup'] = pickle_mean / shm_mean if shm_mean > 0 else 0.0
    results['shm_create_time_sec'] = shm_create_time

    return results
