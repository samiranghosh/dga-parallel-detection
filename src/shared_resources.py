"""
Shared Memory Resource Manager
===============================
Owner: Member 3 (Parallel Engine)

Loads static read-only resources (English dictionary, n-gram table)
into shared memory so all worker processes can access them without
redundant copies. Saves ~30 MB for K=8 workers.
"""

import os
import pickle
from typing import Tuple


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
