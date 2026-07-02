import os
import marisa_trie
import logging
from typing import Iterable, Union

logger = logging.getLogger(__name__)

def build_marisa_trie(words: Iterable[str], output_path: str):
    """Builds a MARISA trie from an iterable of words and saves it to disk."""
    logger.info(f"Building MARISA trie for {len(words) if hasattr(words, '__len__') else 'unknown number of'} words...")
    trie = marisa_trie.Trie(words)
    trie.save(output_path)
    logger.info(f"Saved MARISA trie to {output_path}")

class CompactDictionary:
    """A memory-efficient dictionary using marisa-trie.
    
    This replaces the Python set() for O(m) substring lookups.
    Because marisa_trie is implemented in C++ and uses memory mapping, 
    it provides true zero-copy shared memory across multiprocessing workers
    without the CPython reference counting Copy-on-Write memory bloat trap.
    """
    
    def __init__(self, trie_path: str = None):
        self.trie = marisa_trie.Trie()
        self.trie_path = trie_path
        if trie_path and os.path.exists(trie_path):
            self.load(trie_path)
            
    def load(self, trie_path: str):
        """Loads (mmap) the trie from disk."""
        self.trie_path = trie_path
        self.trie.load(trie_path)
        
    def __contains__(self, word: str) -> bool:
        """Transparent replacement for set.__contains__"""
        return word in self.trie
        
    def __len__(self) -> int:
        return len(self.trie)
        
    def memory_usage_bytes(self) -> int:
        """Returns the size of the trie in bytes (mostly mmapped)."""
        if self.trie_path and os.path.exists(self.trie_path):
            return os.path.getsize(self.trie_path)
        return 0
