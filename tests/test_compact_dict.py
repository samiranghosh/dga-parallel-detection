import os
import pytest
from src.compact_dict import build_marisa_trie, CompactDictionary

@pytest.fixture
def sample_words():
    return ["google", "facebook", "amazon", "microsoft", "apple", "testword", "something"]

def test_compact_dictionary_builder(tmp_path, sample_words):
    trie_path = os.path.join(tmp_path, "test.marisa")
    build_marisa_trie(sample_words, trie_path)
    
    assert os.path.exists(trie_path)
    
    compact_dict = CompactDictionary(trie_path)
    assert len(compact_dict) == len(sample_words)
    
    for word in sample_words:
        assert word in compact_dict
        
    assert "notaword" not in compact_dict
    assert compact_dict.memory_usage_bytes() > 0
    
def test_compact_dictionary_no_file():
    cd = CompactDictionary()
    assert len(cd) == 0
    assert "google" not in cd
