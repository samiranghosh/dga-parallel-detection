"""
Levenshtein Boundary Tests
============================
Verifies that the overlapping chunk strategy correctly handles
Levenshtein distance at chunk boundaries.

Run: pytest tests/test_boundary.py -v
"""

import pytest
import numpy as np
from src.chunker import create_overlapping_chunks


class TestOverlappingChunks:
    """Verify chunk creation with 1-domain overlap."""

    def test_chunk_count(self):
        """Should produce exactly K chunks."""
        domains = [f"domain{i}" for i in range(100)]
        chunks = create_overlapping_chunks(domains, k=4)
        assert len(chunks) == 4

    def test_first_chunk_no_context(self):
        """First chunk should have context=None."""
        domains = [f"domain{i}" for i in range(100)]
        chunks = create_overlapping_chunks(domains, k=4)
        context, _ = chunks[0]
        assert context is None

    def test_subsequent_chunks_have_context(self):
        """Chunks 1+ should have a context domain from the previous chunk."""
        domains = [f"domain{i}" for i in range(100)]
        chunks = create_overlapping_chunks(domains, k=4)
        for i in range(1, len(chunks)):
            context, _ = chunks[i]
            assert context is not None

    def test_context_is_last_of_previous(self):
        """Context domain should be the last domain of the previous chunk."""
        domains = [f"d{i:04d}" for i in range(100)]
        chunks = create_overlapping_chunks(domains, k=4)
        for i in range(1, len(chunks)):
            context, _ = chunks[i]
            _, prev_domains = chunks[i - 1]
            assert context == prev_domains[-1]

    def test_no_domain_loss(self):
        """All domains should appear exactly once across all chunks."""
        domains = [f"domain{i}" for i in range(100)]
        chunks = create_overlapping_chunks(domains, k=4)
        all_domains = []
        for _, chunk_domains in chunks:
            all_domains.extend(chunk_domains)
        assert len(all_domains) == 100
        assert set(all_domains) == set(domains)

    def test_levenshtein_boundary_correctness(self):
        """Levenshtein at chunk boundary should match sequential result.

        This is the key correctness proof for the overlapping strategy.
        The overlapping context domain ensures that the first domain in
        each chunk computes Levenshtein against the correct predecessor,
        not against itself.
        """
        from src.features import calc_levenshtein, extract_all_sequential
        from src.parallel_engine import parallel_extract_features

        # Use a sorted domain list so Levenshtein is meaningful
        domains = sorted([
            "aardvark", "abalone", "banana", "bandana",
            "candle", "candy", "delta", "demon",
            "eagle", "earth", "falcon", "famous",
            "garlic", "gazelle", "hammer", "handle",
            "igloo", "image", "jacket", "jasmine",
        ])

        dictionary = {"band", "can", "del", "ear", "fam", "ham", "image", "jack"}
        ngram_table = {
            "aar": 0.001, "ard": 0.002, "dva": 0.001, "var": 0.002, "ark": 0.003,
            "aba": 0.002, "bal": 0.003, "alo": 0.002, "lon": 0.004, "one": 0.005,
            "ban": 0.005, "ana": 0.004, "nan": 0.003, "can": 0.006, "and": 0.007,
            "ndl": 0.002, "dle": 0.003, "del": 0.004, "elt": 0.002, "lta": 0.001,
            "dem": 0.003, "emo": 0.004, "mon": 0.005, "eag": 0.002, "agl": 0.001,
            "ear": 0.005, "art": 0.004, "rth": 0.003, "fal": 0.003, "alc": 0.002,
            "lco": 0.001, "con": 0.004, "fam": 0.003, "amo": 0.004, "mou": 0.003,
            "ous": 0.005, "gar": 0.003, "arl": 0.002, "rli": 0.001, "lic": 0.003,
            "gaz": 0.001, "aze": 0.001, "zel": 0.002, "ell": 0.004, "lle": 0.003,
            "ham": 0.004, "amm": 0.002, "mme": 0.001, "mer": 0.003, "han": 0.004,
            "igl": 0.001, "glo": 0.002, "loo": 0.003, "ima": 0.003, "mag": 0.002,
            "age": 0.004, "jac": 0.002, "ack": 0.003, "cke": 0.002, "ket": 0.003,
            "jas": 0.001, "asm": 0.001, "smi": 0.002, "min": 0.003, "ine": 0.004,
        }

        # Sequential extraction: ground truth
        seq = extract_all_sequential(domains, dictionary, ngram_table)

        # Parallel with different K values (2, 4, 5 to force uneven splits)
        for k in [2, 4, 5]:
            par = parallel_extract_features(domains, k, dictionary, ngram_table)

            # Check every Levenshtein value (column 5) matches
            assert np.allclose(seq[:, 5], par[:, 5], rtol=1e-10, atol=1e-10), (
                f"Levenshtein mismatch at K={k}:\n"
                f"  seq: {seq[:, 5]}\n"
                f"  par: {par[:, 5]}"
            )

            # Also verify all 6 features match
            assert np.allclose(seq, par, rtol=1e-10, atol=1e-10), (
                f"Full feature mismatch at K={k}"
            )
