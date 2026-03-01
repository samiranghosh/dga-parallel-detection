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
        """
        # TODO: Implement after features.py is done
        # 1. Compute Levenshtein sequentially for all domains
        # 2. Compute using overlapping chunks
        # 3. Assert equal at boundary positions
        pytest.skip("Implement after Phase 5-6")
