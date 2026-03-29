#!/usr/bin/env python3
"""
tests/test_p4s2_callee.py

Phase 4 Story 2: Inter-Procedural Context (callee_summarizer.py + callee_cache.py).

6 mandatory tests:
  1. embed_callee returns 768-d tensor
  2. embed_callee uses cache on second call (not calling CodeBERT again)
  3. get_callee_context: CALL node with known callee -> returns (idx, embed)
  4. get_callee_context: non-CALL node -> not included
  5. get_callee_context: unknown callee -> not included (no crash)
  6. CalleeCache memory fallback: works without Redis connection

Additional tests:
  7. CalleeCache key format: "callee:" + sha256(code)
  8. CalleeCache set/get round-trip
  9. get_callee_context: empty callee_sources -> empty results
  10. get_callee_context: namespace-qualified name (e.g. "lib.sanitize") -> stripped
  11. prepare_batch_callee_context: multi-graph batch
  12. CalleeSummarizer with custom cache
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.scripts.model.callee_cache import CalleeCache
from training.scripts.model.callee_summarizer import CalleeSummarizer, CODEBERT_DIM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cache():
    """In-memory cache (no Redis)."""
    return CalleeCache(redis_url="redis://localhost:99999/0", use_memory_fallback=True)


@pytest.fixture(scope="module")
def summarizer(cache):
    """Shared CalleeSummarizer with memory cache."""
    return CalleeSummarizer(cache=cache, device="cpu")


SAMPLE_CALLEE_CODE = 'void sanitize(char *buf) { buf[0] = 0; }'
SAMPLE_CALLEE_CODE_2 = 'int parse_input(const char *s) { return atoi(s); }'

SAMPLE_CPG = {
    "nodes": [
        {"_label": "METHOD", "name": "main"},
        {"_label": "CALL", "name": "sanitize"},
        {"_label": "IDENTIFIER", "name": "buf"},
        {"_label": "CALL", "name": "unknown_func"},
        {"_label": "CALL", "name": "lib.parse_input"},
        {"_label": "LITERAL", "name": "0"},
    ]
}

CALLEE_SOURCES = {
    "sanitize": SAMPLE_CALLEE_CODE,
    "parse_input": SAMPLE_CALLEE_CODE_2,
}


# =========================================================================
# Test 1 -- embed_callee returns 768-d tensor
# =========================================================================

class TestEmbedCallee:
    def test_embed_returns_768d(self, summarizer):
        embed = summarizer.embed_callee(SAMPLE_CALLEE_CODE)
        assert embed.shape == (768,), f"Expected (768,), got {embed.shape}"
        assert embed.dtype == torch.float32
        assert not torch.isnan(embed).any()

    def test_embed_different_code_different_result(self, summarizer):
        e1 = summarizer.embed_callee(SAMPLE_CALLEE_CODE)
        e2 = summarizer.embed_callee(SAMPLE_CALLEE_CODE_2)
        assert not torch.allclose(e1, e2), "Different code should produce different embeddings"


# =========================================================================
# Test 2 -- embed_callee uses cache on second call
# =========================================================================

class TestEmbedCaching:
    def test_cache_hit_skips_codebert(self, summarizer):
        """Second call for same code must use cache, not call CodeBERT again."""
        code = 'void cached_func(int x) { return; }'

        calls_before = summarizer._call_count
        _ = summarizer.embed_callee(code)
        calls_after_first = summarizer._call_count
        assert calls_after_first == calls_before + 1, "First call should invoke CodeBERT"

        _ = summarizer.embed_callee(code)
        calls_after_second = summarizer._call_count
        assert calls_after_second == calls_after_first, \
            "Second call should use cache, not invoke CodeBERT"

    def test_cache_produces_same_embedding(self, summarizer):
        code = 'int add(int a, int b) { return a + b; }'
        e1 = summarizer.embed_callee(code)
        e2 = summarizer.embed_callee(code)
        assert torch.allclose(e1, e2, atol=1e-6), "Cache should return same embedding"


# =========================================================================
# Test 3 -- get_callee_context: CALL node with known callee
# =========================================================================

class TestGetCalleeContext:
    def test_call_node_known_callee(self, summarizer):
        results = summarizer.get_callee_context(SAMPLE_CPG, CALLEE_SOURCES)
        # Node 1 is CALL "sanitize" (known), node 4 is CALL "lib.parse_input" (namespace stripped)
        node_indices = [idx for idx, _ in results]
        assert 1 in node_indices, "CALL node 'sanitize' (idx=1) should be in results"

    def test_returned_embed_is_768d(self, summarizer):
        results = summarizer.get_callee_context(SAMPLE_CPG, CALLEE_SOURCES)
        for idx, embed in results:
            assert embed.shape == (768,), f"Node {idx}: expected (768,), got {embed.shape}"


# =========================================================================
# Test 4 -- get_callee_context: non-CALL node not included
# =========================================================================

class TestNonCallNodeExcluded:
    def test_method_node_excluded(self, summarizer):
        """METHOD node (idx=0) should never appear in results."""
        results = summarizer.get_callee_context(SAMPLE_CPG, CALLEE_SOURCES)
        node_indices = [idx for idx, _ in results]
        assert 0 not in node_indices, "METHOD node should not be in results"

    def test_identifier_node_excluded(self, summarizer):
        """IDENTIFIER node (idx=2) should never appear in results."""
        results = summarizer.get_callee_context(SAMPLE_CPG, CALLEE_SOURCES)
        node_indices = [idx for idx, _ in results]
        assert 2 not in node_indices, "IDENTIFIER node should not be in results"

    def test_literal_node_excluded(self, summarizer):
        results = summarizer.get_callee_context(SAMPLE_CPG, CALLEE_SOURCES)
        node_indices = [idx for idx, _ in results]
        assert 5 not in node_indices, "LITERAL node should not be in results"


# =========================================================================
# Test 5 -- get_callee_context: unknown callee not included (no crash)
# =========================================================================

class TestUnknownCalleeExcluded:
    def test_unknown_callee_not_in_results(self, summarizer):
        """CALL 'unknown_func' (idx=3) is not in callee_sources -> excluded."""
        results = summarizer.get_callee_context(SAMPLE_CPG, CALLEE_SOURCES)
        node_indices = [idx for idx, _ in results]
        assert 3 not in node_indices, "Unknown callee should not be in results"

    def test_no_crash_on_all_unknown(self, summarizer):
        """All CALL nodes have unknown callees -> empty results, no crash."""
        cpg = {"nodes": [
            {"_label": "CALL", "name": "totally_unknown"},
            {"_label": "CALL", "name": "also_unknown"},
        ]}
        results = summarizer.get_callee_context(cpg, CALLEE_SOURCES)
        assert results == []

    def test_empty_callee_sources(self, summarizer):
        results = summarizer.get_callee_context(SAMPLE_CPG, {})
        assert results == []

    def test_empty_nodes(self, summarizer):
        results = summarizer.get_callee_context({"nodes": []}, CALLEE_SOURCES)
        assert results == []


# =========================================================================
# Test 6 -- CalleeCache memory fallback works without Redis
# =========================================================================

class TestCalleeCacheMemoryFallback:
    def test_memory_fallback_creation(self):
        """Cache should be created without Redis, using memory fallback."""
        c = CalleeCache(redis_url="redis://localhost:99999/0", use_memory_fallback=True)
        assert not c.is_redis
        assert c.size() == 0

    def test_memory_fallback_set_get(self):
        c = CalleeCache(redis_url="redis://localhost:99999/0", use_memory_fallback=True)
        code = "int foo() { return 1; }"
        embed = torch.randn(768)
        c.set(code, embed)
        assert c.size() == 1

        retrieved = c.get(code)
        assert retrieved is not None
        assert torch.allclose(retrieved, embed)

    def test_memory_fallback_miss(self):
        c = CalleeCache(redis_url="redis://localhost:99999/0", use_memory_fallback=True)
        assert c.get("nonexistent code") is None

    def test_no_fallback_raises(self):
        """Without fallback, missing Redis should raise."""
        with pytest.raises(Exception):
            CalleeCache(redis_url="redis://localhost:99999/0", use_memory_fallback=False)


# =========================================================================
# Test 7 -- CalleeCache key format
# =========================================================================

class TestCalleeCacheKeyFormat:
    def test_key_format(self):
        code = "void f() {}"
        expected = "callee:" + hashlib.sha256(code.encode("utf-8")).hexdigest()
        assert CalleeCache._key(code) == expected

    def test_different_code_different_key(self):
        k1 = CalleeCache._key("void a() {}")
        k2 = CalleeCache._key("void b() {}")
        assert k1 != k2


# =========================================================================
# Test 8 -- CalleeCache set/get round-trip preserves values
# =========================================================================

class TestCalleeCacheRoundTrip:
    def test_roundtrip_preserves_tensor(self):
        c = CalleeCache(redis_url="redis://localhost:99999/0", use_memory_fallback=True)
        code = "int bar(int x) { return x * 2; }"
        embed = torch.randn(768)
        c.set(code, embed)
        got = c.get(code)
        assert torch.allclose(got, embed, atol=1e-6)

    def test_get_returns_clone(self):
        """get() should return a clone, not a reference to the stored tensor."""
        c = CalleeCache(redis_url="redis://localhost:99999/0", use_memory_fallback=True)
        code = "void clone_test() {}"
        embed = torch.randn(768)
        c.set(code, embed)
        got1 = c.get(code)
        got1[0] = 999.0  # mutate
        got2 = c.get(code)
        assert got2[0] != 999.0, "get() should return a clone"


# =========================================================================
# Test 9 -- Namespace-qualified callee name stripped
# =========================================================================

class TestNamespaceStripping:
    def test_dotted_name_stripped(self, summarizer):
        """'lib.parse_input' -> 'parse_input' -> matched in callee_sources."""
        results = summarizer.get_callee_context(SAMPLE_CPG, CALLEE_SOURCES)
        node_indices = [idx for idx, _ in results]
        # Node 4 has name "lib.parse_input" which should match "parse_input"
        assert 4 in node_indices, "Namespace-qualified CALL should match after stripping"


# =========================================================================
# Test 10 -- prepare_batch_callee_context
# =========================================================================

class TestBatchContext:
    def test_multi_graph_batch(self, summarizer):
        cpg1 = {"nodes": [{"_label": "CALL", "name": "sanitize"}]}
        cpg2 = {"nodes": [{"_label": "CALL", "name": "parse_input"}]}
        sources1 = {"sanitize": SAMPLE_CALLEE_CODE}
        sources2 = {"parse_input": SAMPLE_CALLEE_CODE_2}

        tuples = summarizer.prepare_batch_callee_context(
            [cpg1, cpg2], [sources1, sources2]
        )

        assert len(tuples) == 2
        # First graph's CALL node
        assert tuples[0][0] == 0  # graph_idx
        assert tuples[0][1] == 0  # node_idx
        assert tuples[0][2].shape == (768,)
        # Second graph's CALL node
        assert tuples[1][0] == 1
        assert tuples[1][1] == 0

    def test_batch_with_none_sources(self, summarizer):
        cpg1 = {"nodes": [{"_label": "CALL", "name": "sanitize"}]}
        cpg2 = {"nodes": [{"_label": "CALL", "name": "foo"}]}

        tuples = summarizer.prepare_batch_callee_context(
            [cpg1, cpg2],
            [{"sanitize": SAMPLE_CALLEE_CODE}, None],  # second has no sources
        )
        assert len(tuples) == 1
        assert tuples[0][0] == 0  # only first graph contributes


# =========================================================================
# Test 11 -- CODEBERT_DIM constant
# =========================================================================

class TestConstants:
    def test_codebert_dim(self):
        assert CODEBERT_DIM == 768
