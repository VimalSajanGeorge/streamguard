# P4-S2: Inter-Procedural Context (`callee_summarizer.py` + `callee_cache.py`)

**Status:** COMPLETE | **Date:** 2026-03-29 | **Tests:** 25/25 PASS

---

## What Was Built

Two new files for Config E ablation: inter-procedural callee context injection.

| File | Purpose |
|------|---------|
| `training/scripts/model/callee_cache.py` | Redis-backed cache (memory fallback) for CodeBERT callee embeddings |
| `training/scripts/model/callee_summarizer.py` | Embed callee functions, map to CALL nodes in CPG |
| `tests/test_p4s2_callee.py` | 25 tests |

## Architecture

```
CalleeSummarizer:
  1. For each CALL node in CPG: extract callee function name
  2. Look up callee source code:
     - Cache hit → return 768-d tensor (skip CodeBERT)
     - Cache miss → CodeBERT([CLS]) on callee body → 768-d → cache
     - Callee body unavailable → not included (zero-vector fallback in model.py)
  3. Return: [(local_node_idx, embed_768d), ...]
     → Passed to model.forward(callee_node_indices=..., callee_embeddings=...)

CalleeCache:
  Backend: Redis (persistent) or in-memory dict (fallback)
  Key:     "callee:" + sha256(callee_source_code)
  Value:   768-d float32 tensor (JSON-serialized for Redis, direct tensor for memory)
  TTL:     None (permanent — callee embeds don't change during training)
```

## How Config E Uses This

```python
# Config E training loop (P4-S5):
summarizer = CalleeSummarizer(cache=CalleeCache(), device=device)

for batch in dataloader:
    # Get callee context for this batch
    callee_tuples = summarizer.prepare_batch_callee_context(
        batch_cpg_jsons, batch_callee_sources
    )

    # Forward pass with inter-proc context
    out = model(
        batch,
        input_ids=...,
        callee_embeddings=callee_embeddings,  # (B, max_callees, 768)
        callee_node_indices=callee_tuples,     # [(graph_idx, node_idx, embed)]
    )

# Configs A-D: callee_embeddings=None → model handles gracefully
```

## CalleeCache Details

| Feature | Implementation |
|---------|---------------|
| Redis backend | `redis.Redis.from_url()` with 2s timeout |
| Memory fallback | Automatic when Redis unreachable + `use_memory_fallback=True` |
| Key format | `"callee:" + sha256(utf-8 encoded code)` |
| Serialization | `json.dumps(tensor.tolist())` for Redis; direct tensor for memory |
| Clone safety | `get()` returns cloned tensor (no mutation of cached data) |
| No TTL | Permanent cache — callee embeddings are deterministic |

## CalleeSummarizer Details

| Feature | Implementation |
|---------|---------------|
| Embedding | CodeBERT [CLS] vector (768-d) via `embed_callee()` |
| Caching | First call computes + caches; subsequent calls use cache |
| CALL detection | `node["_label"] == "CALL"` in CPG JSON |
| Name resolution | Namespace stripping: `"lib.sanitize"` → `"sanitize"` |
| Unknown callees | Silently skipped (no crash, no zero vector) |
| Batch API | `prepare_batch_callee_context()` returns `(graph_idx, node_idx, embed)` tuples |
| Diagnostics | `_call_count` tracks CodeBERT forward passes |

## Risk Mitigations

| Risk | Status | Evidence |
|------|--------|----------|
| R-24 (Config E crashes on None callees) | MITIGATED | model.py zero-pads callee_ctx when absent; summarizer returns empty list for empty/None sources; tested in both P4-S1 and P4-S2 |
| R-30 (CalleeSummarizer not initialized) | ADDRESSED | If not instantiated, model gets callee_embeddings=None → zero-vector fallback → same result as Config D (no inter-proc signal). No crash. |

## Integration with P4-S1 model.py

The P4-S1 model already has:
- `use_interproc=True` creates `interproc_proj` Linear(768, 256)
- `_prepare_interproc_features()` converts tuples to node masks + projected features
- `encode_graph()` injects callee features at call-site nodes via additive residual
- `forward()` concatenates callee context to fused vector (1280 → 1536)
- Zero-pad fallback when callee_embeddings is None

## Test Summary

```
tests/test_p4s2_callee.py — 25/25 PASS

TestEmbedCallee:                2 tests (768-d shape, different code → different embed)
TestEmbedCaching:               2 tests (cache hit skips CodeBERT, same embedding)
TestGetCalleeContext:           2 tests (known CALL matched, 768-d returned)
TestNonCallNodeExcluded:        3 tests (METHOD, IDENTIFIER, LITERAL excluded)
TestUnknownCalleeExcluded:      4 tests (unknown excluded, all unknown, empty sources, empty nodes)
TestCalleeCacheMemoryFallback:  4 tests (creation, set/get, miss, no-fallback raises)
TestCalleeCacheKeyFormat:       2 tests (format, uniqueness)
TestCalleeCacheRoundTrip:       2 tests (preserves values, returns clone)
TestNamespaceStripping:         1 test  (dotted name stripped)
TestBatchContext:               2 tests (multi-graph batch, None sources skipped)
TestConstants:                  1 test  (CODEBERT_DIM == 768)
```

## Combined Test Results (P4-S1 + P4-S2)

```
tests/test_p4s1_model.py  — 37/37 PASS (0 regressions)
tests/test_p4s2_callee.py — 25/25 PASS
Total: 62/62 PASS
```
