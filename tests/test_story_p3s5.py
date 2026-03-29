#!/usr/bin/env python3
"""
tests/test_story_p3s5.py

P3-S5 tests for Stage 5 Node Embedding extensions:
  1. Feature shape assertion triggers on wrong-shaped input (824-d check)
  2. CFA sample (source='sard_cfa') embeds identically to structural equivalent
  3. per-CWE stats file created with correct CWE keys

Does NOT require CodeBERT model download — uses a mock encoder.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.scripts.preprocessing.stage5_embed import (
    CODEBERT_DIM,
    TOTAL_DIM,
    embed_cpg,
    run_stage5,
)


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def mock_encoder():
    """Mock CodeBERT encoder returning deterministic 768-d vectors."""
    encoder = MagicMock()

    def fake_encode(texts):
        result = np.zeros((len(texts), CODEBERT_DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = hash(text) % (2**31)
            rng = np.random.RandomState(seed)
            result[i] = rng.randn(CODEBERT_DIM).astype(np.float32) * 0.1
        return result

    encoder.encode_batch = fake_encode
    return encoder


def _make_cpg(sample_id: str, source: str, cwe: str = "CWE-121", label: int = 1):
    """Build a minimal valid CPG dict."""
    return {
        "sample_id": sample_id,
        "label": label,
        "cwe": cwe,
        "pair_id": "",
        "source": source,
        "nodes": [
            {"id": "1", "_label": "METHOD", "code": "void bad()", "name": "bad",
             "line": 1, "type_full": "void", "taint_role": "NONE"},
            {"id": "2", "_label": "CALL", "code": "gets(buf)", "name": "gets",
             "line": 3, "type_full": "ANY", "taint_role": "SOURCE"},
            {"id": "3", "_label": "IDENTIFIER", "code": "buf", "name": "buf",
             "line": 3, "type_full": "char*", "taint_role": "NONE"},
            {"id": "4", "_label": "CALL", "code": "strcpy(dst, buf)", "name": "strcpy",
             "line": 5, "type_full": "ANY", "taint_role": "SINK"},
            {"id": "5", "_label": "METHOD_RETURN", "code": "RET", "name": "",
             "line": 1, "type_full": "void", "taint_role": "NONE"},
        ],
        "edges": [
            {"src": "1", "dst": "2", "type": 0, "label": "AST"},
            {"src": "1", "dst": "3", "type": 0, "label": "AST"},
            {"src": "1", "dst": "4", "type": 0, "label": "AST"},
            {"src": "1", "dst": "5", "type": 0, "label": "AST"},
            {"src": "2", "dst": "3", "type": 1, "label": "CFG"},
            {"src": "3", "dst": "4", "type": 1, "label": "CFG"},
            {"src": "2", "dst": "3", "type": 2, "label": "DFG"},
            {"src": "3", "dst": "4", "type": 2, "label": "DFG"},
            {"src": "2", "dst": "4", "type": 3, "label": "TPG"},
        ],
        "stats": {"node_count": 5, "edge_count": 9},
    }


# ══════════════════════════════════════════════════════════════════
# Test 1: Feature shape assertion on wrong-shaped input
# ══════════════════════════════════════════════════════════════════

class TestFeatureShapeAssertion:
    """C2: The strict 824-d column assertion fires on wrong shapes."""

    def test_correct_shape_passes(self, mock_encoder):
        """Normal CPG produces (N, 824) without error."""
        cpg = _make_cpg("test-001", "sard")
        features = embed_cpg(cpg, mock_encoder)
        assert features.shape == (5, 824)
        assert features.shape[1] == TOTAL_DIM

    def test_assertion_on_wrong_feature_dim(self, tmp_path, mock_encoder):
        """run_stage5 rejects a sample if embed_cpg somehow produces wrong columns.

        We simulate this by monkeypatching embed_cpg to return 512-d features,
        which should trigger the assert in run_stage5.
        """
        # Write a valid CPG file
        cpg = _make_cpg("aa001234", "sard", cwe="CWE-121")
        shard_dir = tmp_path / "input" / "aa"
        shard_dir.mkdir(parents=True)
        (shard_dir / "aa001234.json").write_text(json.dumps(cpg), encoding="utf-8")

        out_dir = tmp_path / "output"

        # Monkeypatch embed_cpg to return wrong dimension
        import training.scripts.preprocessing.stage5_embed as mod

        original_embed = mod.embed_cpg

        def bad_embed(cpg_data, enc, batch_size=64):
            n = len(cpg_data["nodes"])
            return np.random.randn(n, 512).astype(np.float32)

        mod.embed_cpg = bad_embed
        try:
            with pytest.raises(AssertionError, match="Wrong feature dim"):
                run_stage5(
                    input_dir=str(tmp_path / "input"),
                    output_dir=str(out_dir),
                    device="cpu",
                )
        finally:
            mod.embed_cpg = original_embed


# ══════════════════════════════════════════════════════════════════
# Test 2: CFA sample embeds identically to structural equivalent
# ══════════════════════════════════════════════════════════════════

class TestCFAEmbedding:
    """C1: CFA samples (source ending '_cfa') embed via the same code path."""

    def test_cfa_embeds_same_as_original(self, mock_encoder):
        """Two structurally identical CPGs with different sources produce
        identical feature matrices (source field doesn't affect embedding)."""
        cpg_original = _make_cpg("test-orig", "sard", cwe="CWE-121")
        cpg_cfa = _make_cpg("test-cfa", "sard_cfa", cwe="CWE-121")

        feat_original = embed_cpg(cpg_original, mock_encoder)
        feat_cfa = embed_cpg(cpg_cfa, mock_encoder)

        np.testing.assert_array_equal(
            feat_original, feat_cfa,
            err_msg="CFA and original samples with identical structure must "
                    "produce identical embeddings"
        )


# ══════════════════════════════════════════════════════════════════
# Test 3: Per-CWE stats file created with correct keys
# ══════════════════════════════════════════════════════════════════

class TestPerCWEStats:
    """C3: embed_stats_by_cwe.json is created with expected CWE entries."""

    def test_cwe_stats_file_created(self, tmp_path, mock_encoder):
        """run_stage5 creates embed_stats_by_cwe.json with per-CWE entries."""
        # Create CPG files with two CWEs
        for cwe, sid in [("CWE-121", "aa001111"), ("CWE-122", "bb002222")]:
            cpg = _make_cpg(sid, "sard", cwe=cwe)
            shard_dir = tmp_path / "input" / sid[:2]
            shard_dir.mkdir(parents=True, exist_ok=True)
            (shard_dir / f"{sid}.json").write_text(json.dumps(cpg), encoding="utf-8")

        out_dir = tmp_path / "output"

        # Monkeypatch CodeBERTEncoder to avoid loading real model
        import training.scripts.preprocessing.stage5_embed as mod

        original_init = mod.CodeBERTEncoder.__init__

        def mock_init(self, model_name=None, device=None):
            self.device = device or "cpu"
            self.tokenizer = None
            self.model = None

        original_encode = mod.CodeBERTEncoder.encode_batch

        def mock_encode_method(self, texts):
            return mock_encoder.encode_batch(texts)

        mod.CodeBERTEncoder.__init__ = mock_init
        mod.CodeBERTEncoder.encode_batch = mock_encode_method
        try:
            stats = run_stage5(
                input_dir=str(tmp_path / "input"),
                output_dir=str(out_dir),
                device="cpu",
            )
        finally:
            mod.CodeBERTEncoder.__init__ = original_init
            mod.CodeBERTEncoder.encode_batch = original_encode

        # Verify embed_stats_by_cwe.json
        cwe_stats_path = out_dir / "embed_stats_by_cwe.json"
        assert cwe_stats_path.exists(), "embed_stats_by_cwe.json not created"

        cwe_stats = json.loads(cwe_stats_path.read_text(encoding="utf-8"))
        assert "CWE-121" in cwe_stats
        assert "CWE-122" in cwe_stats

        for cwe_key in ("CWE-121", "CWE-122"):
            entry = cwe_stats[cwe_key]
            assert "sample_count" in entry
            assert "avg_node_count" in entry
            assert "avg_taint_node_pct" in entry
            assert "avg_dfg_degree" in entry
            assert entry["sample_count"] == 1
            assert entry["avg_node_count"] == 5.0  # 5 nodes in our fixture

    def test_cfa_source_stats_tracked(self, tmp_path, mock_encoder):
        """run_stage5 stats include cfa_samples and original_samples counts."""
        # One original, one CFA
        for source, sid in [("sard", "cc003333"), ("sard_cfa", "dd004444")]:
            cpg = _make_cpg(sid, source, cwe="CWE-121")
            shard_dir = tmp_path / "input" / sid[:2]
            shard_dir.mkdir(parents=True, exist_ok=True)
            (shard_dir / f"{sid}.json").write_text(json.dumps(cpg), encoding="utf-8")

        out_dir = tmp_path / "output"

        import training.scripts.preprocessing.stage5_embed as mod

        original_init = mod.CodeBERTEncoder.__init__
        original_encode = mod.CodeBERTEncoder.encode_batch

        def mock_init(self, model_name=None, device=None):
            self.device = device or "cpu"
            self.tokenizer = None
            self.model = None

        def mock_encode_method(self, texts):
            return mock_encoder.encode_batch(texts)

        mod.CodeBERTEncoder.__init__ = mock_init
        mod.CodeBERTEncoder.encode_batch = mock_encode_method
        try:
            stats = run_stage5(
                input_dir=str(tmp_path / "input"),
                output_dir=str(out_dir),
                device="cpu",
            )
        finally:
            mod.CodeBERTEncoder.__init__ = original_init
            mod.CodeBERTEncoder.encode_batch = original_encode

        assert stats["cfa_samples"] == 1
        assert stats["original_samples"] == 1
        assert stats["succeeded"] == 2
