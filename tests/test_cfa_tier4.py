#!/usr/bin/env python3
"""
tests/test_cfa_tier4.py

Story CFA-T4 — Tier 4: Few-Shot Exemplar + CoT CFA Generation.

10 tests covering:
  - ExemplarDatabase build, selection, save/load (T4-01 to T4-04)
  - build_tier4_prompt content checks (T4-05)
  - validate_cwe416_structural (T4-06)
  - tier4_generate: CWE-416 accepted (T4-07), structural check rejection (T4-08)
  - tier4_generate: CWE-119 accepted (T4-09)
  - max_attempts respected (T4-10)

All LLM calls are mocked — no live API key required.
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import training.scripts.preprocessing.cfa_tier4 as tier4_mod
from training.scripts.preprocessing.cfa_exemplar_db import ExemplarDatabase
from training.scripts.preprocessing.cfa_tier4 import (
    TIER4_COT_PROMPTS,
    build_tier4_prompt,
    extract_c_code_tier4,
    tier4_generate,
    validate_cwe416_structural,
)
from training.scripts.preprocessing.stage3_cfa import CfaConfig


# ── Shared fixtures ──────────────────────────────────────────────

CWE416_VULN = """\
void process(int *data, int size) {
    int *ptr = (int *)malloc(size * sizeof(int));
    ptr[0] = 42;
    free(ptr);
    int val = ptr[0];
    (void)val;
}
"""

CWE416_CFA_GOOD = """\
void process(int *data, int size) {
    int *ptr = (int *)malloc(size * sizeof(int));
    ptr[0] = 42;
    int val = ptr[0];
    free(ptr);
    ptr = NULL;
    (void)val;
}
"""

# CFA that passes Gate 4 (no vuln pattern) but fails structural check
# — free is still present, no null-after-free, no guard
CWE416_CFA_NO_STRUCT_FIX = """\
void process(int *data, int size) {
    int *ptr = (int *)malloc(size * sizeof(int));
    ptr[0] = 42;
    free(ptr);
    int val = data[0];
    (void)val;
}
"""

CWE119_VULN = """\
void copy_buf(char *src, int src_len) {
    char buf[64];
    int written = 0;
    memcpy(buf, src, src_len);
    written = src_len;
    (void)written;
}
"""

CWE119_CFA_GOOD = """\
void copy_buf(char *src, int src_len) {
    char buf[64];
    int written = 0;
    int safe_len = src_len;
    if (safe_len > (int)sizeof(buf)) { safe_len = (int)sizeof(buf); }
    memcpy(buf, src, safe_len);
    written = safe_len;
    (void)written;
}
"""

_DEFAULT_CFG = CfaConfig(use_tree_sitter_fallback=True)


# ── T4-01: ExemplarDatabase.build_from_pairs selects real pairs ──

def test_t4_01_build_from_pairs_selects_real():
    """T4-01: build_from_pairs picks real-world pairs + builtin exemplars."""
    samples = [
        # Real pair for CWE-416
        {"id": "a", "source": "cve", "cwe": "CWE-416", "label": 1,
         "pair_id": "p1", "code": CWE416_VULN, "language": "c",
         "collected_at": "2025-01-01"},
        {"id": "b", "source": "cve", "cwe": "CWE-416", "label": 0,
         "pair_id": "p1", "code": CWE416_CFA_GOOD, "language": "c",
         "collected_at": "2025-01-01"},
        # SARD pair — should be ignored by data scan (but builtins still merge)
        {"id": "c", "source": "sard", "cwe": "CWE-416", "label": 1,
         "pair_id": "p2", "code": CWE416_VULN, "language": "c",
         "collected_at": "2025-01-01"},
        {"id": "d", "source": "sard", "cwe": "CWE-416", "label": 0,
         "pair_id": "p2", "code": CWE416_CFA_GOOD, "language": "c",
         "collected_at": "2025-01-01"},
        # Real pair for CWE-119
        {"id": "e", "source": "exploitdb", "cwe": "CWE-119", "label": 1,
         "pair_id": "p3", "code": CWE119_VULN, "language": "c",
         "collected_at": "2025-01-01"},
        {"id": "f", "source": "exploitdb", "cwe": "CWE-119", "label": 0,
         "pair_id": "p3", "code": CWE119_CFA_GOOD, "language": "c",
         "collected_at": "2025-01-01"},
    ]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
        fname = f.name

    db = ExemplarDatabase()
    db.build_from_pairs(fname, max_per_cwe=8, min_real_pairs=1)

    assert "CWE-416" in db
    assert "CWE-119" in db
    # Data pair + 4 builtin exemplars for CWE-416
    assert len(db.get("CWE-416")) >= 5
    # Data pair + 4 builtin exemplars for CWE-119
    assert len(db.get("CWE-119")) >= 5
    # The real data pair must be present (vuln code matches fixture)
    cwe416_vulns = [ex["vuln"] for ex in db.get("CWE-416")]
    assert any(CWE416_VULN.strip() in v for v in cwe416_vulns)


# ── T4-02: ExemplarDatabase.select_exemplars ranks by keyword overlap ──

def test_t4_02_select_exemplars_keyword_ranking():
    """T4-02: select_exemplars returns exemplars ranked by keyword overlap."""
    db = ExemplarDatabase()
    # Manually populate
    db._db["CWE-416"] = [
        {"vuln": "void foo() { int *ptr = malloc(4); free(ptr); use(ptr); }",
         "safe": "void foo() { int *ptr = malloc(4); use(ptr); free(ptr); ptr = NULL; }"},
        {"vuln": "void bar() { char *buf = strdup(s); free(buf); printf(buf); }",
         "safe": "void bar() { char *buf = strdup(s); printf(buf); free(buf); buf = NULL; }"},
    ]

    # Query similar to first exemplar (uses ptr, malloc, free)
    query = "void process() { int *ptr = malloc(8); free(ptr); int x = ptr[0]; }"
    results = db.select_exemplars("CWE-416", query, n=1)
    assert len(results) == 1
    # First result should contain ptr/malloc (higher overlap with query)
    assert "ptr" in results[0]["vuln"]


# ── T4-03: ExemplarDatabase save and load round-trip ──────────────

def test_t4_03_save_load_round_trip():
    """T4-03: save() then load() preserves data + merges builtins."""
    db = ExemplarDatabase()
    db._db["CWE-416"] = [
        {"vuln": CWE416_VULN, "safe": CWE416_CFA_GOOD}
    ]
    db._db["CWE-119"] = [
        {"vuln": CWE119_VULN, "safe": CWE119_CFA_GOOD}
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "exemplar_db.json")
        db.save(path)

        loaded = ExemplarDatabase.load(path)

    # Original data pair + 4 builtins merged on load
    assert len(loaded.get("CWE-416")) >= 5
    assert len(loaded.get("CWE-119")) >= 5
    # Original data pair must survive the round-trip
    cwe416_vulns = [ex["vuln"] for ex in loaded.get("CWE-416")]
    assert any(CWE416_VULN.strip() in v for v in cwe416_vulns)
    cwe119_safes = [ex["safe"] for ex in loaded.get("CWE-119")]
    assert any(CWE119_CFA_GOOD.strip() in s for s in cwe119_safes)


# ── T4-04: build_from_pairs raises on insufficient real pairs ──────

def test_t4_04_builtin_exemplars_prevent_empty_cwe():
    """T4-04: builtin exemplars ensure CWE-119 is populated even without data pairs."""
    # Only CWE-416 data pairs — CWE-119 has no data pairs, but builtins fill the gap
    samples = [
        {"id": "a", "source": "cve", "cwe": "CWE-416", "label": 1,
         "pair_id": "p1", "code": CWE416_VULN, "language": "c",
         "collected_at": "2025-01-01"},
        {"id": "b", "source": "cve", "cwe": "CWE-416", "label": 0,
         "pair_id": "p1", "code": CWE416_CFA_GOOD, "language": "c",
         "collected_at": "2025-01-01"},
    ]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
        fname = f.name

    db = ExemplarDatabase()
    # Should NOT raise — builtin exemplars satisfy the minimum
    db.build_from_pairs(fname, max_per_cwe=8, min_real_pairs=1)
    assert "CWE-119" in db
    assert len(db.get("CWE-119")) >= 4  # 4 builtin CWE-119 exemplars


# ── T4-05: build_tier4_prompt includes exemplars and CoT steps ────

def test_t4_05_build_tier4_prompt_content():
    """T4-05: build_tier4_prompt embeds exemplar pair and CoT step instructions."""
    exemplars = [
        {"vuln": "void bad() { free(p); p->x = 1; }",
         "safe": "void bad() { p->x = 1; free(p); p = NULL; }"},
    ]
    prompt = build_tier4_prompt(CWE416_VULN, "CWE-416", exemplars)

    # Exemplar content must appear
    assert "void bad()" in prompt
    assert "free(p)" in prompt
    assert "p = NULL" in prompt

    # CoT steps must appear
    assert "STEP 1" in prompt
    assert "STEP 2" in prompt
    assert "STEP 3" in prompt

    # Markers must appear
    assert "[FIXED_CODE_START]" in prompt
    assert "[FIXED_CODE_END]" in prompt

    # Target function must appear
    assert "void process" in prompt


# ── T4-06: validate_cwe416_structural ─────────────────────────────

def test_t4_06_validate_cwe416_structural():
    """T4-06: structural validator accepts null-after-free, rejects no-fix."""
    # Good: ptr = NULL after free (Option A)
    good = "void f() { free(ptr); ptr = NULL; }"
    assert validate_cwe416_structural(CWE416_VULN, good) is True

    # Good: fewer free() calls — original has 2 frees, CFA removes one (Option B)
    double_free_original = (
        "void f() { int *ptr = malloc(4); free(ptr); free(ptr); }"
    )
    fewer_frees = "void f() { int *ptr = malloc(4); int v = ptr[0]; free(ptr); }"
    assert validate_cwe416_structural(double_free_original, fewer_frees) is True

    # Good: NULL guard on use (Option C)
    guarded = "void f() { if (ptr != NULL) { use(ptr); } free(ptr); }"
    assert validate_cwe416_structural(CWE416_VULN, guarded) is True

    # Bad: still has free, no null assignment, no guard
    bad = "void f() { free(ptr); int x = 0; (void)x; }"
    assert validate_cwe416_structural(CWE416_VULN, bad) is False


# ── T4-07: tier4_generate CWE-416 accepted ────────────────────────

def test_t4_07_tier4_generate_cwe416_accepted():
    """T4-07: CWE-416 CFA with null-after-free is accepted by all gates."""
    exemplar_db = MagicMock()
    exemplar_db.select_exemplars.return_value = []

    with patch.object(tier4_mod, "call_llm", return_value=CWE416_CFA_GOOD):
        results = tier4_generate(
            CWE416_VULN, "CWE-416",
            config=_DEFAULT_CFG,
            exemplar_db=exemplar_db,
        )

    assert len(results) >= 1
    cfa = results[0]
    # Null-after-free must be present
    assert "ptr = NULL" in cfa
    # free() must still exist (deferred, not removed)
    assert "free(ptr)" in cfa


# ── T4-08: structural check rejects CWE-416 with no fix ───────────

def test_t4_08_structural_check_rejects_no_fix():
    """T4-08: CWE-416 CFA that doesn't fix UAF is rejected by structural check."""
    exemplar_db = MagicMock()
    exemplar_db.select_exemplars.return_value = []

    # This CFA changes ptr[0] read to data[0] but keeps free(ptr) in same order
    # — passes 7 gates (different from original, compiles, no strcpy/gets) but
    # structural validator must reject it (free still present, no null, no guard).
    with patch.object(tier4_mod, "call_llm", return_value=CWE416_CFA_NO_STRUCT_FIX):
        cfg = CfaConfig(
            use_tree_sitter_fallback=True,
            max_attempts_tier4=2,
        )
        results = tier4_generate(
            CWE416_VULN, "CWE-416",
            config=cfg,
            exemplar_db=exemplar_db,
        )

    # Structural check should prevent acceptance
    assert len(results) == 0


# ── T4-09: tier4_generate CWE-119 accepted ────────────────────────

def test_t4_09_tier4_generate_cwe119_accepted():
    """T4-09: CWE-119 CFA with size check before memcpy is accepted."""
    exemplar_db = MagicMock()
    exemplar_db.select_exemplars.return_value = []

    with patch.object(tier4_mod, "call_llm", return_value=CWE119_CFA_GOOD):
        results = tier4_generate(
            CWE119_VULN, "CWE-119",
            config=_DEFAULT_CFG,
            exemplar_db=exemplar_db,
        )

    assert len(results) >= 1
    cfa = results[0]
    # Size check must cap the copy length
    assert re.search(r'if\s*\(', cfa)
    assert "memcpy" in cfa


# ── T4-10: max_attempts_tier4 respected ──────────────────────────

def test_t4_10_max_attempts_respected():
    """T4-10: call_llm is called at most max_attempts_tier4 times."""
    call_count = []

    def counting_llm(*args, **kwargs):
        call_count.append(1)
        return None  # always fails

    exemplar_db = MagicMock()
    exemplar_db.select_exemplars.return_value = []

    with patch.object(tier4_mod, "call_llm", side_effect=counting_llm):
        cfg = CfaConfig(max_attempts_tier4=5)
        tier4_generate(CWE416_VULN, "CWE-416", config=cfg, exemplar_db=exemplar_db)

    assert len(call_count) <= 5, (
        f"Expected at most 5 LLM calls, got {len(call_count)}"
    )
