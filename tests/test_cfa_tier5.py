#!/usr/bin/env python3
"""
tests/test_cfa_tier5.py

Story CFA-T5 — Tier 5: Critique-and-Refine Fallback.

8 tests covering:
  - Feedback content for specific failure reasons (T5-01 to T5-03)
  - tier5_refine: successful refinement of compile_fail and too_similar (T5-04, T5-05)
  - inject_feedback: appended section appears at end (T5-06)
  - Quality score is 0.85x base score (T5-07)
  - Returns None when refinement also fails (T5-08)

All LLM calls are mocked — no live API key required.
"""
from __future__ import annotations

import re
from unittest.mock import patch

import pytest

import training.scripts.preprocessing.cfa_tier5 as tier5_mod
from training.scripts.preprocessing.cfa_tier5 import (
    FEEDBACK_TEMPLATES,
    get_feedback,
    inject_feedback,
    tier5_refine,
)
from training.scripts.preprocessing.stage3_cfa import CfaConfig, validate_cfa_v2


# ── Shared fixtures ───────────────────────────────────────────────

# CWE-121: original has unguarded array write
CWE121_VULN = """\
void store_value(int idx, int val) {
    int arr[10];
    int temp = 0;
    arr[idx] = val;
    temp = arr[0];
}
"""

# A correct CWE-121 fix (bounds check)
CWE121_CFA_GOOD = """\
void store_value(int idx, int val) {
    int arr[10];
    int temp = 0;
    if (idx < 0 || idx >= 10) { return; }
    arr[idx] = val;
    temp = arr[0];
}
"""

# A CWE-89 vulnerable function
CWE89_VULN = """\
void search_user(sqlite3 *db, const char *username) {
    char query[256];
    int result = 0;
    fgets(query, sizeof(query), stdin);
    strcat(query, username);
    sqlite3_exec(db, query, 0, 0, 0);
    result = 1;
    (void)result;
}
"""

_DEFAULT_CFG = CfaConfig(use_tree_sitter_fallback=True)


# ═══════════════════════════════════════════════════════════════
# T5-01: compile_fail feedback contains actionable fix instructions
# ═══════════════════════════════════════════════════════════════

def test_t5_01_compile_fail_feedback_actionable():
    """T5-01: compile_fail feedback mentions all 4 syntax issues to check."""
    fb = get_feedback("CWE-121", "compile_fail")

    assert "brace" in fb.lower() or "braces" in fb.lower(), \
        "compile_fail feedback must mention brace matching"
    assert "semicolon" in fb.lower(), \
        "compile_fail feedback must mention semicolons"
    assert "variable" in fb.lower() or "declared" in fb.lower(), \
        "compile_fail feedback must mention variable declaration"
    assert "syntax" in fb.lower() or "valid C" in fb, \
        "compile_fail feedback must mention valid C syntax"


# ═══════════════════════════════════════════════════════════════
# T5-02: vuln_pattern_remains feedback names the specific pattern
# ═══════════════════════════════════════════════════════════════

def test_t5_02_vuln_pattern_feedback_names_pattern():
    """T5-02: vuln_pattern_remains feedback names the CWE-specific pattern."""
    # CWE-89: should mention SQL / sprintf / strcat / query
    fb_89 = get_feedback("CWE-89", "vuln_pattern_remains")
    assert "CWE-89" in fb_89
    assert any(kw in fb_89.lower() for kw in ["sql", "sprintf", "strcat", "query"]), \
        f"CWE-89 pattern feedback should name the pattern: {fb_89}"

    # CWE-120: should mention strcpy / strcat / gets / sprintf
    fb_120 = get_feedback("CWE-120", "vuln_pattern_remains")
    assert any(kw in fb_120.lower() for kw in ["strcpy", "strcat", "gets", "sprintf"]), \
        f"CWE-120 pattern feedback should name the pattern: {fb_120}"

    # CWE-416: should mention use-after-free or free/pointer
    fb_416 = get_feedback("CWE-416", "vuln_pattern_remains")
    assert any(kw in fb_416.lower() for kw in ["free", "pointer", "null"]), \
        f"CWE-416 pattern feedback should mention free/pointer: {fb_416}"


# ═══════════════════════════════════════════════════════════════
# T5-03: taint_path_intact feedback names the sink function
# ═══════════════════════════════════════════════════════════════

def test_t5_03_taint_feedback_names_sink():
    """T5-03: taint_path_intact feedback names the CWE-specific sink function."""
    # CWE-89: sink is sqlite3_exec / mysql_query
    fb_89 = get_feedback("CWE-89", "taint_path_intact")
    assert any(kw in fb_89.lower() for kw in ["sqlite3_exec", "mysql_query", "sql"]), \
        f"CWE-89 taint feedback must name the SQL sink: {fb_89}"

    # CWE-78: sink is system/popen/exec
    fb_78 = get_feedback("CWE-78", "taint_path_intact")
    assert any(kw in fb_78.lower() for kw in ["system", "popen", "exec"]), \
        f"CWE-78 taint feedback must name the shell sink: {fb_78}"

    # CWE-79: sink is printf/fprintf/HTTP response
    fb_79 = get_feedback("CWE-79", "taint_path_intact")
    assert any(kw in fb_79.lower() for kw in ["printf", "fprintf", "output"]), \
        f"CWE-79 taint feedback must name the output sink: {fb_79}"


# ═══════════════════════════════════════════════════════════════
# T5-04: tier5_refine with compile_fail input: output compiles
# ═══════════════════════════════════════════════════════════════

def test_t5_04_refine_compile_fail_produces_valid_cfa():
    """T5-04: tier5_refine called with compile_fail reason produces a valid CFA."""
    # Simulate a failed CFA (missing closing brace — won't compile)
    failed_cfa = (
        "void store_value(int idx, int val) {\n"
        "    int arr[10];\n"
        "    int temp = 0;\n"
        "    if (idx < 0 || idx >= 10) {\n"  # unclosed block
        "    arr[idx] = val;\n"
        "    temp = arr[0];\n"
        "}\n"
    )

    with patch.object(tier5_mod, "call_llm", return_value=CWE121_CFA_GOOD):
        result = tier5_refine(
            CWE121_VULN, failed_cfa, "CWE-121", "compile_fail", _DEFAULT_CFG
        )

    assert result is not None, "tier5_refine must return a result when LLM succeeds"
    assert "cfa_code" in result
    # Output must contain the fix
    assert re.search(r'if\s*\(', result["cfa_code"])


# ═══════════════════════════════════════════════════════════════
# T5-05: tier5_refine with too_similar input: output has more changes
# ═══════════════════════════════════════════════════════════════

def test_t5_05_refine_too_similar_produces_different_cfa():
    """T5-05: tier5_refine called with too_similar reason produces a more changed CFA."""
    # too_similar failed CFA — nearly identical to original
    too_similar_cfa = """\
void store_value(int idx, int val) {
    int arr[10];
    int temp = 0;
    arr[idx] = val;
    temp = arr[idx];
}
"""

    with patch.object(tier5_mod, "call_llm", return_value=CWE121_CFA_GOOD):
        result = tier5_refine(
            CWE121_VULN, too_similar_cfa, "CWE-121", "too_similar", _DEFAULT_CFG
        )

    assert result is not None
    cfa = result["cfa_code"]
    # The returned CFA must be more different than the failed one
    # (it has a bounds check the too_similar version lacked)
    assert "if" in cfa and ("idx < 0" in cfa or "idx >=" in cfa)


# ═══════════════════════════════════════════════════════════════
# T5-06: inject_feedback: appended section appears at end
# ═══════════════════════════════════════════════════════════════

def test_t5_06_inject_feedback_appends_at_end():
    """T5-06: inject_feedback appends 'PREVIOUS ATTEMPT NOTES:' at the end."""
    original_prompt = (
        "You are fixing CWE-121 in a C function.\n"
        "TASK: Add bounds check before array write.\n"
        "[FIXED_CODE_START]code_here[FIXED_CODE_END]"
    )

    result = inject_feedback(
        original_prompt,
        CWE121_VULN,
        "arr[idx] = val;",  # failed_cfa
        "CWE-121",
        "compile_fail",
    )

    # Original prompt content must still be present
    assert "You are fixing CWE-121" in result
    assert "[FIXED_CODE_START]" in result

    # Appended notes must appear AFTER original prompt
    notes_pos = result.find("PREVIOUS ATTEMPT NOTES:")
    assert notes_pos != -1, "inject_feedback must add 'PREVIOUS ATTEMPT NOTES:' section"
    original_end = result.find("[FIXED_CODE_END]")
    assert notes_pos > original_end, \
        "PREVIOUS ATTEMPT NOTES must appear after the original prompt content"


# ═══════════════════════════════════════════════════════════════
# T5-07: quality_score for Tier 5 output is 0.85x the base score
# ═══════════════════════════════════════════════════════════════

def test_t5_07_quality_score_is_discounted():
    """T5-07: Tier 5 result quality_score is 0.85x the validate_cfa_v2 base score."""
    with patch.object(tier5_mod, "call_llm", return_value=CWE121_CFA_GOOD):
        result = tier5_refine(
            CWE121_VULN, "", "CWE-121", "compile_fail", _DEFAULT_CFG
        )

    assert result is not None

    # Compute what the base score should be by calling validate_cfa_v2 directly
    _, _, _, base_score = validate_cfa_v2(
        CWE121_VULN, CWE121_CFA_GOOD, "CWE-121", tier=5, config=_DEFAULT_CFG
    )
    expected_quality = round(base_score * 0.85, 4)

    assert result["quality_score"] == pytest.approx(expected_quality, abs=1e-3), (
        f"Expected quality_score={expected_quality:.4f} "
        f"(base={base_score:.2f} * 0.85), got {result['quality_score']}"
    )
    assert result["tier"] == 5


# ═══════════════════════════════════════════════════════════════
# T5-08: Tier 5 returns None if refinement also fails (no infinite loop)
# ═══════════════════════════════════════════════════════════════

def test_t5_08_returns_none_on_failed_refinement():
    """T5-08: tier5_refine returns None if the refined CFA also fails validation."""
    # LLM returns None (API unavailable) — refinement cannot succeed
    with patch.object(tier5_mod, "call_llm", return_value=None):
        result = tier5_refine(
            CWE121_VULN, "", "CWE-121", "compile_fail", _DEFAULT_CFG
        )

    assert result is None, (
        "tier5_refine must return None when LLM fails — no retry or infinite loop"
    )

    # Also test: LLM returns something but it's identical to original (Gate 1 fails)
    with patch.object(tier5_mod, "call_llm", return_value=CWE121_VULN):
        result2 = tier5_refine(
            CWE121_VULN, "", "CWE-121", "too_similar", _DEFAULT_CFG
        )

    assert result2 is None, (
        "tier5_refine must return None when refined CFA is also invalid"
    )
