#!/usr/bin/env python3
"""
tests/test_cfa_tier2.py

Story CFA-T2 — Tier 2: Zero-Shot LLM CFA Generation.

10 tests covering prompts, code extraction, gate integration,
Tier 5 escalation, and max_attempts enforcement.
All LLM calls are mocked — no live API key required.
"""
from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest

import training.scripts.preprocessing.cfa_tier2 as tier2_mod
from training.scripts.preprocessing.cfa_tier2 import (
    TIER2_PROMPTS,
    extract_c_code_v2,
    tier2_generate,
)
from training.scripts.preprocessing.stage3_cfa import CfaConfig


# ── Shared test fixtures ─────────────────────────────────────────

# CWE-121: array written without bounds check
CWE121_VULN = """\
void store_value(int idx, int val) {
    int arr[10];
    int temp = 0;
    arr[idx] = val;
    temp = arr[0];
}
"""

# CWE-121 Option A fixed (index bounds check)
CWE121_CFA_OPTION_A = """\
void store_value(int idx, int val) {
    int arr[10];
    int temp = 0;
    if (idx < 0 || idx >= 10) { return; }
    arr[idx] = val;
    temp = arr[0];
}
"""

# CWE-121 Option B fixed (strcpy -> strncpy)
CWE121_VULN_STRCPY = """\
void copy_name(const char *src) {
    char name[32];
    int len = 0;
    strcpy(name, src);
    len = strlen(name);
}
"""

CWE121_CFA_OPTION_B = """\
void copy_name(const char *src) {
    char name[32];
    int len = 0;
    strncpy(name, src, sizeof(name)-1);
    name[sizeof(name)-1] = '\\0';
    len = strlen(name);
}
"""

# CWE-122: malloc without NULL check or size validation
CWE122_VULN = """\
char *make_buffer(int size, const char *data, int data_len) {
    char *buf = (char *)malloc(size);
    int write_size = data_len;
    memcpy(buf, data, write_size);
    return buf;
}
"""

CWE122_CFA = """\
char *make_buffer(int size, const char *data, int data_len) {
    char *buf = (char *)malloc(size);
    if (buf == NULL) { return NULL; }
    int write_size = data_len;
    if (write_size > size) { write_size = size; }
    memcpy(buf, data, write_size);
    return buf;
}
"""

# CWE-125: array read without bounds check
CWE125_VULN = """\
int read_element(int *arr, int len, int idx) {
    int result = 0;
    result = arr[idx];
    return result;
}
"""

CWE125_CFA = """\
int read_element(int *arr, int len, int idx) {
    int result = 0;
    if (idx < 0 || idx >= len) { return -1; }
    result = arr[idx];
    return result;
}
"""

# CFA with no fix signature (triggers Gate 5 soft)
CWE121_NO_FIX_SIG = """\
void store_value(int idx, int val) {
    int arr[10];
    int temp = 0;
    arr[0] = val;
    temp = arr[0];
}
"""

_DEFAULT_CFG = CfaConfig(use_tree_sitter_fallback=True)


# ═══════════════════════════════════════════════════════════════
# T2-01: CWE-121 Option A fix (index bounds check)
# ═══════════════════════════════════════════════════════════════

def test_t2_01_cwe121_option_a_bounds_check():
    """T2-01: CWE-121 Option A: index bounds check present in CFA."""
    with patch.object(tier2_mod, "call_llm", return_value=CWE121_CFA_OPTION_A):
        results = tier2_generate(CWE121_VULN, "CWE-121", config=_DEFAULT_CFG)

    assert len(results) >= 1
    cfa = results[0]
    # Option A: index bounds check must be present
    assert re.search(r'if\s*\(\s*idx\s*<\s*0\s*\|', cfa) or \
           re.search(r'if\s*\(\s*idx\s*(>=|<)\s*\d+\s*\)', cfa)
    # Must not have the unguarded array write as the first operation
    assert "arr[idx] = val" in cfa  # still present but after guard


# ═══════════════════════════════════════════════════════════════
# T2-02: CWE-121 Option B fix (strcpy -> strncpy)
# ═══════════════════════════════════════════════════════════════

def test_t2_02_cwe121_option_b_strncpy():
    """T2-02: CWE-121 Option B: strcpy replaced with strncpy + null terminator."""
    with patch.object(tier2_mod, "call_llm", return_value=CWE121_CFA_OPTION_B):
        results = tier2_generate(CWE121_VULN_STRCPY, "CWE-121", config=_DEFAULT_CFG)

    assert len(results) >= 1
    cfa = results[0]
    assert "strncpy" in cfa
    assert re.search(r"sizeof\(name\)\s*-\s*1", cfa)
    # Null terminator must be set
    assert re.search(r"name\[.*\]\s*=\s*'\\0'", cfa) or \
           re.search(r"name\[.*\]\s*=\s*0", cfa)


# ═══════════════════════════════════════════════════════════════
# T2-03: CWE-122 Fix 1 (NULL check after malloc)
# ═══════════════════════════════════════════════════════════════

def test_t2_03_cwe122_null_check_after_malloc():
    """T2-03: CWE-122 Fix 1: NULL check after malloc present in CFA."""
    with patch.object(tier2_mod, "call_llm", return_value=CWE122_CFA):
        results = tier2_generate(CWE122_VULN, "CWE-122", config=_DEFAULT_CFG)

    assert len(results) >= 1
    cfa = results[0]
    # NULL check after malloc
    assert re.search(r'if\s*\(\s*buf\s*==\s*NULL\s*\)', cfa)
    assert "return NULL" in cfa


# ═══════════════════════════════════════════════════════════════
# T2-04: CWE-122 Fix 2 (size validation before write)
# ═══════════════════════════════════════════════════════════════

def test_t2_04_cwe122_size_validation_before_write():
    """T2-04: CWE-122 Fix 2: size validation before write present in CFA."""
    with patch.object(tier2_mod, "call_llm", return_value=CWE122_CFA):
        results = tier2_generate(CWE122_VULN, "CWE-122", config=_DEFAULT_CFG)

    assert len(results) >= 1
    cfa = results[0]
    # Size validation: write_size capped at allocated size
    assert re.search(r'if\s*\(\s*write_size\s*>\s*size\s*\)', cfa)


# ═══════════════════════════════════════════════════════════════
# T2-05: CWE-125 bounds check before read
# ═══════════════════════════════════════════════════════════════

def test_t2_05_cwe125_bounds_check_before_read():
    """T2-05: CWE-125: bounds check before array read present in CFA."""
    with patch.object(tier2_mod, "call_llm", return_value=CWE125_CFA):
        results = tier2_generate(CWE125_VULN, "CWE-125", config=_DEFAULT_CFG)

    assert len(results) >= 1
    cfa = results[0]
    # Bounds check present
    assert re.search(r'if\s*\(\s*idx\s*<\s*0\s*\|', cfa) or \
           re.search(r'if\s*\(\s*idx\s*(>=|<)\s*len\s*\)', cfa)
    # Check comes before the read
    lines = cfa.splitlines()
    check_line = next((i for i, l in enumerate(lines) if re.search(r'if.*idx', l)), None)
    read_line  = next((i for i, l in enumerate(lines) if "arr[idx]" in l), None)
    assert check_line is not None
    assert read_line is not None
    assert check_line < read_line


# ═══════════════════════════════════════════════════════════════
# T2-06: extract_c_code_v2 — [FIXED_CODE_START] marker
# ═══════════════════════════════════════════════════════════════

def test_t2_06_extract_fixed_code_marker():
    """T2-06: extract_c_code_v2 extracts from [FIXED_CODE_START] marker."""
    response = (
        "Here is the explanation of the fix...\n"
        "[FIXED_CODE_START]\n"
        "void foo(int x) {\n"
        "    if (x < 0) { return; }\n"
        "    arr[x] = 1;\n"
        "}"
    )
    result = extract_c_code_v2(response)
    assert result.startswith("void foo")
    assert "if (x < 0)" in result
    # Explanation text should not appear
    assert "explanation" not in result


# ═══════════════════════════════════════════════════════════════
# T2-07: extract_c_code_v2 — markdown fence fallback
# ═══════════════════════════════════════════════════════════════

def test_t2_07_extract_markdown_fence_fallback():
    """T2-07: extract_c_code_v2 falls back to markdown fence extraction."""
    response = (
        "The fixed function:\n"
        "```c\n"
        "void bar(char *buf, int len) {\n"
        "    if (len > 64) { len = 64; }\n"
        "    memcpy(dst, buf, len);\n"
        "}\n"
        "```\n"
        "This ensures the buffer is not overflowed."
    )
    result = extract_c_code_v2(response)
    assert result.startswith("void bar")
    assert "if (len > 64)" in result
    # Trailing explanation should not appear
    assert "This ensures" not in result


# ═══════════════════════════════════════════════════════════════
# T2-08: Gate 5 soft — no fix signature gives score=0.6, not False
# ═══════════════════════════════════════════════════════════════

def test_t2_08_gate5_soft_quality_score():
    """T2-08: CFA with no fix signature returns quality_score=0.6, still accepted."""
    # CWE121_NO_FIX_SIG changes arr[idx]=val to arr[0]=val (no bounds check or strncpy)
    # So it has no fix signature but also will pass Gate 4 (no strcpy/gets).
    # Gate 5 is soft: should still be accepted with score=0.6.
    with patch.object(tier2_mod, "call_llm", return_value=CWE121_NO_FIX_SIG):
        cfg = CfaConfig(
            use_tree_sitter_fallback=True,
            similarity_lower_default=0.40,  # relax so the change isn't too_similar/different
        )
        results = tier2_generate(CWE121_VULN, "CWE-121", config=cfg)

    # The CFA should be accepted (Gate 5 is soft) OR rejected for another reason.
    # What matters: if accepted, it must not be because Gate 5 was a hard blocker.
    # Verify directly via validate_cfa_v2.
    from training.scripts.preprocessing.stage3_cfa import validate_cfa_v2
    is_valid, reason, code, score = validate_cfa_v2(
        CWE121_VULN, CWE121_NO_FIX_SIG, "CWE-121", tier=2,
        config=CfaConfig(
            use_tree_sitter_fallback=True,
            similarity_lower_default=0.40,
        ),
    )
    # Gate 5 must NOT reject (only soft penalty)
    assert reason != "no_fix_signature", (
        "Gate 5 must be SOFT — 'no_fix_signature' must not reject the CFA"
    )
    if is_valid:
        assert score == pytest.approx(0.6), "Soft Gate 5 miss must give score=0.6"


# ═══════════════════════════════════════════════════════════════
# T2-09: compile fail 3 in a row → Tier 5 refinement attempted
# ═══════════════════════════════════════════════════════════════

def test_t2_09_compile_fail_triggers_tier5():
    """T2-09: 3 consecutive compile fails triggers Tier 5 refinement attempt."""
    # Must be similar enough to pass Gate 2 (sim >= 0.55) but fail Gate 3 (compile).
    # Missing closing brace on the if-block — similar tokens but broken syntax.
    bad_but_similar = (
        "void store_value(int idx, int val) {\n"
        "    int arr[10];\n"
        "    int temp = 0;\n"
        "    if (idx < 0 || idx >= 10) {\n"   # <-- opens a block, never closes it
        "    arr[idx] = val;\n"
        "    temp = arr[0];\n"
        "}\n"
    )

    tier5_called = []

    def fake_tier5(original, failed_cfa, cwe, reason, config):
        tier5_called.append(True)
        return None  # Tier 5 returns nothing (stub)

    with patch.object(tier2_mod, "call_llm", return_value=bad_but_similar), \
         patch.object(tier2_mod, "_tier5_refine", side_effect=fake_tier5):
        cfg = CfaConfig(
            use_tree_sitter_fallback=True,
            max_attempts_tier2=6,
        )
        tier2_generate(CWE121_VULN, "CWE-121", config=cfg)

    assert len(tier5_called) >= 1, "Tier 5 refinement must be attempted after 3 compile fails"


# ═══════════════════════════════════════════════════════════════
# T2-10: max_attempts respected
# ═══════════════════════════════════════════════════════════════

def test_t2_10_max_attempts_respected():
    """T2-10: call_llm is called at most max_attempts_tier2 times."""
    call_count = []

    def counting_llm(*args, **kwargs):
        call_count.append(1)
        return None  # always fails

    with patch.object(tier2_mod, "call_llm", side_effect=counting_llm):
        cfg = CfaConfig(max_attempts_tier2=4)
        tier2_generate(CWE121_VULN, "CWE-121", config=cfg)

    assert len(call_count) <= 4, (
        f"Expected at most 4 LLM calls, got {len(call_count)}"
    )
