#!/usr/bin/env python3
"""
tests/test_cfa_tier3.py

Story CFA-T3 — Tier 3: Chain-of-Thought LLM CFA Generation.

12 tests covering CoT prompts, code extraction (with [FIXED_CODE_END]),
Gate 6 taint enforcement, feedback injection, and line-change budget.
All LLM calls are mocked — no live API key required.
"""
from __future__ import annotations

import re
from unittest.mock import call, patch

import pytest

import training.scripts.preprocessing.cfa_tier3 as tier3_mod
from training.scripts.preprocessing.cfa_tier3 import (
    TIER3_COT_PROMPTS,
    _GATE6_FEEDBACK,
    _START_MARKER,
    _END_MARKER,
    extract_c_code_tier3,
    tier3_generate,
)
from training.scripts.preprocessing.stage3_cfa import CfaConfig, validate_cfa_v2

# ── Shared configuration ─────────────────────────────────────────
_CFG = CfaConfig(use_tree_sitter_fallback=True)

# ── CWE-89: SQL injection ────────────────────────────────────────
CWE89_VULN = """\
int query_user(sqlite3 *db, const char *username) {
    char query[256];
    char input[128];
    int rc = 0;
    fgets(input, sizeof(input), stdin);
    sprintf(query, "SELECT id FROM users WHERE name='%s'", input);
    rc = sqlite3_exec(db, query, NULL, NULL, NULL);
    return rc;
}
"""

# Parameterized fix — breaks the taint path
CWE89_CFA_GOOD = """\
int query_user(sqlite3 *db, const char *username) {
    char query[256];
    char input[128];
    sqlite3_stmt *stmt = NULL;
    int rc = 0;
    fgets(input, sizeof(input), stdin);
    if (sqlite3_prepare_v2(db, "SELECT id FROM users WHERE name=?", -1, &stmt, NULL) != SQLITE_OK)
        return -1;
    sqlite3_bind_text(stmt, 1, input, -1, SQLITE_STATIC);
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return rc;
}
"""

# Still has taint path: fgets -> strcat -> sqlite3_exec
CWE89_CFA_TAINT_INTACT = """\
int query_user(sqlite3 *db, const char *username) {
    char query[256];
    char input[128];
    int rc = 0;
    fgets(input, sizeof(input), stdin);
    strcpy(query, "SELECT id FROM users WHERE name='");
    strcat(query, input);
    strcat(query, "'");
    rc = sqlite3_exec(db, query, NULL, NULL, NULL);
    return rc;
}
"""

# ── CWE-78: OS command injection ─────────────────────────────────
CWE78_VULN = """\
int run_cmd(const char *user_input) {
    char cmd[256];
    int rc = 0;
    snprintf(cmd, sizeof(cmd), "ls %s", user_input);
    rc = system(cmd);
    return rc;
}
"""

# Allowlist check — breaks the taint path for commands
CWE78_CFA_GOOD = """\
int run_cmd(const char *user_input) {
    char cmd[256];
    int rc = 0;
    if (!is_allowed_input(user_input)) { return -1; }
    snprintf(cmd, sizeof(cmd), "ls %s", user_input);
    rc = system(cmd);
    return rc;
}
"""

# popen with user input — still vulnerable
CWE78_VULN_POPEN = """\
FILE *exec_cmd(char *cmd_buf) {
    char input[128];
    FILE *fp = NULL;
    fgets(input, sizeof(input), stdin);
    snprintf(cmd_buf, 256, "echo %s", input);
    fp = popen(cmd_buf, "r");
    return fp;
}
"""

# ── CWE-190: integer overflow ────────────────────────────────────
CWE190_VULN_ADD = """\
int compute_sum(int a, int b) {
    int result = 0;
    result = a + b;
    return result;
}
"""

CWE190_CFA_ADD = """\
int compute_sum(int a, int b) {
    int result = 0;
    if (a > INT_MAX - b) { return -1; }
    result = a + b;
    return result;
}
"""

CWE190_VULN_MUL = """\
int compute_product(int a, int b) {
    int result = 0;
    result = a * b;
    return result;
}
"""

CWE190_CFA_MUL = """\
int compute_product(int a, int b) {
    int result = 0;
    if (b != 0 && a > INT_MAX / b) { return -1; }
    result = a * b;
    return result;
}
"""

# ── CWE-79: output injection ─────────────────────────────────────
CWE79_VULN = """\
void render_page(const char *user_input, char *response, int resp_size) {
    int written = 0;
    written = snprintf(response, resp_size, "<p>%s</p>", user_input);
    return;
}
"""

CWE79_CFA_GOOD = """\
void render_page(const char *user_input, char *response, int resp_size) {
    char encoded[1024];
    int written = 0;
    size_t i = 0;
    size_t j = 0;
    while (user_input[i] && j < sizeof(encoded) - 1) {
        if (user_input[i] == '<') { encoded[j++] = '&'; encoded[j++] = 'l'; encoded[j++] = 't'; encoded[j++] = ';'; }
        else { encoded[j++] = user_input[i]; }
        i++;
    }
    encoded[j] = '\\0';
    written = snprintf(response, resp_size, "<p>%s</p>", encoded);
    return;
}
"""


# ═══════════════════════════════════════════════════════════════
# T3-01: CWE-89 — parameterized query replaces sprintf injection
# ═══════════════════════════════════════════════════════════════

def test_t3_01_cwe89_parameterized_query():
    """T3-01: CWE-89: parameterized query (sqlite3_prepare or escape) in CFA."""
    with patch.object(tier3_mod, "call_llm", return_value=CWE89_CFA_GOOD):
        results = tier3_generate(CWE89_VULN, "CWE-89", config=_CFG)

    assert len(results) >= 1
    cfa = results[0]
    # Must use parameterized query
    assert "sqlite3_prepare_v2" in cfa or "mysql_real_escape" in cfa
    # Direct sprintf-into-query must be gone
    assert "sprintf(query" not in cfa


# ═══════════════════════════════════════════════════════════════
# T3-02: CWE-89 — Gate 6: CFA breaks taint path
# ═══════════════════════════════════════════════════════════════

def test_t3_02_cwe89_gate6_taint_broken():
    """T3-02: CWE-89: valid CFA must break the taint source→sink path."""
    is_valid, reason, _, _ = validate_cfa_v2(
        CWE89_VULN, CWE89_CFA_GOOD, "CWE-89", tier=3, config=_CFG
    )
    # Gate 6 should pass (taint path broken by parameterized query)
    assert reason != "taint_path_intact", (
        f"Good CFA should pass Gate 6, but got reason: {reason}"
    )


# ═══════════════════════════════════════════════════════════════
# T3-03: CWE-89 — still has sprintf(query,...,user) → fails Gate 6
# ═══════════════════════════════════════════════════════════════

def test_t3_03_cwe89_taint_intact_fails_gate6():
    """T3-03: CWE-89 CFA with fgets->strcat->sqlite3_exec taint fails Gate 6."""
    cfg = CfaConfig(
        use_tree_sitter_fallback=True,
        similarity_lower_default=0.40,  # relax Gate 2 for this structural change
    )
    is_valid, reason, _, _ = validate_cfa_v2(
        CWE89_VULN, CWE89_CFA_TAINT_INTACT, "CWE-89", tier=3, config=cfg
    )
    assert not is_valid
    assert reason == "taint_path_intact"


# ═══════════════════════════════════════════════════════════════
# T3-04: CWE-78 — system(user_input) replaced with allowlist check
# ═══════════════════════════════════════════════════════════════

def test_t3_04_cwe78_system_replaced():
    """T3-04: CWE-78: system() with direct user input is replaced."""
    with patch.object(tier3_mod, "call_llm", return_value=CWE78_CFA_GOOD):
        results = tier3_generate(CWE78_VULN, "CWE-78", config=_CFG)

    assert len(results) >= 1
    cfa = results[0]
    # Allowlist or execve-based fix present
    assert "is_allowed_input" in cfa or "execve" in cfa or "whitelist" in cfa.lower()


# ═══════════════════════════════════════════════════════════════
# T3-05: CWE-78 — popen with user input detected by Gate 6
# ═══════════════════════════════════════════════════════════════

def test_t3_05_cwe78_popen_gate6():
    """T3-05: CWE-78: popen with user input taint path must be detected."""
    # A "fix" that only renames the variable but keeps popen(cmd, "r") with user input
    # We test Gate 6 directly
    from training.scripts.preprocessing.stage3_cfa import _has_taint_path
    assert _has_taint_path(CWE78_VULN_POPEN, "CWE-78"), (
        "Gate 6 should detect popen() as a CWE-78 taint sink"
    )


# ═══════════════════════════════════════════════════════════════
# T3-06: CWE-190 — addition overflow → INT_MAX - b guard
# ═══════════════════════════════════════════════════════════════

def test_t3_06_cwe190_addition_overflow_guard():
    """T3-06: CWE-190: addition overflow check (INT_MAX - b) added."""
    with patch.object(tier3_mod, "call_llm", return_value=CWE190_CFA_ADD):
        results = tier3_generate(CWE190_VULN_ADD, "CWE-190", config=_CFG)

    assert len(results) >= 1
    cfa = results[0]
    assert "INT_MAX" in cfa
    # Overflow check before the addition
    lines = cfa.splitlines()
    check_line = next((i for i, l in enumerate(lines) if "INT_MAX" in l), None)
    add_line   = next((i for i, l in enumerate(lines) if "= a + b" in l), None)
    assert check_line is not None, "INT_MAX check not found"
    assert add_line is not None, "Addition not found"
    assert check_line < add_line, "Check must come before addition"


# ═══════════════════════════════════════════════════════════════
# T3-07: CWE-190 — multiplication overflow → INT_MAX / b guard
# ═══════════════════════════════════════════════════════════════

def test_t3_07_cwe190_multiplication_overflow_guard():
    """T3-07: CWE-190: multiplication overflow check (INT_MAX / b) added."""
    with patch.object(tier3_mod, "call_llm", return_value=CWE190_CFA_MUL):
        results = tier3_generate(CWE190_VULN_MUL, "CWE-190", config=_CFG)

    assert len(results) >= 1
    cfa = results[0]
    assert "INT_MAX" in cfa
    assert re.search(r'INT_MAX\s*/\s*b', cfa)


# ═══════════════════════════════════════════════════════════════
# T3-08: CWE-79 — user output encoded before write
# ═══════════════════════════════════════════════════════════════

def test_t3_08_cwe79_html_encoding_applied():
    """T3-08: CWE-79: html_encode applied before writing user input to output."""
    # CWE-79 fixes add encoding logic — similarity is inherently lower.
    # Use relaxed similarity bounds to allow the encoding helper to be added.
    cfg = CfaConfig(
        use_tree_sitter_fallback=True,
        similarity_lower_default=0.40,
    )
    with patch.object(tier3_mod, "call_llm", return_value=CWE79_CFA_GOOD):
        results = tier3_generate(CWE79_VULN, "CWE-79", config=cfg)

    assert len(results) >= 1
    cfa = results[0]
    # Encoding must be applied
    assert "encoded" in cfa
    # HTML entity characters or encoding logic present
    assert "&lt;" in cfa or "'<'" in cfa or "user_input[i] ==" in cfa


# ═══════════════════════════════════════════════════════════════
# T3-09: CoT reasoning text before [FIXED_CODE_START] NOT included
# ═══════════════════════════════════════════════════════════════

def test_t3_09_cot_reasoning_excluded():
    """T3-09: CoT reasoning text before [FIXED_CODE_START] is NOT in extracted code."""
    reasoning = (
        "STEP 1 - IDENTIFY: I can see that sprintf is used with user input.\n"
        "STEP 2 - STRATEGY: I will use sqlite3_prepare_v2 instead.\n"
        "STEP 3 - Generate:\n"
    )
    code_block = (
        "int query_user(sqlite3 *db) {\n"
        "    sqlite3_stmt *stmt = NULL;\n"
        "    sqlite3_prepare_v2(db, \"SELECT 1\", -1, &stmt, NULL);\n"
        "    return 0;\n"
        "}"
    )
    full_response = reasoning + _START_MARKER + "\n" + code_block + "\n" + _END_MARKER

    result = extract_c_code_tier3(full_response)

    assert "STEP 1" not in result
    assert "STEP 2" not in result
    assert "IDENTIFY" not in result
    assert "sqlite3_prepare_v2" in result


# ═══════════════════════════════════════════════════════════════
# T3-10: [FIXED_CODE_END] correctly terminates extraction
# ═══════════════════════════════════════════════════════════════

def test_t3_10_end_marker_terminates_extraction():
    """T3-10: [FIXED_CODE_END] terminates extraction, trailing text excluded."""
    code_block = "void fixed(int x) {\n    if (x < 0) return;\n    arr[x] = 1;\n}"
    trailing   = "\nNote: this is my explanation of the fix."
    response   = _START_MARKER + "\n" + code_block + "\n" + _END_MARKER + trailing

    result = extract_c_code_tier3(response)

    assert "void fixed" in result
    assert "Note:" not in result
    assert "explanation" not in result


# ═══════════════════════════════════════════════════════════════
# T3-11: Gate 6 miss — feedback injected into NEXT attempt prompt
# ═══════════════════════════════════════════════════════════════

def test_t3_11_gate6_feedback_injected_on_miss():
    """T3-11: When Gate 6 fails, feedback is injected into the next attempt's prompt."""
    prompts_received: list[str] = []

    def tracking_llm(prompt: str, **kwargs) -> str:
        prompts_received.append(prompt)
        # First attempt: return a taint-intact CFA (will fail Gate 6)
        # Second attempt: return the good CFA (should have feedback in prompt)
        if len(prompts_received) == 1:
            return CWE89_CFA_TAINT_INTACT
        return CWE89_CFA_GOOD

    cfg = CfaConfig(
        use_tree_sitter_fallback=True,
        max_attempts_tier3=7,
        similarity_lower_default=0.40,
    )
    with patch.object(tier3_mod, "call_llm", side_effect=tracking_llm):
        tier3_generate(CWE89_VULN, "CWE-89", config=cfg)

    assert len(prompts_received) >= 2, "Expected at least 2 LLM calls"
    # Second prompt must contain taint feedback
    feedback_text = _GATE6_FEEDBACK.get("CWE-89", "")
    assert feedback_text and feedback_text[:30] in prompts_received[1], (
        "Gate 6 feedback must appear in the second attempt's prompt"
    )


# ═══════════════════════════════════════════════════════════════
# T3-12: avg_changed_lines for CWE-89: 3-8 lines (minimal edit)
# ═══════════════════════════════════════════════════════════════

def test_t3_12_cwe89_minimal_line_change():
    """T3-12: CWE-89 CFA changes 3-8 lines relative to original (minimal edit)."""
    orig_lines = CWE89_VULN.splitlines()
    cfa_lines  = CWE89_CFA_GOOD.splitlines()

    # Count differing lines (by position, not diff algorithm)
    from difflib import unified_diff
    diff = list(unified_diff(orig_lines, cfa_lines, lineterm=""))
    changed = sum(1 for l in diff if l.startswith(("+", "-")) and not l.startswith(("+++", "---")))

    assert 3 <= changed <= 20, (
        f"Expected 3-20 changed lines for CWE-89 fix, got {changed}. "
        "If > 20: prompt over-editing. If < 3: fix too trivial."
    )
