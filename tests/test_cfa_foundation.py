#!/usr/bin/env python3
"""
tests/test_cfa_foundation.py

Story P3-S3 Foundation — CFA Orchestrator and 7-Gate Validator.

8 tests covering:
  1. CWE_TIER_MAP completeness and correctness
  2. Gate 1: identical code rejected
  3. Gate 2: similarity bounds enforced (tier-aware)
  4. Gate 3: compile fail rejected
  5. Gate 4: vuln pattern still present rejected
  6. Gate 5 SOFT: missing fix signature -> quality_score=0.6, NOT rejected
  7. Gate 6: injection CWE with intact taint path rejected
  8. SARD skip: source='sard' samples get NO CFA generation attempt
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from training.scripts.preprocessing.stage3_cfa import (
    CWE_TIER_MAP,
    CfaConfig,
    CfaQualityTracker,
    FIX_SIGNATURES,
    TIER_GENERATORS,
    VULN_PATTERNS,
    _SARD_SOURCES,
    compiles,
    generate_cfa_for_sample,
    run_stage3_tiered,
    validate_cfa_v2,
)


# ── Helpers ──────────────────────────────────────────────────────
def _make_sample(
    code: str,
    label: int = 1,
    cwe: str = "CWE-121",
    source: str = "cve",
    sample_id: str = "",
    pair_id: str = "",
) -> dict:
    """Build a minimal sample dict for testing."""
    return {
        "id": sample_id or f"test-{id(code)}",
        "source": source,
        "code": code,
        "label": label,
        "cwe": cwe,
        "language": "c",
        "collected_at": "2026-03-22T00:00:00Z",
        "pair_id": pair_id,
        "file_path": "",
        "function_name": "test_func",
        "cfa_origin": "native",
        "severity_score": 0.0,
        "commit_sha": "",
        "cve_id": "",
        "aliases": {},
        "metadata": {},
    }


# A valid C function (vulnerable CWE-120: uses strcpy)
VULN_CWE120 = """\
void copy_input(char *dst, const char *src) {
    char buffer[64];
    int len = 0;
    strcpy(buffer, src);
    len = strlen(buffer);
    printf("%s\\n", buffer);
    return;
}
"""

# A valid CFA for CWE-120: replaces strcpy with strncpy
CFA_CWE120_GOOD = """\
void copy_input(char *dst, const char *src) {
    char buffer[64];
    int len = 0;
    strncpy(buffer, src, sizeof(buffer) - 1);
    buffer[sizeof(buffer) - 1] = '\\0';
    len = strlen(buffer);
    printf("%s\\n", buffer);
    return;
}
"""

# CFA that still has the vuln pattern (strcpy)
CFA_CWE120_STILL_VULN = """\
void copy_input(char *dst, const char *src) {
    char buffer[64];
    int len = 0;
    strcpy(buffer, src);
    len = strlen(buffer);
    printf("copied %s\\n", buffer);
    return;
}
"""

# CFA with no fix signature (removed the copy entirely)
CFA_CWE120_NO_FIX_SIG = """\
void copy_input(char *dst, const char *src) {
    char buffer[64];
    int len = 0;
    memset(buffer, 0, 64);
    len = strlen(buffer);
    printf("%s\\n", buffer);
    return;
}
"""

# Totally different code (too_different)
CFA_CWE120_TOO_DIFFERENT = """\
int main(int argc, char **argv) {
    int result = 0;
    for (int i = 0; i < argc; i++) {
        result += atoi(argv[i]);
    }
    return result;
}
"""

# Injection vuln (CWE-89) with taint path
VULN_CWE89 = """\
void query_user(const char *username) {
    char query[256];
    char input[128];
    fgets(input, sizeof(input), stdin);
    sprintf(query, "SELECT * FROM users WHERE name='%s'", input);
    mysql_query(conn, query);
}
"""

# CFA for CWE-89 that passes Gate 4 (no sprintf...query...%s) but fails Gate 6
# (taint path still intact: fgets -> strcat -> mysql_query)
CFA_CWE89_TAINT_INTACT = """\
void query_user(const char *username) {
    char query[256];
    char input[128];
    fgets(input, sizeof(input), stdin);
    strcpy(query, "SELECT * FROM users WHERE name='");
    strcat(query, input);
    strcat(query, "'");
    mysql_query(conn, query);
    return;
}
"""

# Invalid C syntax
CFA_BAD_SYNTAX = """\
void broken(int x {
    if x > 0
        printf("hello"
    return
"""


# ── Test 1: CWE_TIER_MAP completeness ───────────────────────────
def test_cwe_tier_map_completeness():
    """CWE_TIER_MAP has exactly 12 entries, all values 1-4."""
    assert len(CWE_TIER_MAP) == 12, f"Expected 12 entries, got {len(CWE_TIER_MAP)}"

    expected = {
        "CWE-134": 1, "CWE-120": 1, "CWE-476": 1,
        "CWE-121": 2, "CWE-122": 2, "CWE-125": 2,
        "CWE-89": 3, "CWE-78": 3, "CWE-190": 3, "CWE-79": 3,
        "CWE-416": 4, "CWE-119": 3,
    }
    for cwe, tier in expected.items():
        assert cwe in CWE_TIER_MAP, f"{cwe} missing from CWE_TIER_MAP"
        assert CWE_TIER_MAP[cwe] == tier, (
            f"{cwe}: expected tier {tier}, got {CWE_TIER_MAP[cwe]}"
        )

    # All values in range 1-4
    for cwe, tier in CWE_TIER_MAP.items():
        assert 1 <= tier <= 4, f"{cwe} has tier {tier}, expected 1-4"


# ── Test 2: Gate 1 — identical code rejected ────────────────────
def test_gate1_identical_rejected():
    """Gate 1: CFA identical to original is rejected."""
    is_valid, reason, code, qscore = validate_cfa_v2(
        VULN_CWE120, VULN_CWE120, "CWE-120", tier=1
    )
    assert not is_valid
    assert reason == "identical_to_original"
    assert code == ""
    assert qscore == 0.0


# ── Test 3: Gate 2 — similarity bounds (tier-aware) ─────────────
def test_gate2_similarity_bounds():
    """Gate 2: too_similar and too_different both rejected; Tier 1 uses stricter lower."""
    cfg = CfaConfig(use_tree_sitter_fallback=True)

    # too_similar: nearly identical (just whitespace change)
    almost_same = VULN_CWE120.rstrip() + "\n"  # trivial diff
    is_valid, reason, _, _ = validate_cfa_v2(
        VULN_CWE120, almost_same, "CWE-120", tier=1, config=cfg
    )
    # Either identical_to_original or too_similar
    assert not is_valid
    assert reason in ("identical_to_original", "too_similar")

    # too_different
    is_valid, reason, _, _ = validate_cfa_v2(
        VULN_CWE120, CFA_CWE120_TOO_DIFFERENT, "CWE-120", tier=1, config=cfg
    )
    assert not is_valid
    assert reason == "too_different"


# ── Test 4: Gate 3 — compile fail rejected ───────────────────────
def test_gate3_compile_fail():
    """Gate 3: code with syntax errors is rejected."""
    cfg = CfaConfig(use_tree_sitter_fallback=True)
    is_valid, reason, code, qscore = validate_cfa_v2(
        VULN_CWE120, CFA_BAD_SYNTAX, "CWE-120", tier=1, config=cfg
    )
    # May also fail Gate 2 (too_different) — that's fine, as long as it's rejected
    assert not is_valid
    assert reason in ("compile_fail", "too_different")


# ── Test 5: Gate 4 — vuln pattern still present rejected ────────
def test_gate4_vuln_pattern_remains():
    """Gate 4: CFA that still has the vulnerability pattern is rejected."""
    cfg = CfaConfig(use_tree_sitter_fallback=True)
    is_valid, reason, code, qscore = validate_cfa_v2(
        VULN_CWE120, CFA_CWE120_STILL_VULN, "CWE-120", tier=1, config=cfg
    )
    assert not is_valid
    assert reason == "vuln_pattern_remains"


# ── Test 6: Gate 5 SOFT — no fix signature → score=0.6, NOT rejected
def test_gate5_soft_no_fix_signature():
    """Gate 5: missing fix signature gives quality_score=0.6, NOT rejected."""
    cfg = CfaConfig(use_tree_sitter_fallback=True)
    is_valid, reason, code, qscore = validate_cfa_v2(
        VULN_CWE120, CFA_CWE120_NO_FIX_SIG, "CWE-120", tier=1, config=cfg
    )
    # Gate 4 checks strcpy pattern — CFA_CWE120_NO_FIX_SIG removed strcpy,
    # so Gate 4 passes. Gate 5 should detect missing fix signature.
    # The CFA may also fail Gate 2 depending on similarity.
    # Let's use a config with relaxed similarity for this test.
    cfg_relaxed = CfaConfig(
        use_tree_sitter_fallback=True,
        similarity_lower_tier1=0.40,
        similarity_lower_default=0.40,
    )
    is_valid, reason, code, qscore = validate_cfa_v2(
        VULN_CWE120, CFA_CWE120_NO_FIX_SIG, "CWE-120", tier=1, config=cfg_relaxed
    )
    assert is_valid, f"Should NOT be rejected (soft gate), but got: {reason}"
    assert reason == ""
    assert qscore == pytest.approx(0.6)
    assert code != ""


# ── Test 7: Gate 6 — injection CWE with taint path intact ───────
def test_gate6_taint_path_intact():
    """Gate 6: CWE-89 CFA with source→sink taint path still intact is rejected."""
    cfg = CfaConfig(
        use_tree_sitter_fallback=True,
        similarity_lower_default=0.40,
    )
    is_valid, reason, code, qscore = validate_cfa_v2(
        VULN_CWE89, CFA_CWE89_TAINT_INTACT, "CWE-89", tier=3, config=cfg
    )
    assert not is_valid
    assert reason == "taint_path_intact"
    assert code == ""
    assert qscore == 0.0


# ── Test 8: SARD skip — no CFA generation for SARD sources ─────
def test_sard_skip_no_cfa_generation():
    """SARD samples are passed through without CFA generation attempts."""
    sard_vuln = _make_sample(
        code=VULN_CWE120,
        label=1,
        cwe="CWE-120",
        source="sard",
        sample_id="sard-001",
    )
    sard_safe = _make_sample(
        code=CFA_CWE120_GOOD,
        label=0,
        cwe="CWE-120",
        source="sard",
        sample_id="sard-002",
        pair_id="sard-001",
    )

    # Write test input
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "input.jsonl")
        out_path = os.path.join(tmpdir, "output.jsonl")

        with open(in_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(sard_vuln) + "\n")
            f.write(json.dumps(sard_safe) + "\n")

        stats = run_stage3_tiered(in_path, out_path)

        assert stats["sard_skipped"] == 2
        assert stats["cfa_generated"] == 0
        assert stats["input"] == 2
        assert stats["originals_written"] == 2

        # Output should contain exactly the 2 originals, no CFAs
        with open(out_path, encoding="utf-8") as f:
            output_lines = [json.loads(l) for l in f if l.strip()]
        assert len(output_lines) == 2
        assert output_lines[0]["id"] == "sard-001"
        assert output_lines[1]["id"] == "sard-002"


# ── Bonus: FC-03 — valid CWE-120 CFA returns (True, '', code, 1.0) ──
def test_valid_cfa_cwe120():
    """FC-03: validate_cfa_v2 on a valid CWE-120 CFA returns success."""
    cfg = CfaConfig(use_tree_sitter_fallback=True)
    is_valid, reason, code, qscore = validate_cfa_v2(
        VULN_CWE120, CFA_CWE120_GOOD, "CWE-120", tier=1, config=cfg
    )
    assert is_valid, f"Expected valid, got reason: {reason}"
    assert reason == ""
    assert code.strip() != ""
    assert qscore == pytest.approx(1.0)


# ── Bonus: FC-07 — CfaQualityTracker.get_report() coverage ─────
def test_quality_tracker_all_cwes():
    """FC-07: Tracker report has all 12 CWE keys after recording 1 sample each."""
    tracker = CfaQualityTracker()
    for cwe, tier in CWE_TIER_MAP.items():
        tracker.record_attempt(cwe, tier, None, 0.85)

    report = tracker.get_report()
    assert len(report) == 12
    for cwe in CWE_TIER_MAP:
        assert cwe in report, f"{cwe} missing from report"
        assert report[cwe]["total_attempts"] == 1
        assert report[cwe]["accepted"] == 1


# ── Bonus: FC-06 — run with stub tiers returns originals only ───
def test_run_stubs_originals_only():
    """FC-06: With stub tier (Tier 2+), output has originals only (no CFAs)."""
    # Use CWE-121 (Tier 2, still a stub) to verify stub behaviour
    vuln = _make_sample(
        code=VULN_CWE120,
        label=1,
        cwe="CWE-121",
        source="cve",
        sample_id="cve-001",
    )
    safe = _make_sample(
        code=CFA_CWE120_GOOD,
        label=0,
        cwe="CWE-121",
        source="cve",
        sample_id="cve-002",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "input.jsonl")
        out_path = os.path.join(tmpdir, "output.jsonl")

        with open(in_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(vuln) + "\n")
            f.write(json.dumps(safe) + "\n")

        stats = run_stage3_tiered(in_path, out_path)

        assert stats["cfa_generated"] == 0
        assert stats["originals_written"] == 2

        with open(out_path, encoding="utf-8") as f:
            output_lines = [json.loads(l) for l in f if l.strip()]
        assert len(output_lines) == 2
