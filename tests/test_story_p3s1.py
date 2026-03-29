#!/usr/bin/env python3
"""
tests/test_story_p3s1.py

Story P3-S1 — Stage 1: Clean & Normalize
12 tests covering the updated stage1_clean.py.

Tests:
 1. strip_preprocessor_guards removes #ifdef/#ifndef/#endif lines
 2. strip_preprocessor_guards preserves code between guards
 3. normalize removes // comments
 4. normalize removes /* */ block comments
 5. normalize rejects code under 5 lines
 6. normalize rejects code over 500 lines
 7. normalize rejects code under 10 tokens
 8. SARD path: code with #ifdef → stripped before comment removal
 9. SARD path: no function_definition → returns None / empty list
10. non-SARD path: full file → multiple functions extracted
11. Dry run: no output file written
12. max-samples: stops at limit
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "training" / "scripts" / "preprocessing"),
)
from stage1_clean import (
    SARD_SOURCES,
    extract_functions,
    normalize,
    process_sample,
    run_stage1,
    strip_preprocessor_guards,
    MIN_LINES,
    MAX_LINES,
    MIN_TOKENS,
)


# ── Fixtures ───────────────────────────────────────────────────────

# A valid C function that passes all filters (>= 5 lines, >= 10 tokens)
VALID_C_FUNCTION = """\
void foo(int x) {
    int a = 0;
    int b = 1;
    int c = 2;
    int d = 3;
    if (x > 0) {
        a = x;
    }
    return;
}"""

# Function wrapped in Juliet #ifndef guards
FUNCTION_WITH_IFDEF = """\
#ifndef OMITBAD
void bad_func(char *data) {
    int a = 0;
    int b = 1;
    int c = 2;
    char buf[10];
    strcpy(buf, data);
    return;
}
#endif"""

# Function with both kinds of comments
FUNCTION_WITH_COMMENTS = """\
void bar(int x) {
    // This is a line comment
    int a = 0;
    /* This is a
       block comment */
    int b = 1;
    int c = 2;
    int d = 3;
    return;
}"""

# Full C file with two functions (for non-SARD extraction test)
MULTI_FUNCTION_FILE = """\
#include <stdio.h>

void func_one(int x) {
    int a = 0;
    int b = 1;
    int c = 2;
    int d = 3;
    printf("%d", x + a + b + c + d);
}

int func_two(char *s) {
    int len = 0;
    int a = 0;
    int b = 1;
    int c = 2;
    len = strlen(s);
    return len + a + b + c;
}
"""

# Code that is not a function — just global variable declarations
NOT_A_FUNCTION = """\
int x = 5;
int y = 10;
int z = x + y;
"""


def _make_sard_sample(code: str = VALID_C_FUNCTION, source: str = "sard", **overrides) -> dict:
    base = {
        "id": "test-sard-001",
        "source": source,
        "code": code,
        "label": 1,
        "cwe": "CWE-120",
        "language": "c",
        "collected_at": "2026-01-01T00:00:00Z",
        "pair_id": "pair-001",
    }
    base.update(overrides)
    return base


def _make_nonsard_sample(code: str = MULTI_FUNCTION_FILE, **overrides) -> dict:
    base = {
        "id": "test-cve-001",
        "source": "cve",
        "code": code,
        "label": 1,
        "cwe": "CWE-120",
        "language": "c",
        "collected_at": "2026-01-01T00:00:00Z",
        "pair_id": "",
    }
    base.update(overrides)
    return base


def _write_jsonl(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


# ── Test 1: strip_preprocessor_guards removes directive lines ──────

def test_strip_preprocessor_guards_removes_directives():
    """#ifdef, #ifndef, #elif, #else, #endif lines must be removed."""
    code = "#ifndef OMITBAD\nvoid f() {}\n#endif\n"
    result = strip_preprocessor_guards(code)
    assert "#ifndef" not in result
    assert "#endif" not in result


# ── Test 2: strip_preprocessor_guards preserves code between guards

def test_strip_preprocessor_guards_preserves_code_between():
    """The function body between guards must survive."""
    code = "#ifndef OMITBAD\nvoid bad(char *d) { strcpy(d, d); }\n#endif\n"
    result = strip_preprocessor_guards(code)
    assert "void bad" in result
    assert "strcpy" in result


# ── Test 3: normalize removes // line comments ─────────────────────

def test_normalize_removes_line_comments():
    result = normalize(FUNCTION_WITH_COMMENTS)
    assert result is not None
    assert "//" not in result


# ── Test 4: normalize removes /* */ block comments ────────────────

def test_normalize_removes_block_comments():
    result = normalize(FUNCTION_WITH_COMMENTS)
    assert result is not None
    assert "/*" not in result
    assert "*/" not in result


# ── Test 5: normalize rejects code under 5 lines ──────────────────

def test_normalize_rejects_under_min_lines():
    short = "void f() {\n    return;\n}"  # 3 lines
    result = normalize(short)
    assert result is None


# ── Test 6: normalize rejects code over 500 lines ─────────────────

def test_normalize_rejects_over_max_lines():
    long_code = "void foo() {\n" + "    int x = 0;\n" * (MAX_LINES + 1) + "}\n"
    result = normalize(long_code)
    assert result is None


# ── Test 7: normalize rejects code under 10 tokens ────────────────

def test_normalize_rejects_under_min_tokens():
    # 7-line file but almost no tokens after comment removal
    sparse = "void f() {\n\n\n\n\n\n}"
    result = normalize(sparse)
    assert result is None


# ── Test 8: SARD path — #ifdef stripped before comment removal ─────

def test_sard_path_strips_ifdef_before_comment_removal():
    """
    SARD sample with #ifndef guard: normalize() must strip the guard and
    then successfully remove comments, leaving no directive tokens.
    """
    code_with_guard = (
        "#ifndef OMITBAD\n"
        "void bad_func(char *data) {\n"
        "    // unsafe copy\n"
        "    int a = 0;\n"
        "    int b = 1;\n"
        "    int c = 2;\n"
        "    char buf[10];\n"
        "    strcpy(buf, data);\n"
        "    return;\n"
        "}\n"
        "#endif\n"
    )
    result = normalize(code_with_guard, source="sard")
    assert result is not None, "Expected sample to pass after guard stripping"
    assert "#ifndef" not in result
    assert "#endif" not in result
    assert "//" not in result  # comment also removed


def test_sard_cfa_path_strips_ifdef():
    """sard_cfa source must also get guard stripping (uses SARD_SOURCES set)."""
    assert "sard_cfa" in SARD_SOURCES
    code_with_guard = (
        "#ifndef OMITGOOD\n"
        "void good_func(char *data) {\n"
        "    int a = 0;\n"
        "    int b = 1;\n"
        "    int c = 2;\n"
        "    char buf[10];\n"
        "    strncpy(buf, data, sizeof(buf) - 1);\n"
        "    return;\n"
        "}\n"
        "#endif\n"
    )
    result = normalize(code_with_guard, source="sard_cfa")
    assert result is not None
    assert "#ifndef" not in result
    assert "#endif" not in result


# ── Test 9: SARD path — no function_definition → returns [] ───────

def test_sard_path_no_function_definition_returns_empty():
    """process_sample() on a SARD sample without a function must return []."""
    sample = _make_sard_sample(code=NOT_A_FUNCTION)
    results = process_sample(sample)
    assert results == [] or results is None or len(results) == 0


# ── Test 10: non-SARD path — full file → multiple functions extracted

def test_nonsard_full_file_extracts_multiple_functions():
    """CVE / non-SARD sample with a full two-function file → 2 output samples."""
    sample = _make_nonsard_sample(code=MULTI_FUNCTION_FILE)
    results = process_sample(sample)
    assert len(results) >= 2
    names = [r.get("metadata", {}).get("function_name", "") for r in results]
    # Both function names should be present somewhere in results
    assert any("func_one" in n for n in names)
    assert any("func_two" in n for n in names)


# ── Test 11: Dry run — no output file written ─────────────────────

def test_dry_run_no_output_file(tmp_path):
    """With --dry-run, run_stage1 must NOT create the output file."""
    indir = tmp_path / "raw"
    indir.mkdir()
    samples = [_make_sard_sample(id=f"s{i}") for i in range(3)]
    _write_jsonl(indir / "samples.jsonl", samples)

    output = tmp_path / "out" / "cleaned.jsonl"
    stats = run_stage1([str(indir)], str(output), dry_run=True)

    assert not output.exists(), "Output file must not be created in dry-run mode"
    assert stats["processed"] == 3


# ── Test 12: max-samples stops at limit ───────────────────────────

def test_max_samples_stops_at_limit(tmp_path):
    """run_stage1 must stop after max_samples successfully processed samples."""
    indir = tmp_path / "raw"
    indir.mkdir()
    samples = [_make_sard_sample(id=f"s{i}") for i in range(10)]
    _write_jsonl(indir / "samples.jsonl", samples)

    output = tmp_path / "out" / "cleaned.jsonl"
    stats = run_stage1([str(indir)], str(output), max_samples=4)

    assert stats["processed"] == 4
    lines = output.read_text().strip().split("\n")
    assert len(lines) == 4
