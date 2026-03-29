#!/usr/bin/env python3
"""
tests/test_cfa_tier1.py

Story CFA-T1 — Tier 1: AST Rule-Based Mutation.

14 tests covering CWE-134 (4), CWE-120 (6), CWE-476 (4).
"""
from __future__ import annotations

import re

import pytest

from training.scripts.preprocessing.cfa_tier1 import (
    fix_cwe134,
    fix_cwe120,
    fix_cwe476,
    tier1_generate,
)


# ═══════════════════════════════════════════════════════════════
# CWE-134 tests (format string)
# ═══════════════════════════════════════════════════════════════

def test_t1_01_printf_variable_format():
    """T1-01: printf(user_str) -> printf("%s", user_str)."""
    code = (
        'void log_msg(char *user_str) {\n'
        '    int x = 1;\n'
        '    printf(user_str);\n'
        '    x = 2;\n'
        '}\n'
    )
    result = fix_cwe134(code)
    assert result is not None
    assert '"%s", user_str' in result
    # Original variable still present as second arg
    assert "user_str" in result
    # No unformatted printf(var) remaining
    assert 'printf(user_str)' not in result


def test_t1_02_printf_string_literal_unchanged():
    """T1-02: printf("fixed") is not vulnerable — returns None."""
    code = (
        'void log_msg() {\n'
        '    printf("hello world");\n'
        '}\n'
    )
    result = fix_cwe134(code)
    assert result is None


def test_t1_03_fprintf_variable_format():
    """T1-03: fprintf(f, user_str) -> fprintf(f, "%s", user_str)."""
    code = (
        'void log_to_file(FILE *f, char *user_str) {\n'
        '    int rc = 0;\n'
        '    fprintf(f, user_str);\n'
        '    rc = 1;\n'
        '}\n'
    )
    result = fix_cwe134(code)
    assert result is not None
    assert 'fprintf(f, "%s", user_str)' in result


def test_t1_04_wprintf_variable_format():
    """T1-04: wprintf(w_str) -> wprintf(L"%s", w_str)."""
    code = (
        'void log_wide(wchar_t *w_str) {\n'
        '    int x = 0;\n'
        '    wprintf(w_str);\n'
        '    x = 1;\n'
        '}\n'
    )
    result = fix_cwe134(code)
    assert result is not None
    assert 'L"%s", w_str' in result


# ═══════════════════════════════════════════════════════════════
# CWE-120 tests (buffer copy)
# ═══════════════════════════════════════════════════════════════

def test_t1_05_strcpy_array_fixed():
    """T1-05: strcpy(arr, src) with char arr[100] -> strncpy fixed."""
    code = (
        'void copy(const char *src) {\n'
        '    char arr[100];\n'
        '    strcpy(arr, src);\n'
        '}\n'
    )
    result = fix_cwe120(code)
    assert result is not None
    assert "strncpy(arr, src, sizeof(arr) - 1)" in result
    assert "arr[sizeof(arr) - 1] = '\\0'" in result
    # Original unsafe call removed
    assert "strcpy(" not in result


def test_t1_06_strcpy_pointer_param_escalates():
    """T1-06: strcpy(ptr, src) with char *ptr param -> returns None (escalate)."""
    code = (
        'void copy(char *ptr, const char *src) {\n'
        '    strcpy(ptr, src);\n'
        '}\n'
    )
    result = fix_cwe120(code)
    assert result is None, "Pointer param destination must escalate to Tier 2"


def test_t1_07_gets_to_fgets():
    """T1-07: gets(buf) -> fgets(buf, sizeof(buf), stdin)."""
    code = (
        'void read_input() {\n'
        '    char buf[128];\n'
        '    gets(buf);\n'
        '}\n'
    )
    result = fix_cwe120(code)
    assert result is not None
    assert "fgets(buf, sizeof(buf), stdin)" in result
    assert "gets(" not in result or "fgets(" in result


def test_t1_08_sprintf_to_snprintf():
    """T1-08: sprintf(buf, fmt, arg) -> snprintf(buf, sizeof(buf), fmt, arg)."""
    code = (
        'void format(const char *name, int id) {\n'
        '    char buf[256];\n'
        '    sprintf(buf, "user=%s id=%d", name, id);\n'
        '}\n'
    )
    result = fix_cwe120(code)
    assert result is not None
    assert 'snprintf(buf, sizeof(buf), "user=%s id=%d", name, id)' in result
    # No bare sprintf remaining
    assert re.search(r'\bsprintf\s*\(', result) is None


def test_t1_09_strcat_array_fixed():
    """T1-09: strcat(dst, src) with array -> strncat fixed."""
    code = (
        'void append(const char *extra) {\n'
        '    char dst[64];\n'
        '    dst[0] = 0;\n'
        '    strcat(dst, extra);\n'
        '}\n'
    )
    result = fix_cwe120(code)
    assert result is not None
    assert "strncat(dst, extra, sizeof(dst) - strlen(dst) - 1)" in result


def test_t1_10_no_vulnerable_calls():
    """T1-10: no vulnerable calls found -> returns []."""
    code = (
        'void safe_func(const char *src) {\n'
        '    char buf[64];\n'
        '    strncpy(buf, src, sizeof(buf) - 1);\n'
        '    buf[sizeof(buf) - 1] = 0;\n'
        '}\n'
    )
    results = tier1_generate(code, "CWE-120")
    assert results == []


# ═══════════════════════════════════════════════════════════════
# CWE-476 tests (NULL dereference)
# ═══════════════════════════════════════════════════════════════

def test_t1_11_star_ptr_null_guard():
    """T1-11: *ptr with no NULL check -> NULL guard inserted before dereference."""
    code = (
        'int process(int *ptr) {\n'
        '    int val = *ptr;\n'
        '    return val;\n'
        '}\n'
    )
    result = fix_cwe476(code)
    assert result is not None
    assert "if (ptr == NULL)" in result
    assert "return -1;" in result  # int return type
    # Guard line must appear before the dereference line in the output
    lines = result.splitlines()
    guard_line = next(i for i, l in enumerate(lines) if "if (ptr == NULL)" in l)
    deref_line = next(i for i, l in enumerate(lines) if "= *ptr" in l)
    assert guard_line < deref_line, "Guard must come before dereference"


def test_t1_12_arrow_deref_null_guard():
    """T1-12: ptr->field with no NULL check -> NULL guard inserted."""
    code = (
        'char *get_name(struct Node *node) {\n'
        '    char *name = node->name;\n'
        '    return name;\n'
        '}\n'
    )
    result = fix_cwe476(code)
    assert result is not None
    assert "if (node == NULL)" in result
    assert "return NULL;" in result  # char* return type


def test_t1_13_already_guarded_no_change():
    """T1-13: *ptr already has if(ptr != NULL) check -> no change, returns []."""
    code = (
        'int safe(int *ptr) {\n'
        '    if (ptr != NULL) {\n'
        '        return *ptr;\n'
        '    }\n'
        '    return 0;\n'
        '}\n'
    )
    results = tier1_generate(code, "CWE-476")
    assert results == [], "Already guarded dereference should not generate a CFA"


def test_t1_14_multiple_deref_sites_all_guarded():
    """T1-14: multiple dereference sites -> ALL get NULL guards."""
    code = (
        'int combine(int *a, int *b) {\n'
        '    int x = *a;\n'
        '    int y = *b;\n'
        '    return x + y;\n'
        '}\n'
    )
    result = fix_cwe476(code)
    assert result is not None
    # Both a and b should have NULL guards
    assert "if (a == NULL)" in result
    assert "if (b == NULL)" in result
    # Guards must appear before their respective dereferences (by line)
    lines = result.splitlines()
    a_guard_line = next(i for i, l in enumerate(lines) if "if (a == NULL)" in l)
    a_deref_line = next(i for i, l in enumerate(lines) if "= *a" in l)
    b_guard_line = next(i for i, l in enumerate(lines) if "if (b == NULL)" in l)
    b_deref_line = next(i for i, l in enumerate(lines) if "= *b" in l)
    assert a_guard_line < a_deref_line
    assert b_guard_line < b_deref_line
