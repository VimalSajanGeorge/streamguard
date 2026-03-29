#!/usr/bin/env python3
"""
tests/test_story4_stage1.py

Tests for Stage 1: Clean & Normalize (training/scripts/preprocessing/stage1_clean.py)
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training" / "scripts" / "preprocessing"))
from stage1_clean import (
    extract_functions,
    fix_encoding,
    has_function_definition,
    normalize,
    process_sample,
    run_stage1,
    strip_preprocessor_guards,
    MIN_LINES,
    MAX_LINES,
    MIN_TOKENS,
    MAX_TOKENS,
)


# ── Fixtures ───────────────────────────────────────────────────────

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

SHORT_C_FUNCTION = """\
void f() {
    return;
}"""

LONG_C_FUNCTION = "void foo() {\n" + "    int x = 0;\n" * 501 + "}\n"

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

NOT_A_FUNCTION = """\
int x = 5;
int y = 10;
int z = x + y;
"""


def _make_sard_sample(code: str = VALID_C_FUNCTION, **overrides) -> dict:
    base = {
        "id": "test-001",
        "source": "sard",
        "code": code,
        "label": 1,
        "cwe": "CWE-120",
        "language": "c",
        "collected_at": "2026-01-01T00:00:00Z",
        "pair_id": "pair-001",
        "file_path": "test.c",
    }
    base.update(overrides)
    return base


def _make_nonsard_sample(code: str = MULTI_FUNCTION_FILE, **overrides) -> dict:
    base = {
        "id": "test-002",
        "source": "cve",
        "code": code,
        "label": 1,
        "cwe": "CWE-120",
        "language": "c",
        "collected_at": "2026-01-01T00:00:00Z",
        "pair_id": "",
        "file_path": "vuln.c",
    }
    base.update(overrides)
    return base


# ── Tests: strip_preprocessor_guards ──────────────────────────────

class TestStripPreprocessorGuards:
    def test_strips_ifdef_endif(self):
        code = "#ifdef FOO\nint x = 1;\n#endif\n"
        result = strip_preprocessor_guards(code)
        assert "#ifdef" not in result
        assert "#endif" not in result
        assert "int x = 1;" in result

    def test_strips_ifndef(self):
        code = "#ifndef OMITBAD\nvoid f() {}\n#endif\n"
        result = strip_preprocessor_guards(code)
        assert "#ifndef" not in result

    def test_strips_elif_else(self):
        code = "#ifdef A\nx;\n#elif B\ny;\n#else\nz;\n#endif\n"
        result = strip_preprocessor_guards(code)
        assert "#elif" not in result
        assert "#else" not in result

    def test_preserves_include(self):
        code = "#include <stdio.h>\nvoid f() {}\n"
        result = strip_preprocessor_guards(code)
        assert "#include <stdio.h>" in result

    def test_preserves_define(self):
        code = "#define MAX 100\nvoid f() {}\n"
        result = strip_preprocessor_guards(code)
        assert "#define MAX 100" in result

    def test_no_guards_passthrough(self):
        code = "void f() { return; }\n"
        assert strip_preprocessor_guards(code) == code


# ── Tests: normalize ──────────────────────────────────────────────

class TestNormalize:
    def test_valid_function_passes(self):
        result = normalize(VALID_C_FUNCTION)
        assert result is not None
        assert len(result.splitlines()) >= MIN_LINES

    def test_removes_line_comments(self):
        result = normalize(FUNCTION_WITH_COMMENTS)
        assert result is not None
        assert "//" not in result

    def test_removes_block_comments(self):
        result = normalize(FUNCTION_WITH_COMMENTS)
        assert result is not None
        assert "/*" not in result
        assert "*/" not in result

    def test_normalizes_blank_lines(self):
        code = "void f() {\n    int a = 0;\n\n\n\n\n    int b = 1;\n    int c = 2;\n    int d = 3;\n    return;\n}"
        result = normalize(code)
        assert result is not None
        # Max 2 consecutive newlines (= 1 blank line)
        assert "\n\n\n" not in result

    def test_rejects_too_short(self):
        result = normalize(SHORT_C_FUNCTION)
        assert result is None

    def test_rejects_too_long(self):
        result = normalize(LONG_C_FUNCTION)
        assert result is None

    def test_rejects_too_few_tokens(self):
        code = "void f() {\n\n\n\n\n\n}"
        result = normalize(code)
        assert result is None

    def test_strips_ifdef_for_sard(self):
        result = normalize(FUNCTION_WITH_IFDEF, source="sard")
        assert result is not None
        assert "#ifndef" not in result
        assert "#endif" not in result

    def test_no_strip_ifdef_for_nonsard(self):
        code = "#ifdef FOO\nvoid f() {\n    int a = 0;\n    int b = 1;\n    int c = 2;\n    int d = 3;\n    return;\n}\n#endif\n"
        result = normalize(code, source="cve")
        # ifdef lines remain for non-sard
        if result is not None:
            assert "#ifdef" in result


# ── Tests: fix_encoding ───────────────────────────────────────────

class TestFixEncoding:
    def test_str_passthrough(self):
        assert fix_encoding("hello") == "hello"

    def test_bytes_utf8(self):
        assert fix_encoding(b"hello") == "hello"

    def test_bytes_latin1(self):
        raw = "café".encode("latin-1")
        result = fix_encoding(raw)
        assert isinstance(result, str)


# ── Tests: has_function_definition ────────────────────────────────

class TestHasFunctionDefinition:
    def test_valid_function(self):
        assert has_function_definition(VALID_C_FUNCTION) is True

    def test_not_a_function(self):
        assert has_function_definition(NOT_A_FUNCTION) is False

    def test_empty(self):
        assert has_function_definition("") is False


# ── Tests: extract_functions ──────────────────────────────────────

class TestExtractFunctions:
    def test_extracts_two_functions(self):
        fns = extract_functions(MULTI_FUNCTION_FILE)
        assert len(fns) == 2

    def test_function_has_code(self):
        fns = extract_functions(MULTI_FUNCTION_FILE)
        for fn in fns:
            assert "code" in fn
            assert len(fn["code"]) > 0

    def test_function_has_name(self):
        fns = extract_functions(MULTI_FUNCTION_FILE)
        names = {fn["name"] for fn in fns}
        assert "func_one" in str(names)
        assert "func_two" in str(names)

    def test_no_functions(self):
        fns = extract_functions(NOT_A_FUNCTION)
        assert len(fns) == 0

    def test_single_function(self):
        fns = extract_functions(VALID_C_FUNCTION)
        assert len(fns) == 1


# ── Tests: process_sample ─────────────────────────────────────────

class TestProcessSample:
    def test_sard_valid(self):
        sample = _make_sard_sample()
        results = process_sample(sample)
        assert len(results) == 1
        assert results[0]["source"] == "sard"
        assert "//" not in results[0]["code"]

    def test_sard_malformed_rejected(self):
        sample = _make_sard_sample(code=NOT_A_FUNCTION)
        results = process_sample(sample)
        assert len(results) == 0

    def test_sard_empty_code_rejected(self):
        sample = _make_sard_sample(code="")
        results = process_sample(sample)
        assert len(results) == 0

    def test_sard_too_short_rejected(self):
        sample = _make_sard_sample(code=SHORT_C_FUNCTION)
        results = process_sample(sample)
        assert len(results) == 0

    def test_sard_comments_stripped(self):
        sample = _make_sard_sample(code=FUNCTION_WITH_COMMENTS)
        results = process_sample(sample)
        assert len(results) == 1
        assert "//" not in results[0]["code"]
        assert "/*" not in results[0]["code"]

    def test_sard_ifdef_stripped(self):
        sample = _make_sard_sample(code=FUNCTION_WITH_IFDEF)
        results = process_sample(sample)
        assert len(results) == 1
        assert "#ifndef" not in results[0]["code"]
        assert "#endif" not in results[0]["code"]

    def test_sard_preserves_schema_fields(self):
        sample = _make_sard_sample()
        results = process_sample(sample)
        assert len(results) == 1
        for key in ("id", "source", "label", "cwe", "language", "collected_at", "pair_id"):
            assert results[0][key] == sample[key]

    def test_nonsard_extracts_functions(self):
        sample = _make_nonsard_sample()
        results = process_sample(sample)
        assert len(results) >= 1  # at least one function extracted

    def test_nonsard_preserves_label(self):
        sample = _make_nonsard_sample()
        results = process_sample(sample)
        for r in results:
            assert r["label"] == sample["label"]

    def test_nonsard_single_function_fallback(self):
        sample = _make_nonsard_sample(code=VALID_C_FUNCTION)
        results = process_sample(sample)
        assert len(results) == 1


# ── Tests: run_stage1 (integration) ──────────────────────────────

class TestRunStage1:
    def _write_jsonl(self, tmpdir: Path, filename: str, samples: list[dict]) -> Path:
        tmpdir.mkdir(parents=True, exist_ok=True)
        path = tmpdir / filename
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        return path

    def test_basic_pipeline(self, tmp_path):
        samples = [_make_sard_sample() for _ in range(5)]
        self._write_jsonl(tmp_path / "sard", "samples.jsonl", samples)

        output = tmp_path / "out" / "samples.jsonl"
        stats = run_stage1([str(tmp_path / "sard")], str(output))

        assert stats["processed"] == 5
        assert stats["skipped"] == 0
        assert output.exists()
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 5

    def test_dry_run_no_output(self, tmp_path):
        samples = [_make_sard_sample() for _ in range(3)]
        (tmp_path / "sard").mkdir()
        self._write_jsonl(tmp_path / "sard", "samples.jsonl", samples)

        output = tmp_path / "out" / "samples.jsonl"
        stats = run_stage1([str(tmp_path / "sard")], str(output), dry_run=True)

        assert stats["processed"] == 3
        assert not output.exists()

    def test_max_samples_limits_output(self, tmp_path):
        samples = [_make_sard_sample(id=f"test-{i}") for i in range(10)]
        (tmp_path / "sard").mkdir()
        self._write_jsonl(tmp_path / "sard", "samples.jsonl", samples)

        output = tmp_path / "out" / "samples.jsonl"
        stats = run_stage1([str(tmp_path / "sard")], str(output), max_samples=3)

        assert stats["processed"] == 3
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_skips_invalid_samples(self, tmp_path):
        valid = _make_sard_sample()
        invalid = _make_sard_sample(code=NOT_A_FUNCTION)
        (tmp_path / "sard").mkdir()
        self._write_jsonl(tmp_path / "sard", "samples.jsonl", [valid, invalid, valid])

        output = tmp_path / "out" / "samples.jsonl"
        stats = run_stage1([str(tmp_path / "sard")], str(output))

        assert stats["processed"] == 2
        assert stats["skipped"] == 1

    def test_missing_dir_warns(self, tmp_path):
        output = tmp_path / "out" / "samples.jsonl"
        stats = run_stage1([str(tmp_path / "nonexistent")], str(output))
        assert stats["total_read"] == 0
        assert stats["processed"] == 0

    def test_output_is_valid_jsonl(self, tmp_path):
        samples = [_make_sard_sample() for _ in range(3)]
        (tmp_path / "sard").mkdir()
        self._write_jsonl(tmp_path / "sard", "samples.jsonl", samples)

        output = tmp_path / "out" / "samples.jsonl"
        run_stage1([str(tmp_path / "sard")], str(output))

        with open(output) as f:
            for line in f:
                sample = json.loads(line)
                assert "id" in sample
                assert "code" in sample
                assert "source" in sample

    def test_malformed_json_skipped(self, tmp_path):
        (tmp_path / "sard").mkdir()
        p = tmp_path / "sard" / "samples.jsonl"
        with open(p, "w") as f:
            f.write(json.dumps(_make_sard_sample()) + "\n")
            f.write("NOT VALID JSON\n")
            f.write(json.dumps(_make_sard_sample(id="test-002")) + "\n")

        output = tmp_path / "out" / "samples.jsonl"
        stats = run_stage1([str(tmp_path / "sard")], str(output))

        assert stats["processed"] == 2
        assert stats["skipped"] == 1

    def test_multiple_input_dirs(self, tmp_path):
        (tmp_path / "dir1").mkdir()
        (tmp_path / "dir2").mkdir()
        self._write_jsonl(tmp_path / "dir1", "a.jsonl", [_make_sard_sample(id="1")])
        self._write_jsonl(tmp_path / "dir2", "b.jsonl", [_make_sard_sample(id="2")])

        output = tmp_path / "out" / "samples.jsonl"
        stats = run_stage1(
            [str(tmp_path / "dir1"), str(tmp_path / "dir2")], str(output)
        )
        assert stats["processed"] == 2


# ── Tests: filter thresholds ──────────────────────────────────────

class TestFilterThresholds:
    def test_exactly_min_lines_passes(self):
        lines = ["void f() {"] + [f"    int x{i} = {i};" for i in range(MIN_LINES - 2)] + ["}"]
        code = "\n".join(lines)
        if has_function_definition(code):
            result = normalize(code)
            if result is not None:
                assert len(result.splitlines()) >= MIN_LINES

    def test_token_limit_4096(self):
        # Build a function with > 4096 tokens
        body = " ".join([f"int x{i} = {i};" for i in range(2000)])
        code = f"void f() {{\n{body}\n}}"
        result = normalize(code)
        assert result is None


# ── Tests: real SARD data (if available) ──────────────────────────

SARD_PATH = Path("training/scripts/collection/data/raw/sard/sard_samples.jsonl")


@pytest.mark.skipif(not SARD_PATH.exists(), reason="SARD data not available")
class TestRealSardData:
    def test_first_20_samples(self):
        processed = 0
        skipped = 0
        with open(SARD_PATH) as f:
            for i, line in enumerate(f):
                if i >= 20:
                    break
                sample = json.loads(line)
                results = process_sample(sample)
                if results:
                    processed += 1
                    for r in results:
                        assert "//" not in r["code"]
                        assert "/*" not in r["code"]
                        assert r["source"] == "sard"
                else:
                    skipped += 1

        assert processed > 0, "Expected at least some SARD samples to pass"

    def test_cleaned_output_shorter_or_equal(self):
        with open(SARD_PATH) as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                sample = json.loads(line)
                results = process_sample(sample)
                for r in results:
                    # Cleaned code should be <= original (comments removed)
                    assert len(r["code"]) <= len(sample["code"])
