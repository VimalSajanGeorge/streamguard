#!/usr/bin/env python3
"""
tests/test_story_p2s2.py

Tests for P2-S2: ExploitDB Collector.
8 tests covering:
  1. CWE inference: gets( → CWE-120
  2. CWE inference: system( → CWE-78
  3. CWE inference: no match → sample skipped, saved to _no_cwe.jsonl
  4. Rule-based mutation: gets(buf) → fgets(buf, sizeof(buf), stdin)
  5. Compile gate: valid mutation → pair saved with pair_id
  6. Compile gate: invalid mutation → pair_id cleared, original saved alone
  7. Dry-run: no files written
  8. Checkpoint: resume from mid-collection
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root on path for imports
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "training" / "scripts" / "collection")
)

from training.scripts.collection.exploitdb_collector import (
    ExploitDBCollector,
    CWE_KEYWORD_MAP,
    MUTATION_RULES,
)


# ── Helpers ───────────────────────────────────────────────────────────────

# Minimal valid C code (>= 5 non-blank lines, >= 10 tokens)
VULN_CODE_GETS = """\
#include <stdio.h>
#include <string.h>
void vulnerable_function(void) {
    char buffer[64];
    gets(buffer);
    printf("Input: %s\\n", buffer);
    return;
}
"""

VULN_CODE_SYSTEM = """\
#include <stdlib.h>
#include <string.h>
void run_command(const char *cmd) {
    char full_cmd[256];
    sprintf(full_cmd, "/bin/sh -c '%s'", cmd);
    system(full_cmd);
    return;
}
"""

VULN_CODE_NO_CWE = """\
#include <stdio.h>
#include <math.h>
double compute(double x) {
    double result = sin(x) * cos(x);
    printf("Result: %f\\n", result);
    return result;
}
"""

VULN_CODE_STRCPY = """\
#include <stdio.h>
#include <string.h>
void copy_data(const char *src) {
    char dst[128];
    strcpy(dst, src);
    printf("Copied: %s\\n", dst);
    return;
}
"""


def _make_exploitdb_repo(tmpdir: Path, entries: list[dict]) -> Path:
    """
    Create a minimal exploitdb repo structure:
        tmpdir/
            files_exploits.csv
            exploits/
                <platform>/<type>/<id>.c
    """
    repo_root = tmpdir / "exploitdb_repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    # Write CSV
    fieldnames = ["id", "file", "description", "date_published", "author",
                  "platform", "type", "port"]
    csv_path = repo_root / "files_exploits.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry)

    # Write exploit files
    for entry in entries:
        file_path = repo_root / entry["file"]
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(entry.get("_code", ""), encoding="utf-8")

    return repo_root


def _make_entry(
    exploit_id: str,
    platform: str,
    exploit_type: str,
    description: str,
    code: str,
    filename: str | None = None,
) -> dict:
    """Build a CSV row dict + attach code for file creation."""
    if filename is None:
        filename = f"exploits/{platform}/{exploit_type}/{exploit_id}.c"
    return {
        "id": exploit_id,
        "file": filename,
        "description": description,
        "date_published": "2024-01-01",
        "author": "test",
        "platform": platform,
        "type": exploit_type,
        "port": "",
        "_code": code,
    }


# ── Test 1: CWE inference — gets( → CWE-120 ─────────────────────────────

def test_cwe_inference_gets():
    """gets( in code should infer CWE-120."""
    result = ExploitDBCollector.infer_cwe("buffer overflow exploit", VULN_CODE_GETS)
    assert result == "CWE-120", f"Expected CWE-120, got {result}"


# ── Test 2: CWE inference — system( → CWE-78 ────────────────────────────

def test_cwe_inference_system():
    """system( in code should infer CWE-78."""
    # Description has no CWE-120 keywords, code has system(
    result = ExploitDBCollector.infer_cwe("command execution", VULN_CODE_SYSTEM)
    # sprintf( in code triggers CWE-120 first (priority order), but
    # since sprintf( is checked under CWE-120 before system( under CWE-78,
    # and priority is CWE-120 first, let's verify the actual behavior.
    # The code has sprintf( which matches CWE-120 first in priority.
    # To test CWE-78 specifically, use code WITHOUT any CWE-120 keywords.
    code_system_only = """\
#include <stdlib.h>
void run_cmd(const char *cmd) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%s", cmd);
    system(buf);
    return;
}
"""
    result = ExploitDBCollector.infer_cwe("command execution", code_system_only)
    assert result == "CWE-78", f"Expected CWE-78, got {result}"


# ── Test 3: CWE inference — no match → skipped, saved to _no_cwe.jsonl ──

def test_cwe_no_match_saved_to_no_cwe_jsonl():
    """Samples with no CWE match should be skipped and logged to _no_cwe.jsonl."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"

        entries = [
            _make_entry("9001", "linux", "local", "math computation",
                        VULN_CODE_NO_CWE),
        ]
        repo = _make_exploitdb_repo(tmpdir, entries)

        collector = ExploitDBCollector(
            exploitdb_root=str(repo),
            output_dir=str(output_dir),
            gcc_path="__nonexistent_gcc__",  # force gcc unavailable
        )
        summary = collector.collect()

        assert summary["cwe_no_match"] == 1
        assert summary["samples_saved"] == 0

        # Check _no_cwe.jsonl was created
        no_cwe_path = output_dir / "exploitdb_no_cwe.jsonl"
        assert no_cwe_path.exists(), "exploitdb_no_cwe.jsonl should exist"
        with open(no_cwe_path, encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 1
        assert lines[0]["exploit_id"] == "9001"


# ── Test 4: Rule-based mutation — gets(buf) → fgets(...) ────────────────

def test_mutation_gets_to_fgets():
    """gets(buf) should be mutated to fgets(buf, sizeof(buf), stdin)."""
    code = "char buf[64]; gets(buf);"
    mutated, was_mutated = ExploitDBCollector.apply_mutations(code)
    assert was_mutated is True
    assert "fgets(buf, sizeof(buf), stdin)" in mutated
    assert "gets(buf)" not in mutated


def test_mutation_strcpy_to_strncpy():
    """strcpy(dst, src) should be mutated to strncpy with null terminator."""
    code = "char dst[128]; strcpy(dst, src);"
    mutated, was_mutated = ExploitDBCollector.apply_mutations(code)
    assert was_mutated is True
    assert "strncpy(dst, src, sizeof(dst)-1)" in mutated
    assert "dst[sizeof(dst)-1] = '\\0'" in mutated


def test_mutation_sprintf_to_snprintf():
    """sprintf(buf, ...) should be mutated to snprintf(buf, sizeof(buf), ...)."""
    code = 'sprintf(buf, "%s", input);'
    mutated, was_mutated = ExploitDBCollector.apply_mutations(code)
    assert was_mutated is True
    assert "snprintf(buf, sizeof(buf)," in mutated


def test_mutation_strcat_to_strncat():
    """strcat(dst, src) should be mutated to strncat(...)."""
    code = "strcat(dst, src);"
    mutated, was_mutated = ExploitDBCollector.apply_mutations(code)
    assert was_mutated is True
    assert "strncat(dst, src, sizeof(dst)-strlen(dst)-1)" in mutated


def test_mutation_no_match():
    """Code with no vulnerable patterns should not be mutated."""
    code = "int x = 42; printf(\"%d\\n\", x);"
    mutated, was_mutated = ExploitDBCollector.apply_mutations(code)
    assert was_mutated is False
    assert mutated == code


# ── Test 5: Compile gate — valid mutation → pair with pair_id ────────────

def test_compile_gate_valid_pair():
    """When gcc is available and both compile, save as pair with shared pair_id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"

        entries = [
            _make_entry("1001", "linux", "local",
                        "buffer overflow via gets",
                        VULN_CODE_GETS),
        ]
        repo = _make_exploitdb_repo(tmpdir, entries)

        # Mock gcc as always passing
        def mock_compile_check(self_arg, code):
            return True

        with patch.object(ExploitDBCollector, '_compile_check', mock_compile_check):
            collector = ExploitDBCollector(
                exploitdb_root=str(repo),
                output_dir=str(output_dir),
                gcc_path="gcc",
            )
            # Force gcc available for test
            collector._gcc_available = True

            summary = collector.collect()

        # Should have 2 samples (vuln + safe pair)
        assert summary["samples_saved"] == 2
        assert summary["valid_pairs"] == 1

        # Read JSONL and verify pair_id linkage
        samples_path = output_dir / "exploitdb_samples.jsonl"
        assert samples_path.exists()
        with open(samples_path, encoding="utf-8") as f:
            samples = [json.loads(l) for l in f if l.strip()]

        assert len(samples) == 2
        pair_ids = {s["pair_id"] for s in samples}
        assert len(pair_ids) == 1, "Both samples should share the same pair_id"
        pair_id = pair_ids.pop()
        assert pair_id != "", "pair_id should not be empty"

        labels = {s["label"] for s in samples}
        assert labels == {0, 1}, "Should have one vuln (1) and one safe (0)"

        # Verify mutation was applied in safe version
        safe = [s for s in samples if s["label"] == 0][0]
        assert "fgets(" in safe["code"], "Safe version should use fgets"


# ── Test 6: Compile gate — invalid mutation → original saved alone ───────

def test_compile_gate_invalid_mutation():
    """When mutated code fails gcc, save original alone with empty pair_id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"

        entries = [
            _make_entry("2001", "linux", "remote",
                        "buffer overflow via gets",
                        VULN_CODE_GETS),
        ]
        repo = _make_exploitdb_repo(tmpdir, entries)

        # Mock: original passes, mutated fails
        call_count = [0]

        def mock_compile_check(self_arg, code):
            call_count[0] += 1
            if "fgets(" in code:
                return False  # mutated version fails
            return True  # original passes

        with patch.object(ExploitDBCollector, '_compile_check', mock_compile_check):
            collector = ExploitDBCollector(
                exploitdb_root=str(repo),
                output_dir=str(output_dir),
                gcc_path="gcc",
            )
            collector._gcc_available = True

            summary = collector.collect()

        # Should save only original (1 sample), no pair
        assert summary["samples_saved"] == 1
        assert summary["valid_pairs"] == 0
        assert summary["mutations_failed"] == 1

        samples_path = output_dir / "exploitdb_samples.jsonl"
        with open(samples_path, encoding="utf-8") as f:
            samples = [json.loads(l) for l in f if l.strip()]

        assert len(samples) == 1
        assert samples[0]["label"] == 1
        assert samples[0]["pair_id"] == ""

        # Mutation failure should be logged
        fail_path = output_dir / "exploitdb_mutation_failed.jsonl"
        assert fail_path.exists()


# ── Test 7: Dry-run — no files written ───────────────────────────────────

def test_dry_run_no_files_written():
    """In dry-run mode, no JSONL output files should be created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"

        entries = [
            _make_entry("3001", "linux", "local",
                        "buffer overflow via gets",
                        VULN_CODE_GETS),
            _make_entry("3002", "unix", "remote",
                        "math computation",
                        VULN_CODE_NO_CWE),
        ]
        repo = _make_exploitdb_repo(tmpdir, entries)

        collector = ExploitDBCollector(
            exploitdb_root=str(repo),
            output_dir=str(output_dir),
            dry_run=True,
            gcc_path="__nonexistent_gcc__",
        )
        summary = collector.collect()

        # Should have processed and counted
        assert summary["samples_saved"] >= 1

        # No JSONL output files should exist
        samples_path = output_dir / "exploitdb_samples.jsonl"
        assert not samples_path.exists(), "No JSONL should be written in dry-run"

        no_cwe_path = output_dir / "exploitdb_no_cwe.jsonl"
        assert not no_cwe_path.exists(), "No _no_cwe.jsonl should be written in dry-run"


# ── Test 8: Checkpoint — resume from mid-collection ──────────────────────

def test_checkpoint_resume():
    """Collector should resume from checkpoint, skipping already-processed rows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"
        checkpoint_dir = tmpdir / "checkpoints"

        entries = [
            _make_entry("4001", "linux", "local",
                        "buffer overflow via gets",
                        VULN_CODE_GETS),
            _make_entry("4002", "linux", "local",
                        "strcpy overflow",
                        VULN_CODE_STRCPY),
            _make_entry("4003", "linux", "local",
                        "another gets overflow",
                        VULN_CODE_GETS.replace("buffer", "mybuf")),
        ]
        repo = _make_exploitdb_repo(tmpdir, entries)

        # First run: process only first entry (max_samples=1)
        collector1 = ExploitDBCollector(
            exploitdb_root=str(repo),
            output_dir=str(output_dir),
            checkpoint_dir=str(checkpoint_dir),
            max_samples=1,
            gcc_path="__nonexistent_gcc__",
        )
        summary1 = collector1.collect()
        assert summary1["samples_saved"] == 1

        # Manually write a checkpoint that says we processed index 0
        import json as json_mod
        cp_path = checkpoint_dir / "exploitdb_checkpoint.json"
        cp_path.write_text(json_mod.dumps({"last_processed_idx": 0, "samples_saved": 1}),
                           encoding="utf-8")

        # Second run: should resume from index 1
        collector2 = ExploitDBCollector(
            exploitdb_root=str(repo),
            output_dir=str(output_dir),
            checkpoint_dir=str(checkpoint_dir),
            gcc_path="__nonexistent_gcc__",
        )
        summary2 = collector2.collect()

        # Second run should process entries at index 1 and 2
        # (entry at index 0 was already processed, skipped by checkpoint)
        # Total saved across both runs should be >= 2 (dedup may reduce)
        assert summary2["samples_saved"] >= 1, "Should have processed remaining entries"


# ── Test: CSV platform/type filtering ────────────────────────────────────

def test_csv_filtering():
    """Only linux/unix/freebsd/multiple platforms and local/remote/dos types pass."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"

        entries = [
            # Should pass
            _make_entry("5001", "linux", "local", "gets overflow", VULN_CODE_GETS),
            # Should be filtered: wrong platform
            _make_entry("5002", "windows", "local", "gets overflow", VULN_CODE_GETS,
                        filename="exploits/windows/local/5002.c"),
            # Should be filtered: wrong type
            _make_entry("5003", "linux", "webapps", "gets overflow", VULN_CODE_GETS,
                        filename="exploits/linux/webapps/5003.c"),
            # Should be filtered: not a .c file
            {
                "id": "5004", "file": "exploits/linux/local/5004.py",
                "description": "python exploit", "date_published": "2024-01-01",
                "author": "test", "platform": "linux", "type": "local", "port": "",
                "_code": "print('hello')",
            },
        ]
        repo = _make_exploitdb_repo(tmpdir, entries)

        collector = ExploitDBCollector(
            exploitdb_root=str(repo),
            output_dir=str(output_dir),
            gcc_path="__nonexistent_gcc__",
        )
        summary = collector.collect()

        assert summary["csv_rows_filtered"] == 1, "Only one .c file on valid platform/type"


# ── Test: max_samples limit ──────────────────────────────────────────────

def test_max_samples_limit():
    """Collector should stop after saving max_samples."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"

        entries = [
            _make_entry("6001", "linux", "local", "gets overflow",
                        VULN_CODE_GETS),
            _make_entry("6002", "linux", "local", "strcpy overflow",
                        VULN_CODE_STRCPY),
            _make_entry("6003", "freebsd", "remote", "another gets",
                        VULN_CODE_GETS.replace("buffer", "buf2")),
        ]
        repo = _make_exploitdb_repo(tmpdir, entries)

        collector = ExploitDBCollector(
            exploitdb_root=str(repo),
            output_dir=str(output_dir),
            max_samples=1,
            gcc_path="__nonexistent_gcc__",
        )
        summary = collector.collect()

        assert summary["samples_saved"] == 1


# ── Test: schema validation enforced ─────────────────────────────────────

def test_schema_validation():
    """Samples with too-short code should be rejected by schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"

        # Code has gets( so CWE will be inferred, but < 5 non-blank lines
        short_code = "void f() {\n  gets(buf);\n}\n"

        entries = [
            _make_entry("7001", "linux", "local", "gets overflow", short_code),
        ]
        repo = _make_exploitdb_repo(tmpdir, entries)

        collector = ExploitDBCollector(
            exploitdb_root=str(repo),
            output_dir=str(output_dir),
            gcc_path="__nonexistent_gcc__",
        )
        summary = collector.collect()

        assert summary["samples_saved"] == 0
        assert summary["samples_failed"] == 1


# ── Test: source field is "exploitdb" ────────────────────────────────────

def test_source_field():
    """All samples should have source='exploitdb'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"

        entries = [
            _make_entry("8001", "linux", "local", "gets overflow", VULN_CODE_GETS),
        ]
        repo = _make_exploitdb_repo(tmpdir, entries)

        collector = ExploitDBCollector(
            exploitdb_root=str(repo),
            output_dir=str(output_dir),
            gcc_path="__nonexistent_gcc__",
        )
        collector.collect()

        samples_path = output_dir / "exploitdb_samples.jsonl"
        with open(samples_path, encoding="utf-8") as f:
            samples = [json.loads(l) for l in f if l.strip()]

        for s in samples:
            assert s["source"] == "exploitdb"
            assert s["language"] == "c"
            assert s["label"] in (0, 1)


# ── Test: CWE priority order ────────────────────────────────────────────

def test_cwe_priority_order():
    """CWE-120 keywords take priority over CWE-78 when both present."""
    # Code with both strcpy( (CWE-120) and system( (CWE-78)
    code = """\
#include <stdlib.h>
void f(const char *s) {
    char buf[64];
    strcpy(buf, s);
    system(buf);
}
"""
    result = ExploitDBCollector.infer_cwe("", code)
    assert result == "CWE-120", f"CWE-120 should take priority, got {result}"


# ── Test: no gcc → no pairs, originals saved ────────────────────────────

def test_no_gcc_saves_originals_only():
    """Without gcc, only original vulnerable samples are saved (no pairs)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"

        entries = [
            _make_entry("9101", "linux", "local", "gets overflow", VULN_CODE_GETS),
        ]
        repo = _make_exploitdb_repo(tmpdir, entries)

        collector = ExploitDBCollector(
            exploitdb_root=str(repo),
            output_dir=str(output_dir),
            gcc_path="__nonexistent_gcc__",
        )
        assert collector._gcc_available is False

        summary = collector.collect()

        # Should save original only, no pairs
        assert summary["samples_saved"] == 1
        assert summary["valid_pairs"] == 0
        assert summary["compile_gate_skipped"] == 1

        samples_path = output_dir / "exploitdb_samples.jsonl"
        with open(samples_path, encoding="utf-8") as f:
            samples = [json.loads(l) for l in f if l.strip()]
        assert len(samples) == 1
        assert samples[0]["label"] == 1
        assert samples[0]["pair_id"] == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
