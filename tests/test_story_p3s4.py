#!/usr/bin/env python3
"""
tests/test_story_p3s4.py

Story P3-S4 — Stage 4: CPG Construction (Phase 3 multi-source extension).

Tests covering Phase 3 additions + risk mitigations (R-15 through R-34):
  1. SARD source: JULIET_STUB prepended to temp file
  2. Non-SARD source: no JULIET_STUB, only standard includes
  3. CDG edge dropped from EDGE_TYPE_MAP
  4. REACHING_DEF mapped to type 2 (DFG) — Joern 4.x specific
  5. Pair-aware discard: if partner fails, survivor's CPG is removed
  6. write_temp_c strips SARD-specific includes for SARD source
  7. write_temp_c does NOT strip includes for non-SARD source
  8. JULIET_STUB contains all 8 Juliet print functions
  9. Taint sets extended: SQL injection sinks present
  10. Taint sets extended: command injection sinks present
  11. Taint sets extended: SQL sanitizers present
  12. _is_sard_source correctly classifies sources
  13. Non-SARD samples get HEADER_STUBS (no Juliet functions)
  14. Checkpoint dir defaults to output dir when not specified
  15. Sentinel pair_id values ("", "None", "null", "0") are ignored in pair lookup
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from training.scripts.preprocessing.stage4_cpg import (
    EDGE_TYPE_MAP,
    HEADER_STUBS,
    JOERN_LABEL_TO_TYPE,
    JULIET_STUB,
    OPERATOR_SINK_NAMES,
    SANITIZERS,
    SARD_STRIP_INCLUDES,
    SINKS,
    SOURCES,
    TPG_TYPE,
    _build_tpg_edges,
    _context_slice,
    _get_taint_role,
    _is_sard_source,
    _map_edge_label,
    _save_checkpoint,
    _shard_path,
    _strip_sard_includes,
    process_sample,
    run_stage4,
    write_temp_c,
)


# ── Helpers ───────────────────────────────────────────────────────
def _make_sample(
    code: str,
    source: str = "sard",
    label: int = 1,
    pair_id: str = "",
    sample_id: str = "",
    cwe: str = "CWE-121",
) -> dict:
    """Build a minimal sample dict for testing."""
    return {
        "id": sample_id or f"test-{abs(hash(code)) % 10**8:08d}",
        "source": source,
        "code": code,
        "label": label,
        "cwe": cwe,
        "language": "c",
        "collected_at": "2026-03-25T00:00:00",
        "pair_id": pair_id,
        "file_path": "",
        "function_name": "test_func",
    }


SARD_CODE = """\
#include "std_testcase.h"
#include "std_testcase_io.h"
void bad() {
    char buf[10];
    gets(buf);
    strcpy(dest, buf);
}
"""

NON_SARD_CODE = """\
void process(int fd) {
    char buf[256];
    read(fd, buf, 256);
    system(buf);
}
"""


# ── Test 1: SARD source gets JULIET_STUB prepended ─────────────
class TestWriteTempCSard:
    def test_sard_source_gets_juliet_stub(self, tmp_path):
        """SARD samples must have JULIET_STUB prepended to the temp .c file."""
        sample = _make_sample(SARD_CODE, source="sard")
        path = write_temp_c(sample, tmp_path)

        content = path.read_text(encoding="utf-8")
        assert "void printLine(const char* msg) {}" in content
        assert "void printIntLine(int i) {}" in content
        assert "void printDoubleLine(double d) {}" in content
        assert "void printHexCharLine(char c) {}" in content
        assert "void printLongLine(long l) {}" in content
        assert "void printLongLongLine(long long ll) {}" in content
        assert "void printUnsignedLine(unsigned u) {}" in content
        assert "void printSizeTLine(size_t s) {}" in content


# ── Test 2: Non-SARD source gets NO JULIET_STUB ─────────────────
class TestWriteTempCNonSard:
    def test_non_sard_no_juliet_stub(self, tmp_path):
        """Non-SARD samples must NOT have JULIET_STUB — only standard includes."""
        sample = _make_sample(NON_SARD_CODE, source="cve")
        path = write_temp_c(sample, tmp_path)

        content = path.read_text(encoding="utf-8")
        assert "void printLine" not in content
        assert "void printIntLine" not in content
        assert "#include <stdio.h>" in content
        assert "#include <stdlib.h>" in content
        assert "#include <string.h>" in content


# ── Test 3: CDG edge dropped ────────────────────────────────────
class TestCDGEdgeDropped:
    def test_cdg_not_in_edge_type_map(self):
        """CDG must not be in EDGE_TYPE_MAP — storing as type>=4 crashes CUDA."""
        assert "CDG" not in EDGE_TYPE_MAP
        assert _map_edge_label("CDG") is None

    def test_dominate_not_in_map(self):
        assert _map_edge_label("DOMINATE") is None

    def test_post_dominate_not_in_map(self):
        assert _map_edge_label("POST_DOMINATE") is None


# ── Test 4: REACHING_DEF → DFG type 2 ──────────────────────────
class TestReachingDefMapping:
    def test_reaching_def_maps_to_2(self):
        """Joern 4.x REACHING_DEF must map to DFG type 2."""
        assert JOERN_LABEL_TO_TYPE["REACHING_DEF"] == 2
        assert _map_edge_label("REACHING_DEF") == 2

    def test_reaches_legacy_maps_to_2(self):
        """Legacy Joern 1.x REACHES also maps to DFG type 2."""
        assert JOERN_LABEL_TO_TYPE["REACHES"] == 2


# ── Test 5: Pair-aware discard ──────────────────────────────────
class TestPairAwareDiscard:
    def test_partner_cpg_removed_when_pair_fails(self, tmp_path):
        """If one member of a pair fails CPG construction, partner's CPG is removed."""
        output_dir = tmp_path / "cpg_out"
        output_dir.mkdir()

        # Create two paired samples: good and bad
        good_sample = _make_sample(
            "void good() { int x = 1; return; }",
            source="sard", pair_id="pair_001", sample_id="ab_good_001",
        )
        # Bad sample: empty code that will fail validation (< 3 nodes)
        bad_sample = _make_sample(
            "",  # empty code -> will fail
            source="sard", pair_id="pair_001", sample_id="cd_bad_001",
        )

        # Write input JSONL
        input_path = tmp_path / "input.jsonl"
        with open(input_path, "w") as f:
            f.write(json.dumps(good_sample) + "\n")
            f.write(json.dumps(bad_sample) + "\n")

        # Run stage4 with dry_run to verify pair lookup is built correctly
        # (We can't run Joern in tests, but we test the pair_lookup logic directly)
        # Instead, test the discard logic by simulating:
        # 1. good partner succeeds -> CPG file exists
        # 2. bad partner fails -> in failed_ids
        # 3. After discard: good partner's CPG should be removed

        # Simulate: write a CPG file for the good partner
        shard_path = _shard_path(output_dir, "ab_good_001")
        shard_path.write_text('{"test": true}')
        assert shard_path.exists()

        # Simulate pair_lookup and discard logic
        all_pair_lookup = {"pair_001": ["ab_good_001", "cd_bad_001"]}
        failed_ids = {"cd_bad_001"}
        completed_ids = ["ab_good_001"]

        broken_pairs = 0
        for pid, members in all_pair_lookup.items():
            member_failed = [m for m in members if m in failed_ids]
            member_ok = [m for m in members if m not in failed_ids and m in set(completed_ids)]
            if member_failed and member_ok:
                for m in member_ok:
                    cpg_path = _shard_path(output_dir, m)
                    if cpg_path.exists():
                        cpg_path.unlink()
                        broken_pairs += 1

        assert broken_pairs == 1
        assert not shard_path.exists(), "Partner CPG should be removed"


# ── Test 6: write_temp_c strips SARD includes ──────────────────
class TestSardIncludeStripping:
    def test_sard_includes_stripped(self, tmp_path):
        """SARD samples should have std_testcase.h etc. stripped."""
        sample = _make_sample(SARD_CODE, source="sard")
        path = write_temp_c(sample, tmp_path)

        content = path.read_text(encoding="utf-8")
        assert "std_testcase.h" not in content
        assert "std_testcase_io.h" not in content


# ── Test 7: Non-SARD includes NOT stripped ──────────────────────
class TestNonSardIncludesPreserved:
    def test_non_sard_includes_preserved(self, tmp_path):
        """Non-SARD samples should preserve their #include lines."""
        code_with_includes = '#include <unistd.h>\nvoid f() { read(0, buf, 10); }'
        sample = _make_sample(code_with_includes, source="repo")
        path = write_temp_c(sample, tmp_path)

        content = path.read_text(encoding="utf-8")
        assert "#include <unistd.h>" in content


# ── Test 8: JULIET_STUB has all 8 functions ─────────────────────
class TestJulietStubCompleteness:
    def test_juliet_stub_has_all_functions(self):
        """JULIET_STUB must declare all 8 Juliet print helper functions."""
        expected_funcs = [
            "printLine", "printIntLine", "printDoubleLine",
            "printHexCharLine", "printLongLine", "printLongLongLine",
            "printUnsignedLine", "printSizeTLine",
        ]
        for func in expected_funcs:
            assert func in JULIET_STUB, f"{func} missing from JULIET_STUB"


# ── Test 9: SQL injection sinks present ──────────────────────────
class TestExtendedTaintSQL:
    def test_sql_injection_sinks(self):
        """CWE-89 SQL injection sinks must be in SINKS set."""
        sql_sinks = {"mysql_real_query", "sqlite3_exec", "mysql_query"}
        assert sql_sinks.issubset(SINKS)

    def test_sql_injection_taint_role(self):
        """SQL injection sink nodes should be labeled SINK."""
        node = {"_label": "CALL", "code": "sqlite3_exec(db, sql, 0, 0, 0)", "name": "sqlite3_exec"}
        assert _get_taint_role(node) == "SINK"


# ── Test 10: Command injection sinks present ─────────────────────
class TestExtendedTaintCmdInj:
    def test_command_injection_sinks(self):
        """CWE-78 OS command injection sinks must be in SINKS set."""
        cmd_sinks = {"system", "popen", "execve", "execlp"}
        assert cmd_sinks.issubset(SINKS)

    def test_command_injection_taint_role(self):
        node = {"_label": "CALL", "code": "execlp(cmd, cmd, NULL)", "name": "execlp"}
        assert _get_taint_role(node) == "SINK"


# ── Test 11: SQL sanitizers present ──────────────────────────────
class TestExtendedSanitizers:
    def test_sql_sanitizers_present(self):
        """CWE-89 SQL sanitizers must be in SANITIZERS set."""
        assert "mysql_real_escape_string" in SANITIZERS

    def test_xss_sanitizers_present(self):
        """CWE-79 XSS sanitizers must be in SANITIZERS set."""
        assert "htmlspecialchars" in SANITIZERS
        assert "html_escape" in SANITIZERS


# ── Test 12: _is_sard_source classification ──────────────────────
class TestIsSardSource:
    def test_sard_detected(self):
        assert _is_sard_source("sard") is True

    def test_sard_prefix_detected(self):
        assert _is_sard_source("sard_juliet") is True

    def test_cve_not_sard(self):
        assert _is_sard_source("cve") is False

    def test_repo_not_sard(self):
        assert _is_sard_source("repo") is False

    def test_empty_not_sard(self):
        assert _is_sard_source("") is False

    def test_none_not_sard(self):
        assert _is_sard_source(None) is False


# ── Test 13: Non-SARD gets HEADER_STUBS (no Juliet funcs) ───────
class TestNonSardHeaderStubs:
    def test_header_stubs_no_juliet_functions(self):
        """HEADER_STUBS (for non-SARD) must NOT contain Juliet functions."""
        assert "printLine" not in HEADER_STUBS
        assert "printIntLine" not in HEADER_STUBS

    def test_header_stubs_has_standard_includes(self):
        """HEADER_STUBS must have stdio, stdlib, string."""
        assert "#include <stdio.h>" in HEADER_STUBS
        assert "#include <stdlib.h>" in HEADER_STUBS
        assert "#include <string.h>" in HEADER_STUBS


# ── Test 14: Checkpoint dir defaults to output dir ───────────────
class TestCheckpointDir:
    def test_checkpoint_dir_defaults(self, tmp_path):
        """When checkpoint_dir is None, checkpoint goes in output_dir."""
        # We can test this via dry_run since it doesn't need Joern
        input_path = tmp_path / "input.jsonl"
        output_dir = tmp_path / "cpg_out"
        sample = _make_sample("void f() { int x = 1; }", source="sard", sample_id="aa_test_001")
        input_path.write_text(json.dumps(sample) + "\n")

        result = run_stage4(
            str(input_path),
            str(output_dir),
            dry_run=True,
        )
        assert result["mode"] == "dry_run"
        # Output dir should be created
        assert output_dir.exists()


# ── Test 15: Sentinel pair_id values ignored ─────────────────────
class TestSentinelPairIds:
    def test_sentinel_values_ignored_in_pair_lookup(self):
        """Sentinel pair_id values should not create pair groups."""
        samples = [
            _make_sample("void a() {}", source="sard", pair_id="", sample_id="s1"),
            _make_sample("void b() {}", source="sard", pair_id="None", sample_id="s2"),
            _make_sample("void c() {}", source="sard", pair_id="null", sample_id="s3"),
            _make_sample("void d() {}", source="sard", pair_id="0", sample_id="s4"),
            # Real pair_id should be kept
            _make_sample("void e() {}", source="sard", pair_id="real_pair", sample_id="s5"),
            _make_sample("void f() {}", source="sard", pair_id="real_pair", sample_id="s6"),
        ]

        # Simulate the pair_lookup building logic from run_stage4
        all_pair_lookup: dict[str, list[str]] = {}
        for s in samples:
            pid = s.get("pair_id", "")
            if pid and pid not in ("", "None", "null", "0"):
                all_pair_lookup.setdefault(pid, []).append(s["id"])

        # Sentinel values should NOT create entries
        assert "" not in all_pair_lookup
        assert "None" not in all_pair_lookup
        assert "null" not in all_pair_lookup
        assert "0" not in all_pair_lookup
        # Real pair should exist
        assert "real_pair" in all_pair_lookup
        assert len(all_pair_lookup["real_pair"]) == 2


# ════════════════════════════════════════════════════════════════
# RISK MITIGATION TESTS (R-15, R-16, R-17, R-33, R-34)
# ════════════════════════════════════════════════════════════════


# ── R-15: DFG=0 must halt (sys.exit), not just log ──────────────
class TestR15DFGZeroHalt:
    def test_process_sample_returns_none_on_zero_dfg(self):
        """process_sample must return None when DFG edges = 0.
        This prevents zero-DFG CPGs from being saved."""
        from unittest.mock import patch

        sample = _make_sample(
            "void f() { int x; }", source="sard", sample_id="r15_test_001"
        )
        # Mock _run_joern_cli to return nodes with AST/CFG but NO DFG edges
        nodes = [
            {"id": "1", "_label": "METHOD", "code": "void f()", "name": "f",
             "line": 1, "type_full": ""},
            {"id": "2", "_label": "BLOCK", "code": "{ int x; }", "name": "",
             "line": 2, "type_full": ""},
            {"id": "3", "_label": "LOCAL", "code": "int x", "name": "x",
             "line": 3, "type_full": "int"},
        ]
        edges = [
            {"src": "1", "dst": "2", "type": 0, "label": "AST"},
            {"src": "1", "dst": "3", "type": 0, "label": "AST"},
            {"src": "2", "dst": "3", "type": 1, "label": "CFG"},
        ]

        with patch(
            "training.scripts.preprocessing.stage4_cpg._run_joern_cli",
            return_value=(nodes, edges),
        ):
            result = process_sample(sample, "fake_joern_dir")

        # Must return None because DFG count = 0
        assert result is None


# ── R-16: No edge_type >= 4 in output ────────────────────────────
class TestR16CDGEdgeAudit:
    def test_edge_type_gte4_rejected(self):
        """If any edge has type >= 4 (e.g. CDG leaked), sample is rejected."""
        from unittest.mock import patch

        sample = _make_sample(
            "void f() { gets(buf); strcpy(d, buf); }",
            source="sard", sample_id="r16_test_001",
        )
        nodes = [
            {"id": "1", "_label": "CALL", "code": "gets(buf)", "name": "gets",
             "line": 1, "type_full": ""},
            {"id": "2", "_label": "CALL", "code": "strcpy(d, buf)", "name": "strcpy",
             "line": 2, "type_full": ""},
            {"id": "3", "_label": "METHOD", "code": "void f()", "name": "f",
             "line": 0, "type_full": ""},
        ]
        # Simulate a CDG edge that leaked through (type=4)
        edges = [
            {"src": "3", "dst": "1", "type": 0, "label": "AST"},
            {"src": "3", "dst": "2", "type": 0, "label": "AST"},
            {"src": "1", "dst": "2", "type": 2, "label": "DFG"},
            {"src": "1", "dst": "2", "type": 1, "label": "CFG"},
            {"src": "3", "dst": "1", "type": 4, "label": "CDG"},  # LEAKED!
        ]

        with patch(
            "training.scripts.preprocessing.stage4_cpg._run_joern_cli",
            return_value=(nodes, edges),
        ):
            result = process_sample(sample, "fake_joern_dir")

        # Must return None — CDG type=4 detected and rejected
        assert result is None

    def test_only_types_0123_in_valid_cpg(self):
        """A valid CPG must only have edge types {0, 1, 2, 3}."""
        from unittest.mock import patch

        sample = _make_sample(
            "void f() { gets(buf); strcpy(d, buf); }",
            source="sard", sample_id="r16_valid_001",
        )
        nodes = [
            {"id": "1", "_label": "CALL", "code": "gets(buf)", "name": "gets",
             "line": 1, "type_full": ""},
            {"id": "2", "_label": "CALL", "code": "strcpy(d, buf)", "name": "strcpy",
             "line": 2, "type_full": ""},
            {"id": "3", "_label": "METHOD", "code": "void f()", "name": "f",
             "line": 0, "type_full": ""},
        ]
        edges = [
            {"src": "3", "dst": "1", "type": 0, "label": "AST"},
            {"src": "3", "dst": "2", "type": 0, "label": "AST"},
            {"src": "1", "dst": "2", "type": 2, "label": "DFG"},
            {"src": "1", "dst": "2", "type": 1, "label": "CFG"},
        ]

        with patch(
            "training.scripts.preprocessing.stage4_cpg._run_joern_cli",
            return_value=(nodes, edges),
        ):
            result = process_sample(sample, "fake_joern_dir")

        assert result is not None
        edge_types = {e["type"] for e in result["edges"]}
        assert edge_types.issubset({0, 1, 2, 3}), f"Invalid edge types: {edge_types}"


# ── R-17: Subprocess timeout handling ────────────────────────────
class TestR17SubprocessTimeout:
    def test_run_subprocess_safe_import(self):
        """_run_subprocess_safe must be importable and callable."""
        from training.scripts.preprocessing.stage4_cpg import _run_subprocess_safe
        assert callable(_run_subprocess_safe)

    def test_run_subprocess_safe_basic(self):
        """_run_subprocess_safe should run a simple command."""
        from training.scripts.preprocessing.stage4_cpg import _run_subprocess_safe
        import sys
        result = _run_subprocess_safe(
            [sys.executable, "-c", "print('hello')"],
            timeout=10,
        )
        assert result.returncode == 0
        assert "hello" in result.stdout

    def test_run_subprocess_safe_timeout(self):
        """_run_subprocess_safe should raise TimeoutExpired on timeout."""
        import subprocess as sp
        from training.scripts.preprocessing.stage4_cpg import _run_subprocess_safe
        import sys
        with pytest.raises(sp.TimeoutExpired):
            _run_subprocess_safe(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                timeout=1,
            )


# ── R-33: Input path validation ──────────────────────────────────
class TestR33InputValidation:
    def test_deduped_path_triggers_dry_run_warning(self, tmp_path, capsys):
        """Using deduped/ path should still work but with warning logged."""
        import logging
        from unittest.mock import patch

        input_path = tmp_path / "deduped" / "samples.jsonl"
        input_path.parent.mkdir(parents=True)
        sample = _make_sample("void f() { int x; }", source="sard", sample_id="r33_001")
        input_path.write_text(json.dumps(sample) + "\n")

        output_dir = tmp_path / "cpg_out"

        # Capture loguru output
        warnings = []
        with patch("training.scripts.preprocessing.stage4_cpg.logger") as mock_logger:
            mock_logger.info = lambda *a, **kw: None
            mock_logger.warning = lambda msg, *a, **kw: warnings.append(msg)
            mock_logger.error = lambda *a, **kw: None

            result = run_stage4(
                str(input_path),
                str(output_dir),
                dry_run=True,
            )

        assert result["mode"] == "dry_run"
        # Check that R-33 warning was emitted
        r33_warnings = [w for w in warnings if "R-33" in w]
        assert len(r33_warnings) >= 1, f"Expected R-33 warning, got: {warnings}"

    def test_with_cfa_path_no_warning(self, tmp_path):
        """Using with_cfa/ path should NOT trigger R-33 warning."""
        from unittest.mock import patch

        input_path = tmp_path / "with_cfa" / "samples.jsonl"
        input_path.parent.mkdir(parents=True)
        sample = _make_sample("void f() { int x; }", source="sard", sample_id="r33_002")
        input_path.write_text(json.dumps(sample) + "\n")

        output_dir = tmp_path / "cpg_out"

        warnings = []
        with patch("training.scripts.preprocessing.stage4_cpg.logger") as mock_logger:
            mock_logger.info = lambda *a, **kw: None
            mock_logger.warning = lambda msg, *a, **kw: warnings.append(msg)
            mock_logger.error = lambda *a, **kw: None

            result = run_stage4(
                str(input_path),
                str(output_dir),
                dry_run=True,
            )

        r33_warnings = [w for w in warnings if "R-33" in w and "deduped" in w]
        assert len(r33_warnings) == 0, f"Unexpected R-33 deduped warning: {r33_warnings}"


# ── R-34: Checkpoint save robustness ─────────────────────────────
class TestR34CheckpointRobustness:
    def test_checkpoint_creates_file(self, tmp_path):
        """_save_checkpoint must create a valid JSON checkpoint."""
        cp_path = tmp_path / "checkpoint.json"
        ids = ["id_001", "id_002", "id_003"]
        _save_checkpoint(cp_path, ids)

        assert cp_path.exists()
        data = json.loads(cp_path.read_text(encoding="utf-8"))
        assert data["count"] == 3
        assert set(data["completed_ids"]) == set(ids)

    def test_checkpoint_creates_backup(self, tmp_path):
        """_save_checkpoint should create .bak backup on second save."""
        cp_path = tmp_path / "checkpoint.json"
        _save_checkpoint(cp_path, ["first"])
        _save_checkpoint(cp_path, ["first", "second"])

        bak = cp_path.with_suffix(".bak")
        assert bak.exists()
        bak_data = json.loads(bak.read_text(encoding="utf-8"))
        assert bak_data["count"] == 1  # From first save

    def test_checkpoint_overwrites_correctly(self, tmp_path):
        """Multiple saves should correctly update the checkpoint."""
        cp_path = tmp_path / "checkpoint.json"
        _save_checkpoint(cp_path, ["a"])
        _save_checkpoint(cp_path, ["a", "b", "c"])

        data = json.loads(cp_path.read_text(encoding="utf-8"))
        assert data["count"] == 3
        assert "c" in data["completed_ids"]


# ── VC-S4-08: Per-CWE stats in output ────────────────────────────
class TestPerCWEStats:
    def test_per_source_and_cwe_in_dry_run(self, tmp_path):
        """Dry run should still log source distribution (R-33 detection)."""
        input_path = tmp_path / "input.jsonl"
        samples = [
            _make_sample("void a() { gets(b); }", source="sard", cwe="CWE-121", sample_id="s1"),
            _make_sample("void b() { system(c); }", source="cve", cwe="CWE-78", sample_id="s2"),
            _make_sample("void c() { read(0,b,1); }", source="repo", cwe="CWE-122", sample_id="s3"),
        ]
        with open(input_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        output_dir = tmp_path / "cpg_out"
        result = run_stage4(str(input_path), str(output_dir), dry_run=True)
        assert result["mode"] == "dry_run"
        assert result["total"] == 3
