#!/usr/bin/env python3
"""
tests/test_story_p3s2.py

Story P3-S2 — Stage 2: Deduplicate

10 tests covering all 4 dedup levels + pair repair + full pipeline.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from training.scripts.preprocessing.stage2_dedup import (
    level1_exact,
    level2_cve_id,
    level3_commit_sha,
    level4_minhash_lsh,
    md5_normalize,
    repair_broken_pairs,
    run_stage2,
)


# ── Helpers ───────────────────────────────────────────────────────
def _make_sample(
    code: str,
    label: int = 1,
    pair_id: str = "",
    cve_id: str = "",
    commit_sha: str = "",
    file_path: str = "",
    source: str = "test",
    sample_id: str = "",
) -> dict:
    """Build a minimal sample dict for testing."""
    return {
        "id": sample_id or f"test-{id(code)}",
        "source": source,
        "code": code,
        "label": label,
        "cwe": "CWE-121",
        "language": "c",
        "collected_at": "2026-03-22T00:00:00",
        "pair_id": pair_id,
        "file_path": file_path,
        "function_name": "test_func",
        "cfa_origin": "",
        "severity_score": 0.0,
        "commit_sha": commit_sha,
        "cve_id": cve_id,
        "aliases": [],
        "metadata": {},
    }


CODE_A = """\
void bad_func(char *input) {
    char buf[64];
    strcpy(buf, input);
    printf("%s\\n", buf);
    return;
}
"""

CODE_A_WHITESPACE_DUP = """\
void bad_func(char  *input) {
    char   buf[64];
    strcpy(buf,   input);
    printf("%s\\n",  buf);
    return;
}
"""

CODE_B = """\
void good_func(char *input) {
    char buf[64];
    strncpy(buf, input, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\\0';
    printf("%s\\n", buf);
    return;
}
"""

CODE_C = """\
int overflow_check(int a, int b) {
    if (a > INT_MAX - b) {
        return -1;
    }
    return a + b;
}
"""

# Near-duplicate of CODE_A (>85% Jaccard): one identifier changed
CODE_A_NEAR_DUP = """\
void bad_func(char *data) {
    char buf[64];
    strcpy(buf, data);
    printf("%s\\n", buf);
    return;
}
"""


# ── Test 1: L1 removes exact whitespace-normalized duplicate ─────
class TestLevel1Exact:
    def test_removes_whitespace_duplicate(self):
        s1 = _make_sample(CODE_A, sample_id="s1")
        s2 = _make_sample(CODE_A_WHITESPACE_DUP, sample_id="s2")

        # Verify they have the same MD5 after normalization
        assert md5_normalize(CODE_A) == md5_normalize(CODE_A_WHITESPACE_DUP)

        result = level1_exact([s1, s2])
        assert len(result) == 1

    # ── Test 2: L1 keeps samples with different code ─────────────
    def test_keeps_different_code(self):
        s1 = _make_sample(CODE_A, sample_id="s1")
        s2 = _make_sample(CODE_B, sample_id="s2")

        result = level1_exact([s1, s2])
        assert len(result) == 2


# ── Test 3: L2 removes same (cve_id, file_path) pair ────────────
class TestLevel2CveId:
    def test_removes_same_cve_file(self):
        s1 = _make_sample(
            CODE_A, cve_id="CVE-2024-1234", file_path="vuln.c", sample_id="s1"
        )
        s2 = _make_sample(
            CODE_B, cve_id="CVE-2024-1234", file_path="vuln.c", sample_id="s2"
        )

        result = level2_cve_id([s1, s2])
        assert len(result) == 1

    # ── Test 4: L2 keeps samples with different cve_id ───────────
    def test_keeps_different_cve(self):
        s1 = _make_sample(
            CODE_A, cve_id="CVE-2024-1234", file_path="vuln.c", sample_id="s1"
        )
        s2 = _make_sample(
            CODE_B, cve_id="CVE-2024-5678", file_path="vuln.c", sample_id="s2"
        )

        result = level2_cve_id([s1, s2])
        assert len(result) == 2


# ── Test 5: L3 removes same (commit_sha, file_path) pair ────────
class TestLevel3CommitSha:
    def test_removes_same_sha_file(self):
        s1 = _make_sample(
            CODE_A, commit_sha="abc123", file_path="src/main.c", sample_id="s1"
        )
        s2 = _make_sample(
            CODE_B, commit_sha="abc123", file_path="src/main.c", sample_id="s2"
        )

        result = level3_commit_sha([s1, s2])
        assert len(result) == 1


# ── Test 6: L4 removes near-duplicate (>85% Jaccard) ────────────
class TestLevel4MinHashLSH:
    def test_removes_near_duplicate(self):
        s1 = _make_sample(CODE_A, sample_id="s1")
        s2 = _make_sample(CODE_A_NEAR_DUP, sample_id="s2")

        result = level4_minhash_lsh([s1, s2], threshold=0.85)
        assert len(result) == 1

    # ── Test 7: L4 keeps sufficiently different samples ──────────
    def test_keeps_different_samples(self):
        s1 = _make_sample(CODE_A, sample_id="s1")
        s2 = _make_sample(CODE_C, sample_id="s2")

        result = level4_minhash_lsh([s1, s2], threshold=0.85)
        assert len(result) == 2


# ── Test 8: repair_broken_pairs clears pair_id on orphan ─────────
class TestRepairBrokenPairs:
    def test_clears_orphaned_pair_id(self):
        # Only label=1 survives for pair "P1" (label=0 was removed by L4)
        s1 = _make_sample(CODE_A, label=1, pair_id="P1", sample_id="s1")

        result = repair_broken_pairs([s1])
        assert result[0]["pair_id"] == ""

    # ── Test 9: repair_broken_pairs leaves valid pairs intact ────
    def test_leaves_valid_pair_intact(self):
        s1 = _make_sample(CODE_A, label=1, pair_id="P1", sample_id="s1")
        s2 = _make_sample(CODE_B, label=0, pair_id="P1", sample_id="s2")

        result = repair_broken_pairs([s1, s2])
        assert result[0]["pair_id"] == "P1"
        assert result[1]["pair_id"] == "P1"


# ── Test 10: Full pipeline — 10 samples with 3 dups → correct count
class TestFullPipeline:
    def test_full_pipeline_dedup(self, tmp_path):
        # Build 10 samples:
        # - 2 exact dups (CODE_A + whitespace variant) → L1 removes 1
        # - 2 same CVE/file → L2 removes 1
        # - 2 same commit/file → L3 removes 1
        # - 4 unique samples that survive all levels
        samples = [
            # Exact dup pair (L1)
            _make_sample(CODE_A, sample_id="exact1"),
            _make_sample(CODE_A_WHITESPACE_DUP, sample_id="exact2"),
            # CVE dup pair (L2) — different code, same CVE+file
            _make_sample(
                "void f1() {\n    int x = 1;\n    int y = 2;\n    int z = x + y;\n    return;\n}\n",
                cve_id="CVE-2024-9999",
                file_path="dup.c",
                sample_id="cve1",
            ),
            _make_sample(
                "void f2() {\n    int a = 3;\n    int b = 4;\n    int c = a + b;\n    return;\n}\n",
                cve_id="CVE-2024-9999",
                file_path="dup.c",
                sample_id="cve2",
            ),
            # Commit SHA dup pair (L3) — different code, same SHA+file
            _make_sample(
                "void f3() {\n    char *p = malloc(100);\n    free(p);\n    p = NULL;\n    return;\n}\n",
                commit_sha="deadbeef1234",
                file_path="sha.c",
                sample_id="sha1",
            ),
            _make_sample(
                "void f4() {\n    char *q = malloc(200);\n    free(q);\n    q = NULL;\n    return;\n}\n",
                commit_sha="deadbeef1234",
                file_path="sha.c",
                sample_id="sha2",
            ),
            # 4 unique survivors
            _make_sample(CODE_B, sample_id="unique1"),
            _make_sample(CODE_C, sample_id="unique2"),
            _make_sample(
                "void helper() {\n    FILE *f = fopen(\"test.txt\", \"r\");\n    if (f) fclose(f);\n    return;\n}\n",
                sample_id="unique3",
            ),
            _make_sample(
                "int validate(int x) {\n    if (x < 0) return -1;\n    if (x > 1000) return -1;\n    return x;\n}\n",
                sample_id="unique4",
            ),
        ]

        # Write input
        input_path = tmp_path / "input.jsonl"
        with open(input_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        output_path = tmp_path / "output.jsonl"

        stats = run_stage2(
            str(input_path), str(output_path), dry_run=False, threshold=0.85
        )

        assert stats["input"] == 10
        # L1 removes 1 (whitespace dup), L2 removes 1 (CVE dup), L3 removes 1 (SHA dup)
        assert stats["after_l1_exact"] == 9
        assert stats["after_l2_cve"] == 8
        assert stats["after_l3_sha"] == 7
        # L4 may or may not remove more (the 7 remaining are all distinct enough)
        assert stats["output"] >= 5  # at least the 4 unique + some survivors
        assert stats["output"] <= 7  # at most all 7 survive L4

        # Verify output file exists and is valid JSONL
        assert output_path.exists()
        with open(output_path) as f:
            output_samples = [json.loads(line) for line in f if line.strip()]
        assert len(output_samples) == stats["output"]


# ── Test 11 (bonus): dry-run does not write output ───────────────
class TestDryRun:
    def test_dry_run_no_output(self, tmp_path):
        samples = [_make_sample(CODE_A, sample_id="s1")]

        input_path = tmp_path / "input.jsonl"
        with open(input_path, "w") as f:
            f.write(json.dumps(samples[0]) + "\n")

        output_path = tmp_path / "output.jsonl"

        stats = run_stage2(
            str(input_path), str(output_path), dry_run=True, threshold=0.85
        )

        assert stats["input"] == 1
        assert not output_path.exists()


# ── Test 12 (bonus): atomic write leaves no .tmp on success ──────
class TestAtomicWrite:
    def test_no_tmp_file_after_success(self, tmp_path):
        samples = [_make_sample(CODE_A, sample_id="s1")]

        input_path = tmp_path / "input.jsonl"
        with open(input_path, "w") as f:
            f.write(json.dumps(samples[0]) + "\n")

        output_path = tmp_path / "output.jsonl"
        run_stage2(str(input_path), str(output_path), dry_run=False)

        # No .tmp files should remain
        tmp_files = list(tmp_path.glob("stage2_*.tmp"))
        assert len(tmp_files) == 0
        assert output_path.exists()
