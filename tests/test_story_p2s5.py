#!/usr/bin/env python3
"""
tests/test_story_p2s5.py

Tests for P2-S5: OSV Collector (Linux + OSS-Fuzz ecosystems).
Covers:
  1. Initialization (dirs, tokens, CVE dedup)
  2. GCS ZIP download and parsing
  3. Vulnerability parsing (commit SHAs, CWE, severity, aliases)
  4. Cross-dedup against CVE data
  5. Diff processing (paired samples, .c/.h filter, line bounds, 404)
  6. Schema compliance (validate_sample, language, source, pair_id)
  7. Checkpoint/resume
  8. Integration (dry-run, end-to-end)
"""
from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "training" / "scripts" / "collection")
)

from training.scripts.collection.osv_collector import (
    OSVCollector,
    ECOSYSTEMS,
    OSV_GCS_BASE,
    CHECKPOINT_INTERVAL,
)


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_osv_vuln(
    vuln_id: str = "OSV-2025-1234",
    repo_url: str = "https://github.com/torvalds/linux",
    fixed_sha: str = "a" * 40,
    cwe_ids: list[str] | None = None,
    summary: str = "Buffer overflow in kernel driver",
    details: str = "",
    severity_score: str | None = None,
    aliases: list[str] | None = None,
    range_type: str = "GIT",
) -> dict:
    """Build a minimal OSV vulnerability dict."""
    ranges = []
    if fixed_sha:
        ranges.append({
            "type": range_type,
            "repo": repo_url,
            "events": [
                {"introduced": "0"},
                {"fixed": fixed_sha},
            ],
        })

    db_specific = {}
    if cwe_ids is not None:
        db_specific["cwe_ids"] = cwe_ids

    severity = []
    if severity_score is not None:
        severity.append({"type": "CVSS_V3", "score": severity_score})

    return {
        "id": vuln_id,
        "summary": summary,
        "details": details,
        "aliases": aliases or [],
        "affected": [
            {
                "package": {"name": "linux", "ecosystem": "Linux"},
                "ranges": ranges,
            }
        ],
        "severity": severity,
        "database_specific": db_specific,
    }


def _make_zip_bytes(vulns: list[dict]) -> bytes:
    """Create a ZIP file in memory containing JSON files for each vuln."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for vuln in vulns:
            vuln_id = vuln.get("id", "unknown")
            zf.writestr(f"{vuln_id}.json", json.dumps(vuln))
    return buf.getvalue()


def _make_commit_response(
    sha: str = "a" * 40,
    filename: str = "drivers/net/bug.c",
    status: str = "modified",
    patch: str = None,
) -> dict:
    """Build a mock GitHub commit API response."""
    if patch is None:
        # Ensure >= 5 changed lines (MIN_CHANGED_LINES) for is_qualifying_file
        patch = (
            "@@ -1,15 +1,15 @@\n"
            " #include <stdio.h>\n"
            " #include <string.h>\n"
            " \n"
            " void process(char *input) {\n"
            "     char buf[64];\n"
            "-    strcpy(buf, input);\n"
            "-    printf(buf);\n"
            "-    strcat(buf, input);\n"
            "+    strncpy(buf, input, sizeof(buf)-1);\n"
            "+    buf[sizeof(buf)-1] = '\\0';\n"
            "+    printf(\"%s\", buf);\n"
            "+    strncat(buf, input, sizeof(buf)-strlen(buf)-1);\n"
            " }\n"
            " \n"
            " int main(int argc, char **argv) {\n"
            "-    process(argv[1]);\n"
            "+    if (argc > 1) process(argv[1]);\n"
            "     return 0;\n"
            " }\n"
        )
    return {
        "sha": sha,
        "files": [
            {
                "filename": filename,
                "status": status,
                "patch": patch,
            }
        ],
    }


# ══════════════════════════════════════════════════════════════════════════
# 1. Initialization
# ══════════════════════════════════════════════════════════════════════════

class TestInitialization:
    def test_init_creates_dirs(self):
        """Init should create output and checkpoint directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "osv_out"
            collector = OSVCollector(
                output_dir=str(out),
                github_tokens=["test_token"],
            )
            assert out.exists()
            assert (out / "checkpoints").exists()

    def test_token_resolution_from_env(self):
        """Tokens should be resolved from GITHUB_TOKEN env var."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"GITHUB_TOKEN": "tok1,tok2"}, clear=False):
                collector = OSVCollector(
                    output_dir=tmpdir,
                    github_tokens=None,
                )
                assert len(collector.rotator.tokens) == 2

    def test_no_token_raises(self):
        """Missing token should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(ValueError, match="No GitHub token"):
                    OSVCollector(output_dir=tmpdir)

    def test_cve_dedup_set_loading(self):
        """Should load commit_sha values from CVE samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            osv_dir = tmpdir / "osv"
            cve_dir = tmpdir / "cve"
            osv_dir.mkdir()
            cve_dir.mkdir()

            # Write fake CVE samples
            cve_path = cve_dir / "cve_samples.jsonl"
            samples = [
                {"commit_sha": "abc123", "code": "x", "label": 1},
                {"commit_sha": "def456", "code": "y", "label": 0},
                {"commit_sha": "", "code": "z", "label": 1},  # empty sha
            ]
            with open(cve_path, "w") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")

            collector = OSVCollector(
                output_dir=str(osv_dir),
                github_tokens=["test_token"],
            )
            dedup_set = collector._load_cve_dedup_set()
            assert "abc123" in dedup_set
            assert "def456" in dedup_set
            assert "" not in dedup_set
            assert len(dedup_set) == 2

    def test_cve_dedup_handles_missing_file(self):
        """Should return empty set when CVE data doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = OSVCollector(
                output_dir=tmpdir,
                github_tokens=["test_token"],
            )
            dedup_set = collector._load_cve_dedup_set()
            assert len(dedup_set) == 0


# ══════════════════════════════════════════════════════════════════════════
# 2. GCS Download
# ══════════════════════════════════════════════════════════════════════════

class TestGCSDownload:
    def test_zip_parsing_extracts_vulns(self):
        """Should parse all JSON files from ZIP correctly."""
        vulns = [
            _make_osv_vuln(vuln_id="OSV-2025-001"),
            _make_osv_vuln(vuln_id="OSV-2025-002"),
        ]
        zip_bytes = _make_zip_bytes(vulns)

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = OSVCollector(
                output_dir=tmpdir,
                github_tokens=["test_token"],
            )

            mock_resp = MagicMock()
            mock_resp.content = zip_bytes
            mock_resp.raise_for_status = MagicMock()

            with patch("requests.get", return_value=mock_resp):
                result = collector._download_ecosystem_zip("Linux")

            assert len(result) == 2
            ids = {v["id"] for v in result}
            assert "OSV-2025-001" in ids
            assert "OSV-2025-002" in ids

    def test_corrupt_zip_returns_empty(self):
        """Should return empty list on corrupt ZIP."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = OSVCollector(
                output_dir=tmpdir,
                github_tokens=["test_token"],
            )

            mock_resp = MagicMock()
            mock_resp.content = b"not a zip file"
            mock_resp.raise_for_status = MagicMock()

            with patch("requests.get", return_value=mock_resp):
                result = collector._download_ecosystem_zip("Linux")

            assert result == []

    def test_404_returns_empty(self):
        """Should return empty list on 404."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = OSVCollector(
                output_dir=tmpdir,
                github_tokens=["test_token"],
            )

            import requests as req
            mock_resp = MagicMock()
            mock_resp.status_code = 404
            http_error = req.exceptions.HTTPError(response=mock_resp)
            mock_resp.raise_for_status.side_effect = http_error

            with patch("requests.get", return_value=mock_resp):
                result = collector._download_ecosystem_zip("Linux")

            assert result == []


# ══════════════════════════════════════════════════════════════════════════
# 3. Vulnerability Parsing
# ══════════════════════════════════════════════════════════════════════════

class TestVulnerabilityParsing:
    def test_commit_sha_extraction_git_type(self):
        """Should extract (owner, repo, sha) from GIT range with fixed event."""
        vuln = _make_osv_vuln(
            repo_url="https://github.com/torvalds/linux",
            fixed_sha="abc123def456" + "0" * 28,
        )
        results = OSVCollector._extract_commit_shas(vuln)
        assert len(results) == 1
        owner, repo, sha = results[0]
        assert owner == "torvalds"
        assert repo == "linux"
        assert sha == "abc123def456" + "0" * 28

    def test_skips_non_git_range_types(self):
        """Should skip ECOSYSTEM and SEMVER range types."""
        vuln = _make_osv_vuln(range_type="ECOSYSTEM", fixed_sha="abc123")
        results = OSVCollector._extract_commit_shas(vuln)
        assert len(results) == 0

    def test_skips_semver_range(self):
        """Should skip SEMVER range type."""
        vuln = _make_osv_vuln(range_type="SEMVER", fixed_sha="abc123")
        results = OSVCollector._extract_commit_shas(vuln)
        assert len(results) == 0

    def test_cwe_from_database_specific(self):
        """Should extract CWE from database_specific.cwe_ids with tier1 confidence."""
        vuln = _make_osv_vuln(cwe_ids=["CWE-416"])
        cwe, confidence = OSVCollector._extract_cwe(vuln)
        assert cwe == "CWE-416"
        assert confidence == "tier1_direct"

    def test_cwe_from_database_specific_invalid_skipped(self):
        """Non-target CWE should fall through to keyword inference (tier3)."""
        vuln = _make_osv_vuln(
            cwe_ids=["CWE-999"],
            summary="integer overflow in parser",
        )
        cwe, confidence = OSVCollector._extract_cwe(vuln)
        assert cwe == "CWE-190"
        assert confidence == "tier3_inferred"

    def test_cwe_keyword_inference(self):
        """Should infer CWE from summary keywords when no db_specific."""
        vuln = _make_osv_vuln(
            cwe_ids=None,
            summary="use after free in memory allocator",
        )
        # Remove database_specific entirely
        vuln.pop("database_specific", None)
        cwe, confidence = OSVCollector._extract_cwe(vuln)
        assert cwe == "CWE-416"
        assert confidence == "tier3_inferred"

    def test_cwe_no_match_returns_none(self):
        """Should return None when no CWE match found."""
        vuln = _make_osv_vuln(
            cwe_ids=None,
            summary="unrelated issue",
            details="nothing relevant",
        )
        vuln.pop("database_specific", None)
        cwe, confidence = OSVCollector._extract_cwe(vuln)
        assert cwe is None
        assert confidence == ""

    def test_cvss_score_extraction_numeric(self):
        """Should extract numeric CVSS score."""
        vuln = _make_osv_vuln()
        vuln["severity"] = [{"type": "CVSS_V3", "score": 7.5}]
        result = OSVCollector._extract_severity(vuln)
        assert result == 7.5

    def test_cvss_score_vector_string(self):
        """CVSS vector string returns 0.0 (OSV doesn't store numeric score separately)."""
        vuln = _make_osv_vuln()
        vuln["severity"] = [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"}]
        result = OSVCollector._extract_severity(vuln)
        assert result == 0.0

    def test_cvss_no_severity_returns_zero(self):
        """No severity field should return 0.0."""
        vuln = _make_osv_vuln()
        vuln["severity"] = []
        result = OSVCollector._extract_severity(vuln)
        assert result == 0.0

    def test_alias_extraction(self):
        """Should extract CVE and GHSA aliases."""
        vuln = _make_osv_vuln(aliases=["CVE-2025-1234", "GHSA-xxxx-yyyy-zzzz"])
        result = OSVCollector._extract_aliases(vuln)
        assert result["cve"] == "CVE-2025-1234"
        assert result["ghsa"] == "GHSA-xxxx-yyyy-zzzz"

    def test_alias_empty(self):
        """No aliases should return empty dict."""
        vuln = _make_osv_vuln(aliases=[])
        result = OSVCollector._extract_aliases(vuln)
        assert result == {}

    def test_repo_url_with_dot_git_suffix(self):
        """Should handle repo URLs ending in .git."""
        vuln = _make_osv_vuln(
            repo_url="https://github.com/curl/curl.git",
            fixed_sha="b" * 40,
        )
        results = OSVCollector._extract_commit_shas(vuln)
        assert len(results) == 1
        assert results[0][0] == "curl"
        assert results[0][1] == "curl"


# ══════════════════════════════════════════════════════════════════════════
# 4. Cross-Dedup
# ══════════════════════════════════════════════════════════════════════════

class TestCrossDedup:
    def test_skips_vulns_with_cve_commit_sha(self):
        """Should skip vulns whose commit SHA is already in CVE data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            osv_dir = tmpdir / "osv"
            osv_dir.mkdir()

            collector = OSVCollector(
                output_dir=str(osv_dir),
                github_tokens=["test_token"],
            )

            sha = "a" * 40
            collector._cve_dedup_shas = {sha}

            vuln = _make_osv_vuln(fixed_sha=sha, cwe_ids=["CWE-120"])
            done_shas: set[str] = set()

            collector._process_vulnerability(vuln, "Linux", done_shas)

            assert collector.vulns_deduped_cve >= 1
            assert collector._samples_saved == 0

    def test_loads_dedup_set_from_cve_jsonl(self):
        """Should load dedup set from existing cve_samples.jsonl."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            osv_dir = tmpdir / "osv"
            cve_dir = tmpdir / "cve"
            osv_dir.mkdir()
            cve_dir.mkdir()

            sha1 = "a" * 40
            sha2 = "b" * 40
            cve_path = cve_dir / "cve_samples.jsonl"
            with open(cve_path, "w") as f:
                f.write(json.dumps({"commit_sha": sha1}) + "\n")
                f.write(json.dumps({"commit_sha": sha2}) + "\n")

            collector = OSVCollector(
                output_dir=str(osv_dir),
                github_tokens=["test_token"],
            )
            dedup_set = collector._load_cve_dedup_set()
            assert sha1 in dedup_set
            assert sha2 in dedup_set

    def test_handles_missing_cve_data(self):
        """Should handle missing CVE data gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = OSVCollector(
                output_dir=tmpdir,
                github_tokens=["test_token"],
            )
            dedup_set = collector._load_cve_dedup_set()
            assert len(dedup_set) == 0


# ══════════════════════════════════════════════════════════════════════════
# 5. Diff Processing
# ══════════════════════════════════════════════════════════════════════════

class TestDiffProcessing:
    def test_creates_paired_samples(self):
        """Should create vuln (label=1) + safe (label=0) pair from commit diff."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = OSVCollector(
                output_dir=tmpdir,
                github_tokens=["test_token"],
            )
            collector._cve_dedup_shas = set()

            sha = "a" * 40
            vuln = _make_osv_vuln(fixed_sha=sha, cwe_ids=["CWE-120"])
            commit_resp = _make_commit_response(sha=sha)

            done_shas: set[str] = set()

            with patch.object(collector, '_fetch_commit', return_value=commit_resp), \
                 patch.object(collector, 'jitter_sleep'):
                collector._process_vulnerability(vuln, "Linux", done_shas)

            assert collector._samples_saved == 2
            assert collector.pairs_saved == 1

            # Verify JSONL output
            out_path = Path(tmpdir) / "osv_samples.jsonl"
            assert out_path.exists()
            samples = []
            with open(out_path) as f:
                for line in f:
                    samples.append(json.loads(line.strip()))

            assert len(samples) == 2
            labels = {s["label"] for s in samples}
            assert labels == {0, 1}
            assert samples[0]["pair_id"] == samples[1]["pair_id"]

    def test_only_processes_c_h_files(self):
        """Should only process .c and .h files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = OSVCollector(
                output_dir=tmpdir,
                github_tokens=["test_token"],
            )
            collector._cve_dedup_shas = set()

            sha = "a" * 40
            vuln = _make_osv_vuln(fixed_sha=sha, cwe_ids=["CWE-120"])

            commit_resp = _make_commit_response(
                sha=sha,
                filename="src/main.py",  # Python file — should be skipped
            )

            done_shas: set[str] = set()

            with patch.object(collector, '_fetch_commit', return_value=commit_resp), \
                 patch.object(collector, 'jitter_sleep'):
                collector._process_vulnerability(vuln, "Linux", done_shas)

            assert collector._samples_saved == 0
            assert collector.files_qualifying == 0

    def test_respects_changed_line_bounds(self):
        """Files with too few changed lines should be rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = OSVCollector(
                output_dir=tmpdir,
                github_tokens=["test_token"],
            )
            collector._cve_dedup_shas = set()

            sha = "a" * 40
            vuln = _make_osv_vuln(fixed_sha=sha, cwe_ids=["CWE-120"])

            # Patch with only 2 changed lines (below MIN_CHANGED_LINES=5)
            tiny_patch = "@@ -1,3 +1,3 @@\n #include <stdio.h>\n-int x;\n+int y;\n"
            commit_resp = _make_commit_response(sha=sha, patch=tiny_patch)

            done_shas: set[str] = set()

            with patch.object(collector, '_fetch_commit', return_value=commit_resp), \
                 patch.object(collector, 'jitter_sleep'):
                collector._process_vulnerability(vuln, "Linux", done_shas)

            assert collector._samples_saved == 0

    def test_handles_404_stale_commits(self):
        """Should handle 404 stale commits gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = OSVCollector(
                output_dir=tmpdir,
                github_tokens=["test_token"],
            )
            collector._cve_dedup_shas = set()

            sha = "a" * 40
            vuln = _make_osv_vuln(fixed_sha=sha, cwe_ids=["CWE-120"])

            done_shas: set[str] = set()

            with patch.object(collector, '_fetch_commit', return_value=None), \
                 patch.object(collector, 'jitter_sleep'):
                collector._process_vulnerability(vuln, "Linux", done_shas)

            assert collector.commits_stale_404 == 1
            assert collector._samples_saved == 0
            assert sha in done_shas


# ══════════════════════════════════════════════════════════════════════════
# 6. Schema Compliance
# ══════════════════════════════════════════════════════════════════════════

class TestSchemaCompliance:
    def test_all_saved_samples_pass_validation(self):
        """Every saved sample must pass validate_sample()."""
        from training.scripts.collection.schema import validate_sample

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = OSVCollector(
                output_dir=tmpdir,
                github_tokens=["test_token"],
            )
            collector._cve_dedup_shas = set()

            sha = "a" * 40
            vuln = _make_osv_vuln(fixed_sha=sha, cwe_ids=["CWE-120"])
            commit_resp = _make_commit_response(sha=sha)
            done_shas: set[str] = set()

            with patch.object(collector, '_fetch_commit', return_value=commit_resp), \
                 patch.object(collector, 'jitter_sleep'):
                collector._process_vulnerability(vuln, "Linux", done_shas)

            out_path = Path(tmpdir) / "osv_samples.jsonl"
            with open(out_path) as f:
                for line in f:
                    sample = json.loads(line.strip())
                    is_valid, errors = validate_sample(sample)
                    assert is_valid, f"Sample failed validation: {errors}"

    def test_language_always_c(self):
        """All samples must have language='c'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = OSVCollector(
                output_dir=tmpdir,
                github_tokens=["test_token"],
            )
            collector._cve_dedup_shas = set()

            sha = "a" * 40
            vuln = _make_osv_vuln(fixed_sha=sha, cwe_ids=["CWE-120"])
            commit_resp = _make_commit_response(sha=sha)
            done_shas: set[str] = set()

            with patch.object(collector, '_fetch_commit', return_value=commit_resp), \
                 patch.object(collector, 'jitter_sleep'):
                collector._process_vulnerability(vuln, "Linux", done_shas)

            out_path = Path(tmpdir) / "osv_samples.jsonl"
            with open(out_path) as f:
                for line in f:
                    sample = json.loads(line.strip())
                    assert sample["language"] == "c"

    def test_source_always_osv(self):
        """All samples must have source='osv'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = OSVCollector(
                output_dir=tmpdir,
                github_tokens=["test_token"],
            )
            collector._cve_dedup_shas = set()

            sha = "a" * 40
            vuln = _make_osv_vuln(fixed_sha=sha, cwe_ids=["CWE-120"])
            commit_resp = _make_commit_response(sha=sha)
            done_shas: set[str] = set()

            with patch.object(collector, '_fetch_commit', return_value=commit_resp), \
                 patch.object(collector, 'jitter_sleep'):
                collector._process_vulnerability(vuln, "Linux", done_shas)

            out_path = Path(tmpdir) / "osv_samples.jsonl"
            with open(out_path) as f:
                for line in f:
                    sample = json.loads(line.strip())
                    assert sample["source"] == "osv"

    def test_pair_id_links_vuln_and_safe(self):
        """Paired samples should share the same pair_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = OSVCollector(
                output_dir=tmpdir,
                github_tokens=["test_token"],
            )
            collector._cve_dedup_shas = set()

            sha = "a" * 40
            vuln = _make_osv_vuln(fixed_sha=sha, cwe_ids=["CWE-120"])
            commit_resp = _make_commit_response(sha=sha)
            done_shas: set[str] = set()

            with patch.object(collector, '_fetch_commit', return_value=commit_resp), \
                 patch.object(collector, 'jitter_sleep'):
                collector._process_vulnerability(vuln, "Linux", done_shas)

            out_path = Path(tmpdir) / "osv_samples.jsonl"
            samples = []
            with open(out_path) as f:
                for line in f:
                    samples.append(json.loads(line.strip()))

            assert len(samples) == 2
            pair_ids = {s["pair_id"] for s in samples}
            assert len(pair_ids) == 1  # Both share same pair_id
            assert "" not in pair_ids  # pair_id is not empty

    def test_cwe_confidence_stored_in_metadata(self):
        """R-04: cwe_confidence must be stored in metadata for label smoothing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = OSVCollector(
                output_dir=tmpdir,
                github_tokens=["test_token"],
            )
            collector._cve_dedup_shas = set()

            sha = "a" * 40
            # db_specific CWE → tier1_direct
            vuln = _make_osv_vuln(fixed_sha=sha, cwe_ids=["CWE-120"])
            commit_resp = _make_commit_response(sha=sha)
            done_shas: set[str] = set()

            with patch.object(collector, '_fetch_commit', return_value=commit_resp), \
                 patch.object(collector, 'jitter_sleep'):
                collector._process_vulnerability(vuln, "Linux", done_shas)

            out_path = Path(tmpdir) / "osv_samples.jsonl"
            with open(out_path) as f:
                for line in f:
                    sample = json.loads(line.strip())
                    assert "cwe_confidence" in sample.get("metadata", {}), \
                        "cwe_confidence missing from metadata"
                    assert sample["metadata"]["cwe_confidence"] == "tier1_direct"

    def test_cwe_confidence_tier3_for_keyword_inference(self):
        """R-04: keyword-inferred CWE must be tagged tier3_inferred."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = OSVCollector(
                output_dir=tmpdir,
                github_tokens=["test_token"],
            )
            collector._cve_dedup_shas = set()

            sha = "c" * 40
            # No db_specific CWE, but summary triggers keyword match
            vuln = _make_osv_vuln(
                fixed_sha=sha,
                cwe_ids=None,
                summary="uses strcpy( without bounds checking",
            )
            vuln.pop("database_specific", None)

            # Distinct patch to avoid MD5 dedup collision with other tests
            tier3_patch = (
                "@@ -1,15 +1,15 @@\n"
                " #include <stdio.h>\n"
                " #include <stdlib.h>\n"
                " \n"
                " void tier3_handler(char *data) {\n"
                "     char dest[128];\n"
                "-    strcpy(dest, data);\n"
                "-    printf(dest);\n"
                "-    strcat(dest, data);\n"
                "+    strncpy(dest, data, sizeof(dest)-1);\n"
                "+    dest[sizeof(dest)-1] = '\\0';\n"
                "+    printf(\"%s\", dest);\n"
                "+    strncat(dest, data, sizeof(dest)-strlen(dest)-1);\n"
                " }\n"
                " \n"
                " int main(int argc, char **argv) {\n"
                "-    tier3_handler(argv[1]);\n"
                "+    if (argc > 1) tier3_handler(argv[1]);\n"
                "     return 0;\n"
                " }\n"
            )
            commit_resp = _make_commit_response(sha=sha, patch=tier3_patch)
            done_shas: set[str] = set()

            with patch.object(collector, '_fetch_commit', return_value=commit_resp), \
                 patch.object(collector, 'jitter_sleep'):
                collector._process_vulnerability(vuln, "Linux", done_shas)

            out_path = Path(tmpdir) / "osv_samples.jsonl"
            assert out_path.exists(), "No samples saved — check CWE inference and patch size"
            with open(out_path) as f:
                for line in f:
                    sample = json.loads(line.strip())
                    assert sample["metadata"]["cwe_confidence"] == "tier3_inferred"


# ══════════════════════════════════════════════════════════════════════════
# 7. Checkpoint/Resume
# ══════════════════════════════════════════════════════════════════════════

class TestCheckpointResume:
    def test_checkpoint_saves_after_processing(self):
        """Checkpoint should be saved during collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = OSVCollector(
                output_dir=tmpdir,
                github_tokens=["test_token"],
                max_samples=2,
            )

            vulns = [_make_osv_vuln(vuln_id="OSV-001", cwe_ids=["CWE-120"])]
            commit_resp = _make_commit_response()

            def mock_download(eco):
                return vulns if eco == "Linux" else []

            with patch.object(collector, '_download_ecosystem_zip', side_effect=mock_download), \
                 patch.object(collector, '_fetch_commit', return_value=commit_resp), \
                 patch.object(collector, '_load_cve_dedup_set', return_value=set()), \
                 patch.object(collector, 'jitter_sleep'):
                collector.collect()

            # Checkpoint file should exist
            cp_path = Path(tmpdir) / "checkpoints" / "osv_checkpoint.json"
            assert cp_path.exists()
            cp = json.loads(cp_path.read_text())
            assert "done_shas" in cp
            assert "completed" in cp

    def test_resume_skips_processed_vulns(self):
        """Resume should skip already-processed vulns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            cp_dir = tmpdir / "checkpoints"
            cp_dir.mkdir()

            # Write checkpoint with already-processed vuln
            cp_path = cp_dir / "osv_checkpoint.json"
            cp_path.write_text(json.dumps({
                "done_shas": [],
                "processed_vulns": ["OSV-already-done"],
                "completed_ecosystems": [],
                "samples_saved": 0,
            }), encoding="utf-8")

            collector = OSVCollector(
                output_dir=str(tmpdir),
                checkpoint_dir=str(cp_dir),
                github_tokens=["test_token"],
                max_samples=10,
            )

            # Create two vulns: one already processed, one new
            vulns = [
                _make_osv_vuln(vuln_id="OSV-already-done", cwe_ids=["CWE-120"]),
                _make_osv_vuln(vuln_id="OSV-brand-new", cwe_ids=["CWE-120"], fixed_sha="b" * 40),
            ]

            def mock_download(eco):
                return vulns if eco == "Linux" else []

            commit_resp = _make_commit_response(sha="b" * 40)

            with patch.object(collector, '_download_ecosystem_zip', side_effect=mock_download), \
                 patch.object(collector, '_fetch_commit', return_value=commit_resp), \
                 patch.object(collector, '_load_cve_dedup_set', return_value=set()), \
                 patch.object(collector, 'jitter_sleep'):
                collector.collect()

            # Only the new vuln should have been processed
            # The already-done vuln should have been skipped
            assert collector._samples_saved == 2  # One pair from new vuln

    def test_checkpoint_deleted_on_completion(self):
        """Checkpoint should indicate completion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = OSVCollector(
                output_dir=tmpdir,
                github_tokens=["test_token"],
                max_samples=2,
            )

            vulns = [_make_osv_vuln(vuln_id="OSV-001", cwe_ids=["CWE-120"])]
            commit_resp = _make_commit_response()

            def mock_download(eco):
                return vulns if eco == "Linux" else []

            with patch.object(collector, '_download_ecosystem_zip', side_effect=mock_download), \
                 patch.object(collector, '_fetch_commit', return_value=commit_resp), \
                 patch.object(collector, '_load_cve_dedup_set', return_value=set()), \
                 patch.object(collector, 'jitter_sleep'):
                collector.collect()

            cp_path = Path(tmpdir) / "checkpoints" / "osv_checkpoint.json"
            cp = json.loads(cp_path.read_text())
            assert cp.get("completed") is True


# ══════════════════════════════════════════════════════════════════════════
# 8. Integration
# ══════════════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_dry_run_processes_but_doesnt_save(self):
        """Dry-run mode should process but not write samples to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = OSVCollector(
                output_dir=tmpdir,
                github_tokens=["test_token"],
                dry_run=True,
                max_samples=10,
            )

            vulns = [_make_osv_vuln(vuln_id="OSV-001", cwe_ids=["CWE-120"])]
            commit_resp = _make_commit_response()

            def mock_download(eco):
                return vulns if eco == "Linux" else []

            with patch.object(collector, '_download_ecosystem_zip', side_effect=mock_download), \
                 patch.object(collector, '_fetch_commit', return_value=commit_resp), \
                 patch.object(collector, '_load_cve_dedup_set', return_value=set()), \
                 patch.object(collector, 'jitter_sleep'):
                summary = collector.collect()

            # Samples counted but not saved to disk
            assert summary["samples_saved"] == 2
            out_path = Path(tmpdir) / "osv_samples.jsonl"
            assert not out_path.exists()

    def test_end_to_end_with_mocked_responses(self):
        """End-to-end test with mocked GCS + GitHub responses."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = OSVCollector(
                output_dir=tmpdir,
                github_tokens=["test_token"],
                max_samples=10,
            )

            sha1 = "a" * 40
            sha2 = "b" * 40
            vulns_linux = [
                _make_osv_vuln(vuln_id="OSV-L1", fixed_sha=sha1, cwe_ids=["CWE-120"]),
            ]
            vulns_ossfuzz = [
                _make_osv_vuln(
                    vuln_id="OSV-F1",
                    fixed_sha=sha2,
                    cwe_ids=["CWE-416"],
                    repo_url="https://github.com/example/project",
                ),
            ]

            # Different patch for OSS-Fuzz to avoid MD5 dedup collision
            ossfuzz_patch = (
                "@@ -1,15 +1,15 @@\n"
                " #include <stdlib.h>\n"
                " #include <string.h>\n"
                " \n"
                " void handle(void *ptr) {\n"
                "-    free(ptr);\n"
                "-    memcpy(ptr, \"data\", 4);\n"
                "-    ptr = NULL;\n"
                "+    if (ptr != NULL) {\n"
                "+        free(ptr);\n"
                "+        ptr = NULL;\n"
                "+    }\n"
                " }\n"
                " \n"
                " int main(void) {\n"
                "-    void *p = malloc(64);\n"
                "+    void *p = calloc(1, 64);\n"
                "     handle(p);\n"
                "     return 0;\n"
                " }\n"
            )

            def mock_download(eco):
                if eco == "Linux":
                    return vulns_linux
                elif eco == "OSS-Fuzz":
                    return vulns_ossfuzz
                return []

            def mock_fetch_commit(owner, repo, sha):
                if sha == sha2:
                    return _make_commit_response(sha=sha, patch=ossfuzz_patch)
                return _make_commit_response(sha=sha)

            with patch.object(collector, '_download_ecosystem_zip', side_effect=mock_download), \
                 patch.object(collector, '_fetch_commit', side_effect=mock_fetch_commit), \
                 patch.object(collector, '_load_cve_dedup_set', return_value=set()), \
                 patch.object(collector, 'jitter_sleep'):
                summary = collector.collect()

            assert summary["samples_saved"] == 4  # 2 pairs
            assert summary["pairs_saved"] == 2

            # Verify output
            out_path = Path(tmpdir) / "osv_samples.jsonl"
            assert out_path.exists()
            samples = []
            with open(out_path) as f:
                for line in f:
                    samples.append(json.loads(line.strip()))

            assert len(samples) == 4
            assert all(s["source"] == "osv" for s in samples)
            assert all(s["language"] == "c" for s in samples)

            # Check CWE distribution
            cwes = {s["cwe"] for s in samples}
            assert "CWE-120" in cwes
            assert "CWE-416" in cwes

    def test_max_samples_stops_collection(self):
        """Collection should stop when max_samples is reached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = OSVCollector(
                output_dir=tmpdir,
                github_tokens=["test_token"],
                max_samples=2,  # Stop after 1 pair (2 samples)
            )

            sha1 = "a" * 40
            sha2 = "b" * 40
            vulns = [
                _make_osv_vuln(vuln_id="OSV-001", fixed_sha=sha1, cwe_ids=["CWE-120"]),
                _make_osv_vuln(vuln_id="OSV-002", fixed_sha=sha2, cwe_ids=["CWE-416"]),
            ]

            def mock_download(eco):
                return vulns if eco == "Linux" else []

            def mock_fetch_commit(owner, repo, sha):
                return _make_commit_response(sha=sha)

            with patch.object(collector, '_download_ecosystem_zip', side_effect=mock_download), \
                 patch.object(collector, '_fetch_commit', side_effect=mock_fetch_commit), \
                 patch.object(collector, '_load_cve_dedup_set', return_value=set()), \
                 patch.object(collector, 'jitter_sleep'):
                summary = collector.collect()

            assert summary["samples_saved"] == 2  # Stopped at max

    def test_ecosystems_match_spec(self):
        """ECOSYSTEMS constant should be Linux and OSS-Fuzz per spec."""
        assert ECOSYSTEMS == ["Linux", "OSS-Fuzz"]

    def test_summary_includes_all_fields(self):
        """Summary dict should include all expected fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = OSVCollector(
                output_dir=tmpdir,
                github_tokens=["test_token"],
                max_samples=2,
            )

            # Mock empty collection
            with patch.object(collector, '_download_ecosystem_zip', return_value=[]), \
                 patch.object(collector, '_load_cve_dedup_set', return_value=set()):
                summary = collector.collect()

            expected_keys = {
                "vulns_scanned", "vulns_no_git_range", "vulns_no_cwe",
                "vulns_deduped_cve", "commits_processed", "commits_stale_404",
                "files_qualifying", "samples_saved", "samples_failed",
                "pairs_saved", "label_distribution", "cwe_distribution",
                "ecosystem_distribution", "tokens_available",
            }
            assert expected_keys.issubset(set(summary.keys()))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
