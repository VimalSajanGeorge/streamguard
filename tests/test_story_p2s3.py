#!/usr/bin/env python3
"""
tests/test_story_p2s3.py

Tests for P2-S3: CVE/NVD Two-Phase Collector.
Covers:
  1.  NVD v2.0 URL assertion (not v1.0)
  2.  vulnerabilities[] parsing from mock NVD response
  3.  GitHub commit URL extraction from references list
  4.  SHA normalization: short SHA -> full SHA resolution
  5.  Diff filter: modified .c file with 10 changed lines -> kept
  6.  Diff filter: .py file -> rejected
  7.  Diff filter: 200 changed lines -> rejected
  8.  Before/after code extraction from unified diff
  9.  Checkpoint resume: already-done commit SHAs skipped
  10. 503 response -> sleeps and retries (mock)
  11. Multi-token rotator: rotation on exhaustion
  12. Phase 2 full pipeline with mock data
  13. Stale 404 commit -> logged to cve_stale.jsonl
  14. Index deduplication across CWEs
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "training" / "scripts" / "collection")
)

from training.scripts.collection.cve_collector_enhanced import (
    NVD_API_V2_URL,
    NVDIndexBuilder,
    GitHubDiffFetcher,
    GitHubTokenRotator,
)


# ── Test 1: NVD v2.0 URL assertion ──────────────────────────────────────

def test_nvd_v2_url():
    """NVD API URL must be v2.0 endpoint, not v1.0."""
    assert "2.0" in NVD_API_V2_URL
    assert "cves/2.0" in NVD_API_V2_URL
    assert "1.0" not in NVD_API_V2_URL


# ── Test 2: vulnerabilities[] parsing from mock NVD response ─────────────

def test_nvd_response_parsing():
    """NVDIndexBuilder should parse vulnerabilities from NVD API response structure."""
    mock_response = {
        "resultsPerPage": 2,
        "startIndex": 0,
        "totalResults": 2,
        "vulnerabilities": [
            {
                "cve": {
                    "id": "CVE-2024-0001",
                    "references": [
                        {"url": "https://github.com/owner/repo/commit/abc123def456"},
                    ],
                    "weaknesses": [
                        {
                            "description": [
                                {"value": "CWE-120"}
                            ]
                        }
                    ],
                    "published": "2024-01-15T10:00:00.000",
                }
            },
            {
                "cve": {
                    "id": "CVE-2024-0002",
                    "references": [
                        {"url": "https://example.com/advisory/123"},
                    ],
                    "weaknesses": [],
                    "published": "2024-01-16T10:00:00.000",
                }
            },
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        builder = NVDIndexBuilder(
            output_dir=tmpdir,
            nvd_api_key="test_key",
            max_samples=10,
        )

        # Mock _query_nvd to return our response once, then empty
        call_count = [0]
        def mock_query(cwe, start_index=0):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_response
            return {"totalResults": 0, "vulnerabilities": []}

        with patch.object(builder, '_query_nvd', side_effect=mock_query):
            # Process just one CWE
            cwe_progress = {}
            seen = set()
            builder._process_cwe("CWE-120", 0, seen, cwe_progress)

        # CVE-2024-0001 has GitHub URL, CVE-2024-0002 does not
        assert builder.cves_with_github == 1
        assert builder.cves_without_github == 1
        assert "CVE-2024-0001" in seen


# ── Test 3: GitHub commit URL extraction from references ─────────────────

def test_github_commit_url_extraction():
    """Only URLs with github.com AND /commit/ should be extracted."""
    refs = [
        {"url": "https://github.com/openssl/openssl/commit/abc123def"},
        {"url": "https://github.com/owner/repo/pull/456"},
        {"url": "https://github.com/owner/repo/issues/789"},
        {"url": "https://nvd.nist.gov/vuln/detail/CVE-2024-0001"},
    ]
    result = NVDIndexBuilder.extract_github_commit_url(refs)
    assert result == "https://github.com/openssl/openssl/commit/abc123def"


def test_github_commit_url_no_match():
    """No GitHub commit URL -> returns None."""
    refs = [
        {"url": "https://github.com/owner/repo/pull/456"},
        {"url": "https://nvd.nist.gov/vuln/detail/CVE-2024-0001"},
    ]
    result = NVDIndexBuilder.extract_github_commit_url(refs)
    assert result is None


def test_github_commit_url_excludes_pull_with_commit():
    """URLs with both /pull/ and /commit/ should be excluded."""
    refs = [
        {"url": "https://github.com/owner/repo/pull/123/commits/abc"},
    ]
    # This URL contains /pull/ so should be excluded
    result = NVDIndexBuilder.extract_github_commit_url(refs)
    assert result is None


# ── Test 4: SHA normalization ────────────────────────────────────────────

def test_sha_normalization_short():
    """Short SHA should be resolved to full 40-char SHA."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fetcher = GitHubDiffFetcher(
            output_dir=tmpdir,
            github_tokens=["test_token"],
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"sha": "a" * 40}
        mock_resp.headers = {"X-RateLimit-Remaining": "4999", "X-RateLimit-Reset": str(time.time() + 3600)}

        with patch.object(fetcher, 'safe_get', return_value=mock_resp):
            result = fetcher._resolve_full_sha("owner", "repo", "abc123")

        assert result == "a" * 40
        assert len(result) == 40


def test_sha_normalization_full_passthrough():
    """40-char SHA should pass through without API call."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fetcher = GitHubDiffFetcher(
            output_dir=tmpdir,
            github_tokens=["test_token"],
        )

        full_sha = "a" * 40
        result = fetcher._resolve_full_sha("owner", "repo", full_sha)
        assert result == full_sha


# ── Test 5: Diff filter — modified .c file with 10 changed lines -> kept ─

def test_diff_filter_valid_c_file():
    """Modified .c file with 10 changed lines should qualify."""
    # 10 changed lines (5 adds + 5 removes)
    patch_content = "\n".join(
        [f"-old_line_{i}" for i in range(5)] +
        [f"+new_line_{i}" for i in range(5)]
    )
    file_info = {
        "filename": "src/vuln.c",
        "status": "modified",
        "patch": patch_content,
    }
    qualifies, reason = GitHubDiffFetcher.is_qualifying_file(file_info)
    assert qualifies is True
    assert reason == ""


# ── Test 6: Diff filter — .py file -> rejected ──────────────────────────

def test_diff_filter_rejects_python():
    """.py files should be rejected."""
    file_info = {
        "filename": "script.py",
        "status": "modified",
        "patch": "+line1\n-line2\n" * 5,
    }
    qualifies, reason = GitHubDiffFetcher.is_qualifying_file(file_info)
    assert qualifies is False
    assert reason == "extension"


def test_diff_filter_accepts_h_file():
    """.h header files should qualify."""
    patch_content = "\n".join(
        [f"-old_{i}" for i in range(5)] + [f"+new_{i}" for i in range(5)]
    )
    file_info = {
        "filename": "include/header.h",
        "status": "modified",
        "patch": patch_content,
    }
    qualifies, reason = GitHubDiffFetcher.is_qualifying_file(file_info)
    assert qualifies is True


# ── Test 7: Diff filter — 200 changed lines -> rejected ─────────────────

def test_diff_filter_rejects_mega_commit():
    """Files with >150 changed lines should be rejected."""
    patch_content = "\n".join([f"+line_{i}" for i in range(200)])
    file_info = {
        "filename": "src/main.c",
        "status": "modified",
        "patch": patch_content,
    }
    qualifies, reason = GitHubDiffFetcher.is_qualifying_file(file_info)
    assert qualifies is False
    assert reason == "size"


def test_diff_filter_rejects_added_files():
    """Files with status='added' should be rejected (only 'modified' allowed)."""
    file_info = {
        "filename": "src/new.c",
        "status": "added",
        "patch": "+int x = 1;\n" * 10,
    }
    qualifies, reason = GitHubDiffFetcher.is_qualifying_file(file_info)
    assert qualifies is False
    assert reason == "status"


# ── Test 8: Before/after code extraction from unified diff ───────────────

def test_before_after_extraction():
    """Should correctly split unified diff into before and after code."""
    patch = """\
@@ -10,6 +10,8 @@ void func() {
     int x = 0;
-    gets(buf);
-    strcpy(dst, src);
+    fgets(buf, sizeof(buf), stdin);
+    strncpy(dst, src, sizeof(dst)-1);
+    dst[sizeof(dst)-1] = '\\0';
     return;
 }"""

    before, after = GitHubDiffFetcher.extract_before_after(patch)

    # Before should have the removed lines
    assert "gets(buf);" in before
    assert "strcpy(dst, src);" in before
    assert "fgets(" not in before

    # After should have the added lines
    assert "fgets(buf, sizeof(buf), stdin);" in after
    assert "strncpy(dst, src, sizeof(dst)-1);" in after
    assert "gets(buf);" not in after

    # Context lines should be in both
    assert "int x = 0;" in before
    assert "int x = 0;" in after
    assert "return;" in before
    assert "return;" in after


# ── Test 9: Checkpoint resume — done SHAs skipped ────────────────────────

def test_checkpoint_resume_skips_done_shas():
    """Already-processed commit SHAs should be skipped on resume."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"
        checkpoint_dir = tmpdir / "checkpoints"

        # Create a minimal index
        index_path = output_dir / "cve_index.jsonl"
        output_dir.mkdir(parents=True)
        entries = [
            {"cve_id": "CVE-2024-0001", "github_commit_url": "https://github.com/o/r/commit/" + "a" * 40, "cwe": "CWE-120"},
            {"cve_id": "CVE-2024-0002", "github_commit_url": "https://github.com/o/r/commit/" + "b" * 40, "cwe": "CWE-120"},
        ]
        with open(index_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        # Write checkpoint with first SHA already done
        checkpoint_dir.mkdir(parents=True)
        cp_path = checkpoint_dir / "cve_checkpoint.json"
        cp_path.write_text(json.dumps({
            "done_shas": ["a" * 40],
            "samples_saved": 2,
        }), encoding="utf-8")

        fetcher = GitHubDiffFetcher(
            output_dir=str(output_dir),
            checkpoint_dir=str(checkpoint_dir),
            github_tokens=["test_token"],
            max_samples=10,
        )

        # Track which SHAs are actually fetched
        fetched_shas = []

        def mock_fetch(owner, repo, sha):
            fetched_shas.append(sha)
            return {"files": []}  # No qualifying files

        with patch.object(fetcher, '_fetch_commit', side_effect=mock_fetch):
            fetcher.collect()

        # Only the second SHA should be fetched (first was in checkpoint)
        assert "b" * 40 in fetched_shas
        assert "a" * 40 not in fetched_shas


# ── Test 10: 503 response handling (via BaseCollector.safe_get) ──────────

def test_503_handling():
    """503 responses should trigger sleep + retry via BaseCollector.safe_get."""
    with tempfile.TemporaryDirectory() as tmpdir:
        builder = NVDIndexBuilder(
            output_dir=tmpdir,
            nvd_api_key="test_key",
        )

        call_count = [0]
        mock_responses = []

        # First call returns 503, retry returns 200
        resp_503 = MagicMock()
        resp_503.status_code = 503
        resp_503.text = "Service Unavailable"

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {"totalResults": 0, "vulnerabilities": []}

        # BaseCollector.safe_get handles 503 internally, so we mock requests.get
        def mock_get(url, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return resp_503
            return resp_200

        with patch('requests.get', side_effect=mock_get), \
             patch('time.sleep'):  # Don't actually sleep in tests
            result = builder.safe_get(NVD_API_V2_URL, params={"cweId": "CWE-120"})

        # Should have retried after 503
        assert call_count[0] == 2
        assert result.status_code == 200


# ── Test 11: Multi-token rotator ─────────────────────────────────────────

def test_token_rotator_basic():
    """Rotator should cycle through tokens."""
    rotator = GitHubTokenRotator(["tok_a", "tok_b", "tok_c"])
    assert rotator.current_token == "tok_a"

    # Simulate token A exhausted
    rotator._limits[0] = {"remaining": 10, "reset": time.time() + 3600}
    rotator.rotate_if_needed()

    # Should rotate to tok_b (which has default high remaining)
    assert rotator.current_token == "tok_b"


def test_token_rotator_auth_headers():
    """Auth headers should use current token."""
    rotator = GitHubTokenRotator(["ghp_test123"])
    headers = rotator.auth_headers
    assert headers["Authorization"] == "Bearer ghp_test123"
    assert "application/vnd.github" in headers["Accept"]


def test_token_rotator_requires_at_least_one():
    """Rotator should raise ValueError if no tokens provided."""
    with pytest.raises(ValueError, match="At least one"):
        GitHubTokenRotator([])


# ── Test 12: Phase 2 full pipeline with mock ─────────────────────────────

def test_phase2_full_pipeline():
    """End-to-end Phase 2: index entry -> commit fetch -> diff extract -> sample save."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"

        # Create index with one entry
        output_dir.mkdir(parents=True)
        index_path = output_dir / "cve_index.jsonl"
        sha = "a1b2c3d4e5" * 4  # 40 chars
        entry = {
            "cve_id": "CVE-2024-9999",
            "github_commit_url": f"https://github.com/test/repo/commit/{sha}",
            "cwe": "CWE-120",
        }
        index_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        # Build a realistic unified diff with >= 5 changed lines
        # Need at least MIN_CHANGED_LINES (5) additions+deletions
        patch_text = "\n".join([
            "@@ -1,12 +1,14 @@",
            " #include <stdio.h>",
            " #include <string.h>",
            " void process(const char *input) {",
            " ",
            "     char buffer[64];",
            "-    strcpy(buffer, input);",
            "-    printf(buffer);",
            "-    strcat(buffer, input);",
            "+    strncpy(buffer, input, sizeof(buffer)-1);",
            "+    buffer[sizeof(buffer)-1] = '\\0';",
            "+    printf(\"%s\", buffer);",
            "+    strncat(buffer, input, sizeof(buffer)-strlen(buffer)-1);",
            " ",
            "     int len = strlen(buffer);",
            " ",
            "     return;",
            " }",
        ])

        mock_commit = {
            "sha": sha,
            "files": [
                {
                    "filename": "src/vuln.c",
                    "status": "modified",
                    "patch": patch_text,
                },
                {
                    "filename": "README.md",
                    "status": "modified",
                    "patch": "+some text\n-old text\n" * 5,
                },
            ],
        }

        fetcher = GitHubDiffFetcher(
            output_dir=str(output_dir),
            github_tokens=["test_token"],
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_commit
        mock_resp.headers = {"X-RateLimit-Remaining": "4999", "X-RateLimit-Reset": str(time.time() + 3600)}

        with patch.object(fetcher, 'safe_get', return_value=mock_resp):
            summary = fetcher.collect()

        # Should have processed the .c file, rejected .md
        assert summary["commits_processed"] == 1
        assert summary["files_qualifying"] >= 1
        assert summary["files_rejected_ext"] >= 1  # README.md rejected


# ── Test 13: Stale 404 logged ────────────────────────────────────────────

def test_stale_404_logged():
    """404 commits should be logged to cve_stale.jsonl."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"

        output_dir.mkdir(parents=True)
        sha = "d" * 40
        index_path = output_dir / "cve_index.jsonl"
        entry = {
            "cve_id": "CVE-2024-STALE",
            "github_commit_url": f"https://github.com/gone/repo/commit/{sha}",
            "cwe": "CWE-78",
        }
        index_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        fetcher = GitHubDiffFetcher(
            output_dir=str(output_dir),
            github_tokens=["test_token"],
        )

        # Mock safe_get to return 404
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.headers = {"X-RateLimit-Remaining": "4999", "X-RateLimit-Reset": str(time.time() + 3600)}

        with patch.object(fetcher, 'safe_get', return_value=mock_resp):
            summary = fetcher.collect()

        assert summary["commits_stale_404"] == 1

        stale_path = output_dir / "cve_stale.jsonl"
        assert stale_path.exists()
        with open(stale_path, encoding="utf-8") as f:
            stale_entries = [json.loads(l) for l in f if l.strip()]
        assert len(stale_entries) == 1
        assert stale_entries[0]["cve_id"] == "CVE-2024-STALE"


# ── Test 14: URL parsing ────────────────────────────────────────────────

def test_github_url_parsing():
    """parse_github_commit_url should extract owner/repo/sha."""
    url = "https://github.com/openssl/openssl/commit/abc123def456"
    result = GitHubDiffFetcher.parse_github_commit_url(url)
    assert result == ("openssl", "openssl", "abc123def456")


def test_github_url_parsing_invalid():
    """Invalid URLs should return None."""
    assert GitHubDiffFetcher.parse_github_commit_url("https://example.com") is None
    assert GitHubDiffFetcher.parse_github_commit_url("not a url") is None


# ── Test 15: CWE extraction from weaknesses ─────────────────────────────

def test_cwe_extraction():
    """Should extract valid CWE from NVD weaknesses structure."""
    weaknesses = [
        {"description": [{"value": "CWE-120"}]}
    ]
    result = NVDIndexBuilder.extract_cwe_from_weaknesses(weaknesses)
    assert result == "CWE-120"


def test_cwe_extraction_invalid_cwe():
    """Non-target CWEs should return None."""
    weaknesses = [
        {"description": [{"value": "CWE-999"}]}
    ]
    result = NVDIndexBuilder.extract_cwe_from_weaknesses(weaknesses)
    assert result is None


def test_cwe_extraction_empty():
    """Empty weaknesses should return None."""
    assert NVDIndexBuilder.extract_cwe_from_weaknesses([]) is None


# ── Test 16: changed_lines counting ──────────────────────────────────────

def test_count_changed_lines():
    """Should count additions and deletions but not context or file headers."""
    patch = """\
--- a/file.c
+++ b/file.c
@@ -1,5 +1,5 @@
 context line
-removed line 1
-removed line 2
+added line 1
+added line 2
+added line 3
 another context"""

    count = GitHubDiffFetcher.count_changed_lines(patch)
    # 2 removals + 3 additions = 5 (--- and +++ are excluded)
    assert count == 5


# ── Test 17: Phase 1 dry-run no files ────────────────────────────────────

def test_phase1_dry_run():
    """Phase 1 dry-run should not write index file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        builder = NVDIndexBuilder(
            output_dir=tmpdir,
            nvd_api_key="test_key",
            dry_run=True,
            max_samples=5,
        )

        mock_response = {
            "totalResults": 1,
            "vulnerabilities": [
                {
                    "cve": {
                        "id": "CVE-2024-DRY",
                        "references": [
                            {"url": "https://github.com/o/r/commit/" + "f" * 40},
                        ],
                        "weaknesses": [{"description": [{"value": "CWE-120"}]}],
                        "published": "2024-01-01",
                    }
                }
            ],
        }

        call_count = [0]
        def mock_query(cwe, start_index=0):
            call_count[0] += 1
            if call_count[0] <= 1:
                return mock_response
            return {"totalResults": 0, "vulnerabilities": []}

        with patch.object(builder, '_query_nvd', side_effect=mock_query):
            summary = builder.collect()

        assert summary["index_entries_saved"] >= 1
        index_path = Path(tmpdir) / "cve_index.jsonl"
        assert not index_path.exists(), "Dry-run should not write index file"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
