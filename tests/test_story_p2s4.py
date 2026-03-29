#!/usr/bin/env python3
"""
tests/test_story_p2s4.py

Tests for P2-S4: GitHub Advisory GraphQL Collector.
Covers:
  1. C project filter — openssl kept
  2. C project filter — python-requests rejected
  3. Cursor recovery on GraphQL error
  4. Seen advisory IDs skipped on restart
  5. CWE from cwes.nodes
  6. CWE inference fallback from description
  7. Rate limit sleep when remaining < 400
  8. Checkpoint saved after each page
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "training" / "scripts" / "collection")
)

from training.scripts.collection.github_advisory_collector import (
    GitHubAdvisoryCollector,
    C_PROJECT_KEYWORDS,
    RATE_LIMIT_FLOOR,
)


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_advisory(
    ghsa_id: str = "GHSA-test-0001",
    package_name: str = "openssl",
    ecosystem: str = "OTHER",
    cwe_id: str | None = "CWE-120",
    summary: str = "Buffer overflow in openssl",
    description: str = "",
    commit_url: str | None = "https://github.com/openssl/openssl/commit/" + "a" * 40,
    cvss_score: float = 7.5,
) -> dict:
    """Build a minimal advisory node matching the GraphQL response shape."""
    cwes_nodes = []
    if cwe_id:
        cwes_nodes.append({"cweId": cwe_id, "name": cwe_id})

    references = []
    if commit_url:
        references.append({"url": commit_url})

    return {
        "ghsaId": ghsa_id,
        "summary": summary,
        "description": description,
        "publishedAt": "2025-01-01T00:00:00Z",
        "vulnerabilities": {
            "nodes": [
                {
                    "package": {
                        "name": package_name,
                        "ecosystem": ecosystem,
                    }
                }
            ]
        },
        "references": references,
        "cvss": {"score": cvss_score},
        "cwes": {"nodes": cwes_nodes},
    }


# ── Test 1: C project filter — openssl kept ──────────────────────────────

def test_c_project_filter_openssl_kept():
    """Advisory with 'openssl' package name should be kept."""
    advisory = _make_advisory(package_name="openssl")
    assert GitHubAdvisoryCollector._is_c_project(advisory) is True


def test_c_project_filter_linux_kernel_kept():
    """Advisory with 'linux-kernel' package name should match 'linux' keyword."""
    advisory = _make_advisory(package_name="linux-kernel")
    assert GitHubAdvisoryCollector._is_c_project(advisory) is True


# ── Test 2: C project filter — python-requests rejected ──────────────────

def test_c_project_filter_python_rejected():
    """Advisory with 'python-requests' package should be rejected."""
    advisory = _make_advisory(package_name="python-requests")
    assert GitHubAdvisoryCollector._is_c_project(advisory) is False


def test_c_project_filter_lodash_rejected():
    """Advisory with 'lodash' (JS library) should be rejected."""
    advisory = _make_advisory(package_name="lodash")
    assert GitHubAdvisoryCollector._is_c_project(advisory) is False


# ── Test 3: Cursor recovery on GraphQL error ─────────────────────────────

def test_cursor_recovery_on_graphql_error():
    """GraphQL error with 'cursor' in message should clear checkpoint cursor and restart."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = GitHubAdvisoryCollector(
            output_dir=tmpdir,
            github_tokens=["test_token"],
            max_samples=5,
        )

        call_count = [0]
        checkpoints_saved = []

        original_save_checkpoint = collector.save_checkpoint

        def track_checkpoint(state):
            checkpoints_saved.append(state.copy())
            original_save_checkpoint(state)

        def mock_query(after_cursor=None):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: return cursor error
                return {"_cursor_error": True}
            elif call_count[0] == 2:
                # Second call (after restart): return empty result to stop
                return {
                    "data": {
                        "securityAdvisories": {
                            "nodes": [],
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                        },
                        "rateLimit": {"remaining": 5000, "resetAt": "2026-01-01T00:00:00Z"},
                    }
                }
            return None

        with patch.object(collector, '_query_advisories', side_effect=mock_query), \
             patch.object(collector, 'save_checkpoint', side_effect=track_checkpoint):
            collector.collect()

        # Should have been called twice (error + retry)
        assert call_count[0] == 2
        # First checkpoint after cursor error should have last_cursor = None
        assert checkpoints_saved[0]["last_cursor"] is None


# ── Test 4: Seen advisory IDs skipped on restart ─────────────────────────

def test_seen_advisory_ids_skipped():
    """Previously seen ghsaId should be skipped on restart."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write checkpoint with a seen advisory ID
        cp_dir = tmpdir / "checkpoints"
        cp_dir.mkdir()
        cp_path = cp_dir / "github_advisory_checkpoint.json"
        cp_path.write_text(json.dumps({
            "last_cursor": None,
            "seen_advisory_ids": ["GHSA-already-seen"],
            "page_count": 0,
            "samples_saved": 0,
        }), encoding="utf-8")

        collector = GitHubAdvisoryCollector(
            output_dir=str(tmpdir),
            checkpoint_dir=str(cp_dir),
            github_tokens=["test_token"],
            max_samples=10,
        )

        # Build a page with one already-seen and one new advisory
        seen_advisory = _make_advisory(ghsa_id="GHSA-already-seen", package_name="openssl")
        new_advisory = _make_advisory(ghsa_id="GHSA-brand-new", package_name="curl")

        processed_ghsa_ids = []
        original_process = collector._process_advisory

        def tracking_process(advisory):
            processed_ghsa_ids.append(advisory.get("ghsaId"))
            # Don't actually process to avoid needing commit mock
            return

        def mock_query(after_cursor=None):
            return {
                "data": {
                    "securityAdvisories": {
                        "nodes": [seen_advisory, new_advisory],
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                    },
                    "rateLimit": {"remaining": 5000, "resetAt": "2026-01-01T00:00:00Z"},
                }
            }

        with patch.object(collector, '_query_advisories', side_effect=mock_query), \
             patch.object(collector, '_process_advisory', side_effect=tracking_process):
            collector.collect()

        # Only the new advisory should be processed
        assert "GHSA-brand-new" in processed_ghsa_ids
        assert "GHSA-already-seen" not in processed_ghsa_ids


# ── Test 5: CWE from cwes.nodes ──────────────────────────────────────────

def test_cwe_from_cwes_nodes():
    """CWE should be populated from advisory cwes.nodes[0].cweId."""
    advisory = _make_advisory(cwe_id="CWE-416")
    result = GitHubAdvisoryCollector._extract_cwe(advisory)
    assert result == "CWE-416"


def test_cwe_from_cwes_nodes_invalid_skipped():
    """Non-target CWE in cwes.nodes should fall through to keyword inference."""
    advisory = _make_advisory(
        cwe_id=None,
        summary="integer overflow vulnerability",
        description="",
    )
    # Manually set a non-target CWE in nodes
    advisory["cwes"] = {"nodes": [{"cweId": "CWE-999", "name": "CWE-999"}]}
    result = GitHubAdvisoryCollector._extract_cwe(advisory)
    # Should fall through to keyword inference: "integer overflow" -> CWE-190
    assert result == "CWE-190"


# ── Test 6: CWE inference fallback from description ──────────────────────

def test_cwe_inference_fallback():
    """When cwes.nodes is empty, CWE should be inferred from description keywords."""
    advisory = _make_advisory(
        cwe_id=None,
        summary="strcpy buffer overflow in parser",
        description="The function uses strcpy() without bounds checking",
    )
    advisory["cwes"] = {"nodes": []}
    result = GitHubAdvisoryCollector._extract_cwe(advisory)
    assert result == "CWE-120"  # strcpy( matches CWE-120


def test_cwe_inference_no_match():
    """When no CWE nodes and no keyword match, should return None."""
    advisory = _make_advisory(
        cwe_id=None,
        summary="something unrelated",
        description="no vulnerability keywords here",
    )
    advisory["cwes"] = {"nodes": []}
    result = GitHubAdvisoryCollector._extract_cwe(advisory)
    assert result is None


# ── Test 7: Rate limit sleep when remaining < 400 ────────────────────────

def test_rate_limit_sleep():
    """When remaining < RATE_LIMIT_FLOOR (400), collector should sleep."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = GitHubAdvisoryCollector(
            output_dir=tmpdir,
            github_tokens=["test_token"],
        )

        rate_limit_data = {
            "remaining": 50,  # Below floor of 400
            "resetAt": "2026-01-01T00:00:00Z",
        }

        with patch("time.sleep") as mock_sleep:
            collector._check_graphql_rate_limit(rate_limit_data)

        mock_sleep.assert_called_once()
        sleep_secs = mock_sleep.call_args[0][0]
        assert sleep_secs > 0


def test_rate_limit_no_sleep_when_above_floor():
    """When remaining >= RATE_LIMIT_FLOOR, no sleep should occur."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = GitHubAdvisoryCollector(
            output_dir=tmpdir,
            github_tokens=["test_token"],
        )

        rate_limit_data = {
            "remaining": 4000,  # Above floor
            "resetAt": "2026-01-01T00:00:00Z",
        }

        with patch("time.sleep") as mock_sleep:
            collector._check_graphql_rate_limit(rate_limit_data)

        mock_sleep.assert_not_called()


# ── Test 8: Checkpoint saved after each page ─────────────────────────────

def test_checkpoint_saved_after_each_page():
    """save_checkpoint should be called after each page response is received."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = GitHubAdvisoryCollector(
            output_dir=tmpdir,
            github_tokens=["test_token"],
            max_samples=100,
        )

        page1_advisory = _make_advisory(
            ghsa_id="GHSA-page1",
            package_name="django",
        )
        page2_advisory = _make_advisory(
            ghsa_id="GHSA-page2",
            package_name="flask",
        )

        call_count = [0]

        def mock_query(after_cursor=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return {
                    "data": {
                        "securityAdvisories": {
                            "nodes": [page1_advisory],
                            "pageInfo": {"hasNextPage": True, "endCursor": "cursor_page1"},
                        },
                        "rateLimit": {"remaining": 5000, "resetAt": "2026-01-01T00:00:00Z"},
                    }
                }
            elif call_count[0] == 2:
                return {
                    "data": {
                        "securityAdvisories": {
                            "nodes": [page2_advisory],
                            "pageInfo": {"hasNextPage": False, "endCursor": "cursor_page2"},
                        },
                        "rateLimit": {"remaining": 4999, "resetAt": "2026-01-01T00:00:00Z"},
                    }
                }
            return None

        checkpoint_calls = []
        original_save_cp = collector.save_checkpoint

        def track_save_cp(state):
            checkpoint_calls.append(state.copy())
            original_save_cp(state)

        with patch.object(collector, '_query_advisories', side_effect=mock_query), \
             patch.object(collector, 'save_checkpoint', side_effect=track_save_cp), \
             patch.object(collector, '_process_advisory'):
            collector.collect()

        # Should have at least one checkpoint per page (2 pages) plus final
        assert len(checkpoint_calls) >= 2
        # First page checkpoint should have cursor_page1
        assert checkpoint_calls[0]["last_cursor"] == "cursor_page1"
        assert checkpoint_calls[0]["page_count"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
