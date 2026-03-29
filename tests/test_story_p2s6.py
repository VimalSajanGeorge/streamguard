#!/usr/bin/env python3
"""
tests/test_story_p2s6.py

Unit tests for P2-S6: Repo Miner (API-Only).
9 tests covering scored keyword filter, repo order, checkpoint resume,
cross-dedup, SHA normalization, and --repo-filter flag.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the collection package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.scripts.collection.repo_miner_enhanced import (
    REPOS,
    RepoMiner,
)


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_miner(tmp_path, **kwargs):
    """Create a RepoMiner with a fake token and temp dirs."""
    defaults = dict(
        output_dir=str(tmp_path / "output"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        github_tokens=["fake_token_for_test"],
        dry_run=True,
    )
    defaults.update(kwargs)
    return RepoMiner(**defaults)


# ══════════════════════════════════════════════════════════════════════════
# Test 1: Scored keyword filter — 'cve-2024' → score 3 → fetch diff
# ══════════════════════════════════════════════════════════════════════════

def test_keyword_score_cve_high(tmp_path):
    """Commit message with 'cve-2024' should score >= 2 (HIGH_WEIGHT=3)."""
    score, has_exclusion = RepoMiner.score_commit_message(
        "Fix CVE-2024-1234: buffer overflow in TLS handshake"
    )
    assert not has_exclusion
    assert score >= 2, f"Expected score >= 2 for CVE mention, got {score}"
    # 'cve-' = 3, 'overflow' = 2, 'buffer' = 2, 'fix' = 1 → should be 8+
    assert score >= 3, f"Expected HIGH_WEIGHT >= 3, got {score}"


# ══════════════════════════════════════════════════════════════════════════
# Test 2: Scored keyword filter — 'fix typo' → exclusion match → skip
# ══════════════════════════════════════════════════════════════════════════

def test_keyword_exclusion_typo(tmp_path):
    """Commit message with 'typo' should be excluded regardless of other keywords."""
    score, has_exclusion = RepoMiner.score_commit_message("fix typo in security docs")
    assert has_exclusion is True
    assert score == 0


# ══════════════════════════════════════════════════════════════════════════
# Test 3: Scored keyword filter — 'fix buffer overflow' → score 3 → fetch
# ══════════════════════════════════════════════════════════════════════════

def test_keyword_score_buffer_overflow(tmp_path):
    """'fix buffer overflow' has fix(1) + buffer(2) + overflow(2) = 5."""
    score, has_exclusion = RepoMiner.score_commit_message("fix buffer overflow")
    assert not has_exclusion
    assert score >= 2, f"Expected score >= 2, got {score}"
    assert score == 5, f"Expected fix(1)+buffer(2)+overflow(2)=5, got {score}"


# ══════════════════════════════════════════════════════════════════════════
# Test 4: Scored keyword filter — 'update changelog' → score 0 → skip
# ══════════════════════════════════════════════════════════════════════════

def test_keyword_score_zero(tmp_path):
    """'update changelog' has no matching keywords → score 0."""
    score, has_exclusion = RepoMiner.score_commit_message("update changelog")
    assert not has_exclusion
    assert score == 0


# ══════════════════════════════════════════════════════════════════════════
# Test 5: Repo order — torvalds/linux must be last in REPOS list
# ══════════════════════════════════════════════════════════════════════════

def test_linux_is_last_repo():
    """torvalds/linux must be the last entry in REPOS (slowest repo)."""
    assert REPOS[-1] == "torvalds/linux", (
        f"Expected torvalds/linux as last repo, got {REPOS[-1]}"
    )
    assert len(REPOS) == 15, f"Expected 15 repos, got {len(REPOS)}"


# ══════════════════════════════════════════════════════════════════════════
# Test 6: Per-repo checkpoint — complete repos skipped on resume
# ══════════════════════════════════════════════════════════════════════════

def test_checkpoint_skips_complete_repos(tmp_path):
    """Repos with status='complete' should be skipped on resume."""
    miner = _make_miner(tmp_path, repo_filter="openssl/openssl")

    # Simulate a checkpoint where openssl is already complete
    checkpoint = {
        "repo_status": {"openssl/openssl": "complete"},
        "repo_page": {"openssl/openssl": 100},
        "done_shas": ["abc123def456" * 3 + "abcd"],
        "total_saved": 42,
    }
    miner.save_checkpoint(checkpoint)

    # Collect should skip openssl entirely (no API calls needed)
    with patch.object(miner, "_list_commits") as mock_list:
        summary = miner.collect()
        mock_list.assert_not_called()

    assert miner.repos_completed == 1


# ══════════════════════════════════════════════════════════════════════════
# Test 7: Cross-dedup — commit_sha in cve_shas → skipped
# ══════════════════════════════════════════════════════════════════════════

def test_cross_dedup_skips_known_sha(tmp_path):
    """Commits already in CVE dataset should be skipped."""
    # Create a fake cve_samples.jsonl with a known SHA
    known_sha = "a" * 40
    cve_dir = tmp_path / "cve"
    cve_dir.mkdir()
    cve_path = cve_dir / "cve_samples.jsonl"
    cve_path.write_text(
        json.dumps({"commit_sha": known_sha, "code": "x", "id": "1"}) + "\n",
        encoding="utf-8",
    )

    miner = _make_miner(tmp_path, cve_samples_path=str(cve_path))
    assert known_sha in miner._dedup_shas


# ══════════════════════════════════════════════════════════════════════════
# Test 8: SHA normalization — short SHA resolved and cached
# ══════════════════════════════════════════════════════════════════════════

def test_sha_normalization_caches(tmp_path):
    """Short SHAs should be resolved via API and cached."""
    miner = _make_miner(tmp_path)

    full_sha = "a" * 40
    short_sha = "a" * 7

    # Mock safe_get to return a fake commit with full SHA
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"sha": full_sha}
    mock_resp.headers = {"X-RateLimit-Remaining": "4999", "X-RateLimit-Reset": "9999999999"}

    with patch.object(miner, "safe_get", return_value=mock_resp):
        result = miner._resolve_full_sha("openssl", "openssl", short_sha)
        assert result == full_sha

        # Second call should use cache (no API call)
        result2 = miner._resolve_full_sha("openssl", "openssl", short_sha)
        assert result2 == full_sha
        # safe_get should have been called only once
        assert miner.safe_get.call_count == 1

    # Already-full SHAs should not call API
    with patch.object(miner, "safe_get") as mock_get:
        result3 = miner._resolve_full_sha("openssl", "openssl", full_sha)
        assert result3 == full_sha
        mock_get.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════
# Test 9: --repo-filter flag — only specified repo processed
# ══════════════════════════════════════════════════════════════════════════

def test_repo_filter_limits_to_one(tmp_path):
    """--repo-filter should restrict collection to a single repo."""
    miner = _make_miner(tmp_path, repo_filter="curl/curl")

    # Mock _list_commits to return empty (end of commits)
    with patch.object(miner, "_list_commits", return_value=[]):
        summary = miner.collect()

    # Only curl/curl should have been processed
    assert miner.repos_completed == 1
    # Verify we didn't process other repos by checking checkpoint
    cp = miner.load_checkpoint()
    assert "curl/curl" in cp.get("repo_status", {})
    assert "openssl/openssl" not in cp.get("repo_status", {})
