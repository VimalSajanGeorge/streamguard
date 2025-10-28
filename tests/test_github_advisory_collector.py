"""Tests for Enhanced GitHub Security Advisory Collector."""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "training" / "scripts" / "collection"))

from github_advisory_collector_enhanced import GitHubAdvisoryCollectorEnhanced


@pytest.fixture
def mock_github_token():
    """Mock GitHub token."""
    with patch.dict(os.environ, {"GITHUB_TOKEN": "test_token_123"}):
        yield "test_token_123"


@pytest.fixture
def collector(tmp_path, mock_github_token):
    """Create collector instance with temporary directory."""
    return GitHubAdvisoryCollectorEnhanced(
        output_dir=str(tmp_path / "github"),
        cache_enabled=False
    )


@pytest.fixture
def sample_advisory():
    """Sample advisory data from GitHub API."""
    return {
        "ghsaId": "GHSA-test-1234-5678",
        "summary": "SQL Injection in example-package",
        "description": "A SQL injection vulnerability was found in the query builder.",
        "severity": "HIGH",
        "publishedAt": "2024-01-15T10:00:00Z",
        "updatedAt": "2024-01-15T10:00:00Z",
        "identifiers": [
            {"type": "GHSA", "value": "GHSA-test-1234-5678"},
            {"type": "CVE", "value": "CVE-2024-1234"}
        ],
        "references": [
            {"url": "https://github.com/owner/repo/commit/abc123def456"},
            {"url": "https://github.com/owner/repo/security/advisories/GHSA-test-1234-5678"}
        ],
        "vulnerabilities": {
            "nodes": [
                {
                    "package": {
                        "name": "example-package",
                        "ecosystem": "PIP"
                    },
                    "vulnerableVersionRange": "< 1.2.3",
                    "firstPatchedVersion": {
                        "identifier": "1.2.3"
                    }
                }
            ]
        }
    }


@pytest.fixture
def sample_commit_response():
    """Sample commit response from GitHub API."""
    return {
        "sha": "abc123def456",
        "files": [
            {
                "filename": "src/query_builder.py",
                "status": "modified",
                "patch": """@@ -10,7 +10,7 @@ def execute_query(user_input):
 def execute_query(user_input):
     # Vulnerable code
-    query = "SELECT * FROM users WHERE name = '" + user_input + "'"
+    query = "SELECT * FROM users WHERE name = ?"
-    cursor.execute(query)
+    cursor.execute(query, (user_input,))
     return cursor.fetchall()
"""
            }
        ]
    }


@pytest.fixture
def sample_pypi_response():
    """Sample PyPI package metadata."""
    return {
        "info": {
            "name": "example-package",
            "version": "1.2.3",
            "project_urls": {
                "Source": "https://github.com/owner/repo"
            },
            "home_page": "https://github.com/owner/repo"
        }
    }


class TestGitHubAdvisoryCollectorEnhanced:
    """Test suite for GitHubAdvisoryCollectorEnhanced."""

    def test_initialization(self, collector):
        """Test collector initialization."""
        assert collector.github_token == "test_token_123"
        assert collector.GRAPHQL_API == "https://api.github.com/graphql"
        assert len(collector.ECOSYSTEMS) == 8
        assert len(collector.SEVERITIES) == 4

    def test_initialization_without_token(self):
        """Test that initialization fails without GitHub token."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="GITHUB_TOKEN"):
                GitHubAdvisoryCollectorEnhanced(output_dir="test")

    def test_process_advisory(self, collector, sample_advisory):
        """Test processing a single advisory."""
        with patch.object(collector, 'extract_code_with_diff', return_value=("vulnerable", "fixed")):
            sample = collector._process_advisory(sample_advisory, "PIP", "HIGH")

            assert sample is not None
            assert sample["advisory_id"] == "GHSA-test-1234-5678"
            assert sample["ecosystem"] == "PIP"
            assert sample["severity"] == "HIGH"
            assert sample["vulnerable_code"] == "vulnerable"
            assert sample["fixed_code"] == "fixed"
            assert "sql injection" in sample["description"].lower()
            assert sample["metadata"]["package_name"] == "example-package"
            assert sample["metadata"]["patched_version"] == "1.2.3"

    def test_parse_patch(self, collector):
        """Test parsing unified diff patch."""
        patch = """@@ -10,7 +10,7 @@ def execute_query(user_input):
-    query = "SELECT * FROM users WHERE name = '" + user_input + "'"
+    query = "SELECT * FROM users WHERE name = ?"
-    cursor.execute(query)
+    cursor.execute(query, (user_input,))
"""
        before, after = collector._parse_patch(patch)

        assert before is not None
        assert after is not None
        assert "SELECT * FROM users WHERE name = '" in before
        assert "SELECT * FROM users WHERE name = ?" in after
        assert "cursor.execute(query)" in before
        assert "cursor.execute(query, (user_input,))" in after

    def test_find_repo_from_references(self, collector):
        """Test extracting repository URL from references."""
        references = [
            "https://github.com/owner/repo/commit/abc123",
            "https://github.com/owner/repo/security/advisories/GHSA-1234"
        ]

        repo_url = collector._find_repo_from_references(references)
        assert repo_url == "https://github.com/owner/repo"

    def test_extract_commit_from_references(self, collector):
        """Test extracting commit SHA from references."""
        references = [
            "https://github.com/owner/repo/commit/abc123def456789012345678901234567890abcd",
            "https://example.com/advisory"
        ]

        commit_sha = collector._extract_commit_from_references(references)
        assert commit_sha == "abc123def456789012345678901234567890abcd"

    def test_is_code_file(self, collector):
        """Test code file detection."""
        assert collector._is_code_file("src/main.py") is True
        assert collector._is_code_file("app.js") is True
        assert collector._is_code_file("package.json") is False
        assert collector._is_code_file("README.md") is False
        assert collector._is_code_file("test.java") is True

    @patch('requests.get')
    def test_find_repo_for_package_pip(self, mock_get, collector, sample_pypi_response):
        """Test finding repository for PIP package."""
        mock_response = Mock()
        mock_response.json.return_value = sample_pypi_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        repo_url = collector.find_repo_for_package("example-package", "PIP")

        assert repo_url == "https://github.com/owner/repo"
        mock_get.assert_called_once()

    @patch('requests.get')
    def test_find_repo_for_package_npm(self, mock_get, collector):
        """Test finding repository for NPM package."""
        npm_response = {
            "name": "example-package",
            "repository": {
                "type": "git",
                "url": "git+https://github.com/owner/repo.git"
            }
        }

        mock_response = Mock()
        mock_response.json.return_value = npm_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        repo_url = collector.find_repo_for_package("example-package", "NPM")

        assert repo_url == "https://github.com/owner/repo"

    @patch('requests.post')
    def test_query_advisories(self, mock_post, collector):
        """Test GraphQL query execution."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": {
                "securityAdvisories": {
                    "nodes": [{"ghsaId": "GHSA-test-1234"}],
                    "pageInfo": {
                        "hasNextPage": False,
                        "endCursor": None
                    }
                },
                "rateLimit": {
                    "cost": 1,
                    "remaining": 4999,
                    "resetAt": "2024-01-15T11:00:00Z"
                }
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = collector._query_advisories("PIP", "HIGH", first=100)

        assert result is not None
        assert "data" in result
        assert collector.rate_limit_remaining == 4999
        mock_post.assert_called_once()

    @patch('requests.get')
    def test_fetch_commit_diff(self, mock_get, collector, sample_commit_response):
        """Test fetching commit diff."""
        mock_response = Mock()
        mock_response.json.return_value = sample_commit_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        before, after = collector._fetch_commit_diff("owner", "repo", "abc123")

        assert before is not None
        assert after is not None
        assert "SELECT * FROM users WHERE name = '" in before
        assert "SELECT * FROM users WHERE name = ?" in after

    def test_extract_vulnerability_type(self, collector):
        """Test vulnerability type extraction."""
        assert collector.extract_vulnerability_type("SQL injection vulnerability") == "sql_injection"
        assert collector.extract_vulnerability_type("XSS attack possible") == "xss"
        assert collector.extract_vulnerability_type("Command injection bug") == "command_injection"
        assert collector.extract_vulnerability_type("Some other issue") == "unknown"

    def test_validate_code(self, collector):
        """Test code validation."""
        # Valid code
        valid_code = "def vulnerable_function():\n    query = 'SELECT * FROM users'\n    execute(query)"
        assert collector.validate_code(valid_code) is True

        # Too short
        short_code = "x = 1"
        assert collector.validate_code(short_code) is False

        # Empty
        assert collector.validate_code("") is False
        assert collector.validate_code(None) is False

    def test_deduplicate_samples(self, collector):
        """Test sample deduplication."""
        samples = [
            {"advisory_id": "GHSA-1", "code": "vulnerable1"},
            {"advisory_id": "GHSA-2", "code": "vulnerable2"},
            {"advisory_id": "GHSA-3", "code": "vulnerable1"},  # Duplicate code
        ]

        unique = collector.deduplicate_samples(samples, key="advisory_id")
        assert len(unique) == 3  # All have unique advisory_ids

        unique = collector.deduplicate_samples(samples, key="code")
        assert len(unique) == 2  # Two unique code samples

    def test_stats_tracking(self, collector, sample_advisory):
        """Test statistics tracking."""
        with patch.object(collector, 'extract_code_with_diff', return_value=("vuln", "fixed")):
            sample = collector._process_advisory(sample_advisory, "PIP", "HIGH")

            assert collector.stats["total_advisories"] == 1
            assert collector.stats["successful_extractions"] == 1
            assert collector.stats["by_ecosystem"]["PIP"] == 1
            assert collector.stats["by_severity"]["HIGH"] == 1

    def test_save_samples(self, collector, tmp_path):
        """Test saving samples to file."""
        samples = [
            {"advisory_id": "GHSA-1", "description": "Test 1"},
            {"advisory_id": "GHSA-2", "description": "Test 2"}
        ]

        output_file = collector.save_samples(samples, "test_output.jsonl")

        assert output_file.exists()

        # Read and verify
        with open(output_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2

            first_sample = json.loads(lines[0])
            assert first_sample["advisory_id"] == "GHSA-1"

    def test_rate_limit_checking(self, collector):
        """Test rate limit checking and waiting."""
        from datetime import timezone, timedelta

        collector.rate_limit_remaining = 50  # Below buffer

        with patch('time.sleep') as mock_sleep:
            # Set reset time to 10 seconds in future
            collector.rate_limit_reset_time = datetime.now(timezone.utc) + timedelta(seconds=10)

            collector._check_rate_limits()

            # Should have slept
            mock_sleep.assert_called_once()

    def test_cache_operations(self, tmp_path, mock_github_token):
        """Test cache save and load."""
        collector = GitHubAdvisoryCollectorEnhanced(
            output_dir=str(tmp_path / "github"),
            cache_enabled=True
        )

        # Save to cache
        test_data = {"key": "value", "number": 123}
        cache_key = collector.make_cache_key("test", "cache")
        collector.save_cache(cache_key, test_data)

        # Load from cache
        loaded_data = collector.load_cache(cache_key)
        assert loaded_data == test_data

    @patch('requests.post')
    def test_collect_by_ecosystem_severity(self, mock_post, collector):
        """Test collecting advisories for specific ecosystem and severity."""
        # Mock GraphQL response
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": {
                "securityAdvisories": {
                    "nodes": [
                        {
                            "ghsaId": "GHSA-test-1",
                            "summary": "Test advisory",
                            "description": "Test description",
                            "severity": "HIGH",
                            "publishedAt": "2024-01-15T10:00:00Z",
                            "references": [],
                            "vulnerabilities": {
                                "nodes": [
                                    {
                                        "package": {"name": "test-pkg", "ecosystem": "PIP"},
                                        "vulnerableVersionRange": "< 1.0",
                                        "firstPatchedVersion": {"identifier": "1.0"}
                                    }
                                ]
                            }
                        }
                    ],
                    "pageInfo": {
                        "hasNextPage": False,
                        "endCursor": None
                    }
                },
                "rateLimit": {"cost": 1, "remaining": 5000, "resetAt": "2024-01-15T11:00:00Z"}
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.object(collector, 'extract_code_with_diff', return_value=("vuln", "fixed")):
            samples = collector.collect_by_ecosystem_severity("PIP", "HIGH", max_samples=10)

            assert len(samples) >= 1
            assert samples[0]["ecosystem"] == "PIP"
            assert samples[0]["severity"] == "HIGH"


def test_main_entry_point():
    """Test main function entry point."""
    from github_advisory_collector_enhanced import main

    with patch('sys.argv', ['script.py', '--target-samples', '100', '--no-cache']):
        with patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"}):
            with patch.object(GitHubAdvisoryCollectorEnhanced, 'collect_all_advisories', return_value=[]):
                try:
                    main()
                except SystemExit:
                    pass  # argparse may cause exit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
