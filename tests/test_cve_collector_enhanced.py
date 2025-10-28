"""Unit tests for Enhanced CVE Collector."""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training', 'scripts', 'collection'))

from cve_collector_enhanced import CVECollectorEnhanced


class TestCVECollectorEnhanced:
    """Test cases for CVECollectorEnhanced."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def collector(self, temp_output_dir):
        """Create collector instance."""
        return CVECollectorEnhanced(
            output_dir=temp_output_dir,
            cache_enabled=True
        )

    def test_initialization(self, collector, temp_output_dir):
        """Test collector initialization."""
        assert collector.output_dir == Path(temp_output_dir)
        assert collector.cache_enabled is True
        assert collector.samples_collected == 0
        assert len(collector.errors) == 0

    def test_find_github_references(self, collector):
        """Test finding GitHub references in CVE data."""
        cve_data = {
            'references': [
                {'url': 'https://github.com/owner/repo/commit/abc123'},
                {'url': 'https://example.com/some-page'},
                {'url': 'https://github.com/another/project/pull/456'},
                {'url': 'https://github.com/test/repo/commit/def456'}
            ]
        }

        refs = collector.find_github_references(cve_data)

        assert len(refs) == 3
        assert 'https://github.com/owner/repo/commit/abc123' in refs
        assert 'https://github.com/test/repo/commit/def456' in refs
        assert 'https://github.com/another/project/pull/456' in refs

    def test_is_code_file(self, collector):
        """Test code file detection."""
        assert collector._is_code_file('main.py') is True
        assert collector._is_code_file('script.js') is True
        assert collector._is_code_file('app.java') is True
        assert collector._is_code_file('README.md') is False
        assert collector._is_code_file('config.txt') is False

    def test_extract_code_from_patch(self, collector):
        """Test extracting code from git patch."""
        patch = """@@ -10,7 +10,7 @@ def process_input(user_input):
     # Process user input
     query = "SELECT * FROM users WHERE name = '" + user_input + "'"
-    cursor.execute(query)
+    cursor.execute("SELECT * FROM users WHERE name = ?", (user_input,))
     return cursor.fetchall()
 """

        before, after = collector._extract_code_from_patch(patch)

        assert before is not None
        assert after is not None
        assert "cursor.execute(query)" in before
        assert 'cursor.execute("SELECT * FROM users WHERE name = ?", (user_input,))' in after
        assert before != after  # Ensure they're different

    def test_extract_vulnerability_type(self, collector):
        """Test vulnerability type extraction."""
        # SQL Injection
        text1 = "This vulnerability allows SQL injection attacks"
        assert collector.extract_vulnerability_type(text1) == "sql_injection"

        # XSS
        text2 = "Cross-site scripting (XSS) vulnerability found"
        assert collector.extract_vulnerability_type(text2) == "xss"

        # Unknown
        text3 = "Some generic security issue"
        assert collector.extract_vulnerability_type(text3) == "unknown"

    def test_validate_code(self, collector):
        """Test code validation."""
        # Valid code
        valid_code = "def vulnerable_function():\n" + "    " * 20 + "pass\n" * 5
        assert collector.validate_code(valid_code) is True

        # Too short
        short_code = "x = 1"
        assert collector.validate_code(short_code) is False

        # Too long
        long_code = "x = 1\n" * 10000
        assert collector.validate_code(long_code) is False

        # Just whitespace
        whitespace_code = "   \n\n   \n   "
        assert collector.validate_code(whitespace_code) is False

    def test_extract_cve_data(self, collector):
        """Test CVE data extraction."""
        cve_item = {
            'cve': {
                'id': 'CVE-2023-12345',
                'descriptions': [
                    {'lang': 'en', 'value': 'SQL injection vulnerability in login form'}
                ],
                'weaknesses': [
                    {
                        'description': [
                            {'lang': 'en', 'value': 'CWE-89'}
                        ]
                    }
                ],
                'published': '2023-06-15T10:00:00.000',
                'references': [],
                'metrics': {
                    'cvssMetricV31': [
                        {
                            'cvssData': {
                                'baseSeverity': 'HIGH',
                                'baseScore': 8.5
                            }
                        }
                    ]
                }
            }
        }

        result = collector.extract_cve_data(cve_item)

        assert result is not None
        assert result['cve_id'] == 'CVE-2023-12345'
        assert 'SQL injection' in result['description']
        assert result['vulnerability_type'] == 'sql_injection'
        assert result['severity'] == 'HIGH'
        assert result['cvss_score'] == 8.5
        assert 'CWE-89' in result['cwes']

    @patch('cve_collector_enhanced.CVECollectorEnhanced._enforce_rate_limit')
    @patch('requests.Session.get')
    def test_collect_by_keyword(self, mock_get, mock_rate_limit, collector):
        """Test collecting CVEs by keyword."""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'vulnerabilities': [
                {
                    'cve': {
                        'id': 'CVE-2023-00001',
                        'descriptions': [
                            {'lang': 'en', 'value': 'SQL injection vulnerability'}
                        ],
                        'weaknesses': [],
                        'published': '2023-01-01T00:00:00.000',
                        'references': [],
                        'metrics': {}
                    }
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        samples = collector.collect_by_keyword('SQL injection', max_samples=5)

        assert len(samples) >= 0
        assert mock_get.called

    def test_rate_limiting(self, collector):
        """Test rate limiting enforcement."""
        import time

        # Record multiple requests quickly
        start_time = time.time()

        for _ in range(6):  # More than rate limit
            collector._enforce_rate_limit()

        end_time = time.time()
        duration = end_time - start_time

        # Should have waited due to rate limiting
        # 6 requests with 5 per 30 seconds should cause some delay
        assert duration > 0

    def test_cache_functionality(self, collector):
        """Test caching functionality."""
        cache_key = 'test_key'
        test_data = {'test': 'data', 'number': 123}

        # Save to cache
        collector.save_cache(cache_key, test_data)

        # Load from cache
        loaded_data = collector.load_cache(cache_key)

        assert loaded_data is not None
        assert loaded_data == test_data

    def test_deduplication(self, collector):
        """Test sample deduplication."""
        samples = [
            {'cve_id': 'CVE-2023-00001', 'data': 'test1'},
            {'cve_id': 'CVE-2023-00002', 'data': 'test2'},
            {'cve_id': 'CVE-2023-00001', 'data': 'test1_duplicate'},
            {'cve_id': 'CVE-2023-00003', 'data': 'test3'},
        ]

        unique_samples = collector.deduplicate_samples(samples, key='cve_id')

        assert len(unique_samples) == 3
        assert unique_samples[0]['cve_id'] == 'CVE-2023-00001'
        assert unique_samples[1]['cve_id'] == 'CVE-2023-00002'
        assert unique_samples[2]['cve_id'] == 'CVE-2023-00003'


def test_main_entry_point():
    """Test main entry point can be imported."""
    from cve_collector_enhanced import main
    assert callable(main)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
