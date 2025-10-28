"""Enhanced CVE Collector for StreamGuard data collection.

Collects CVE samples from NVD API 2.0 with GitHub code extraction.
Implements parallel collection, rate limiting, and caching.
"""

import json
import time
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from base_collector import BaseCollector


class CVECollectorEnhanced(BaseCollector):
    """Enhanced CVE collector with GitHub code extraction."""

    # NVD API Configuration
    NVD_API_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    GITHUB_API_BASE = "https://api.github.com"

    # Rate limiting (5 requests per 30 seconds for public API)
    RATE_LIMIT_REQUESTS = 5
    RATE_LIMIT_WINDOW = 30  # seconds

    # Collection parameters
    TARGET_SAMPLES = 15000
    RESULTS_PER_PAGE = 2000
    START_YEAR = 2020
    END_YEAR = 2025

    # Extended vulnerability keywords
    KEYWORDS = [
        "SQL injection",
        "XSS",
        "cross-site scripting",
        "command injection",
        "path traversal",
        "SSRF",
        "XXE",
        "CSRF",
        "deserialization",
        "remote code execution",
        "directory traversal",
        "insecure deserialization"
    ]

    def __init__(self, output_dir: str, cache_enabled: bool = True,
                 github_token: Optional[str] = None):
        """
        Initialize enhanced CVE collector.

        Args:
            output_dir: Directory to save collected data
            cache_enabled: Whether to cache API responses
            github_token: Optional GitHub API token for higher rate limits
        """
        super().__init__(output_dir, cache_enabled)

        self.github_token = github_token
        self.last_request_times = []

        # Configure session with retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # GitHub authentication
        if self.github_token:
            self.session.headers.update({
                'Authorization': f'token {self.github_token}'
            })

    def collect(self) -> List[Dict]:
        """
        Collect CVE samples with parallel processing by keyword.

        Returns:
            List of collected CVE samples
        """
        print(f"\n{'='*80}")
        print(f"Enhanced CVE Collection Started")
        print(f"{'='*80}")
        print(f"Target samples: {self.TARGET_SAMPLES}")
        print(f"Time range: {self.START_YEAR}-{self.END_YEAR}")
        print(f"Keywords: {len(self.KEYWORDS)}")
        print(f"Cache enabled: {self.cache_enabled}")
        print(f"{'='*80}\n")

        # Collect CVEs in parallel by keyword
        samples = self.collect_cves_parallel()

        # Deduplicate based on CVE ID
        samples = self.deduplicate_samples(samples, key='cve_id')

        # Save samples
        output_file = self.save_samples(samples, 'cve_data.jsonl')

        # Print statistics
        stats = self.get_stats()
        print(f"\n{'='*80}")
        print(f"Collection Complete")
        print(f"{'='*80}")
        print(f"Total samples collected: {len(samples)}")
        print(f"Samples with code: {sum(1 for s in samples if s.get('vulnerable_code'))}")
        print(f"Errors encountered: {stats['errors_count']}")
        print(f"Output file: {output_file}")
        print(f"{'='*80}\n")

        return samples

    def collect_cves_parallel(self) -> List[Dict]:
        """
        Collect CVEs in parallel by keyword using multiprocessing.

        Returns:
            List of collected CVE samples
        """
        # Determine number of processes (use fewer to respect rate limits)
        num_processes = min(cpu_count(), 4)

        print(f"Starting parallel collection with {num_processes} processes...\n")

        all_samples = []
        samples_per_keyword = self.TARGET_SAMPLES // len(self.KEYWORDS)

        # Process keywords sequentially to better manage rate limits
        # (parallel processing within each keyword search)
        for keyword in self.KEYWORDS:
            print(f"\n--- Processing keyword: '{keyword}' ---")
            samples = self.collect_by_keyword(keyword, samples_per_keyword)
            all_samples.extend(samples)

            print(f"Collected {len(samples)} samples for '{keyword}'")
            print(f"Total samples so far: {len(all_samples)}")

            # Break if we've reached target
            if len(all_samples) >= self.TARGET_SAMPLES:
                print(f"\nReached target of {self.TARGET_SAMPLES} samples!")
                break

        return all_samples

    def collect_by_keyword(self, keyword: str, max_samples: int) -> List[Dict]:
        """
        Collect CVEs for a specific keyword.

        Args:
            keyword: Vulnerability keyword to search for
            max_samples: Maximum samples to collect for this keyword

        Returns:
            List of CVE samples matching the keyword
        """
        samples = []

        # NVD API has 120-day limit, so we need to query in chunks
        from datetime import datetime, timedelta

        end_date = datetime.now()
        # Use last 120 days for recent data
        start_date = end_date - timedelta(days=119)

        start_index = 0

        while len(samples) < max_samples:
            # Check cache first
            cache_key = self.make_cache_key('nvd', keyword, start_index,
                                           start_date.strftime('%Y%m%d'),
                                           end_date.strftime('%Y%m%d'))
            cached_data = self.load_cache(cache_key)

            if cached_data:
                cves = cached_data.get('vulnerabilities', [])
                print(f"  Loaded {len(cves)} CVEs from cache (index: {start_index})")
            else:
                # Apply rate limiting
                self._enforce_rate_limit()

                # Fetch from NVD API with 120-day window
                try:
                    params = {
                        'keywordSearch': keyword,
                        'resultsPerPage': min(self.RESULTS_PER_PAGE, max_samples - len(samples)),
                        'startIndex': start_index,
                        'pubStartDate': start_date.strftime('%Y-%m-%dT00:00:00.000'),
                        'pubEndDate': end_date.strftime('%Y-%m-%dT23:59:59.999')
                    }

                    response = self.session.get(self.NVD_API_BASE, params=params, timeout=30)
                    response.raise_for_status()

                    data = response.json()
                    cves = data.get('vulnerabilities', [])

                    # Cache the response
                    self.save_cache(cache_key, data)

                    print(f"  Fetched {len(cves)} CVEs from API (index: {start_index})")

                except Exception as e:
                    self.log_error(f"Failed to fetch CVEs for keyword '{keyword}'",
                                 {'error': str(e), 'start_index': start_index})
                    break

            if not cves:
                break

            # Extract data from each CVE
            for cve_item in cves:
                if len(samples) >= max_samples:
                    break

                try:
                    cve_data = self.extract_cve_data(cve_item)
                    if cve_data:
                        samples.append(cve_data)
                        self.samples_collected += 1
                except Exception as e:
                    cve_id = cve_item.get('cve', {}).get('id', 'unknown')
                    self.log_error(f"Failed to extract data for CVE {cve_id}",
                                 {'error': str(e)})

            start_index += len(cves)

            # Break if we got fewer results than requested (end of results)
            if len(cves) < self.RESULTS_PER_PAGE:
                break

        return samples

    def extract_cve_data(self, cve_item: Dict) -> Optional[Dict]:
        """
        Parse CVE record and extract relevant data.

        Args:
            cve_item: CVE item from NVD API response

        Returns:
            Extracted CVE data or None if extraction fails
        """
        try:
            cve = cve_item.get('cve', {})
            cve_id = cve.get('id')

            if not cve_id:
                return None

            # Extract description
            descriptions = cve.get('descriptions', [])
            description = next((d.get('value', '') for d in descriptions
                              if d.get('lang') == 'en'), '')

            # Extract severity and CVSS score
            metrics = cve_item.get('cve', {}).get('metrics', {})
            severity = 'UNKNOWN'
            cvss_score = None

            # Try CVSS v3.1 first, then v3.0, then v2.0
            for version in ['cvssMetricV31', 'cvssMetricV30', 'cvssMetricV2']:
                if version in metrics and metrics[version]:
                    metric = metrics[version][0]
                    severity = metric.get('cvssData', {}).get('baseSeverity', 'UNKNOWN')
                    cvss_score = metric.get('cvssData', {}).get('baseScore')
                    break

            # Extract CWEs
            weaknesses = cve.get('weaknesses', [])
            cwes = []
            for weakness in weaknesses:
                for desc in weakness.get('description', []):
                    if desc.get('lang') == 'en':
                        cwe_value = desc.get('value', '')
                        if cwe_value.startswith('CWE-'):
                            cwes.append(cwe_value)

            # Extract published date
            published_date = cve.get('published', '')

            # Extract vulnerability type from description
            vulnerability_type = self.extract_vulnerability_type(description)

            # Find GitHub references and extract code
            github_refs = self.find_github_references(cve)
            vulnerable_code = None
            fixed_code = None
            code_source = None

            for ref in github_refs:
                try:
                    vuln, fixed, source = self.fetch_code_from_github(ref)
                    if vuln and fixed:
                        if self.validate_code(vuln) and self.validate_code(fixed):
                            vulnerable_code = vuln
                            fixed_code = fixed
                            code_source = source
                            break
                except Exception as e:
                    self.log_error(f"Failed to fetch code from GitHub for {cve_id}",
                                 {'error': str(e), 'ref': ref})

            return {
                'cve_id': cve_id,
                'description': description,
                'vulnerable_code': vulnerable_code,
                'fixed_code': fixed_code,
                'vulnerability_type': vulnerability_type,
                'severity': severity,
                'cvss_score': cvss_score,
                'cwes': cwes,
                'published_date': published_date,
                'source': code_source or 'nvd',
                'collected_at': datetime.now().isoformat()
            }

        except Exception as e:
            self.log_error("Failed to extract CVE data", {'error': str(e)})
            return None

    def find_github_references(self, cve: Dict) -> List[str]:
        """
        Find GitHub URLs in CVE references.

        Args:
            cve: CVE record from NVD API

        Returns:
            List of GitHub URLs found in references
        """
        github_urls = []

        references = cve.get('references', [])
        for ref in references:
            url = ref.get('url', '')

            # Match GitHub commit URLs
            if 'github.com' in url and '/commit/' in url:
                github_urls.append(url)
            # Match GitHub pull request URLs
            elif 'github.com' in url and '/pull/' in url:
                github_urls.append(url)

        return github_urls

    def fetch_code_from_github(self, github_url: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Fetch before/after code from GitHub commit.

        Args:
            github_url: GitHub commit or PR URL

        Returns:
            Tuple of (vulnerable_code, fixed_code, source_url)
        """
        # Parse GitHub URL to extract owner, repo, and commit hash
        # Example: https://github.com/owner/repo/commit/hash
        pattern = r'github\.com/([^/]+)/([^/]+)/commit/([a-f0-9]+)'
        match = re.search(pattern, github_url)

        if not match:
            return None, None, None

        owner, repo, commit_hash = match.groups()

        # Check cache
        cache_key = self.make_cache_key('github', owner, repo, commit_hash)
        cached_data = self.load_cache(cache_key)

        if cached_data:
            return (cached_data.get('vulnerable_code'),
                   cached_data.get('fixed_code'),
                   cached_data.get('source'))

        # Apply rate limiting for GitHub
        self._enforce_rate_limit()

        try:
            # Fetch commit data from GitHub API
            api_url = f"{self.GITHUB_API_BASE}/repos/{owner}/{repo}/commits/{commit_hash}"
            response = self.session.get(api_url, timeout=30)
            response.raise_for_status()

            commit_data = response.json()
            files = commit_data.get('files', [])

            vulnerable_code = None
            fixed_code = None

            # Look for code files with actual changes
            for file in files:
                filename = file.get('filename', '')

                # Focus on code files
                if not self._is_code_file(filename):
                    continue

                patch = file.get('patch', '')
                if not patch:
                    continue

                # Extract before and after code from patch
                vuln, fixed = self._extract_code_from_patch(patch)

                if vuln and fixed:
                    if self.validate_code(vuln) and self.validate_code(fixed):
                        vulnerable_code = vuln
                        fixed_code = fixed
                        break

            source = f"github:{owner}/{repo}:{commit_hash}"

            # Cache the result
            self.save_cache(cache_key, {
                'vulnerable_code': vulnerable_code,
                'fixed_code': fixed_code,
                'source': source
            })

            return vulnerable_code, fixed_code, source

        except Exception as e:
            self.log_error(f"Failed to fetch GitHub commit",
                         {'error': str(e), 'url': github_url})
            return None, None, None

    def _is_code_file(self, filename: str) -> bool:
        """Check if file is a code file based on extension."""
        code_extensions = {
            '.py', '.js', '.java', '.c', '.cpp', '.cs', '.rb', '.php',
            '.go', '.rs', '.swift', '.kt', '.scala', '.ts', '.jsx', '.tsx'
        }
        return any(filename.endswith(ext) for ext in code_extensions)

    def _extract_code_from_patch(self, patch: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract before and after code from git patch.

        Args:
            patch: Git patch string

        Returns:
            Tuple of (before_code, after_code)
        """
        lines = patch.split('\n')

        before_lines = []
        after_lines = []

        for line in lines:
            if line.startswith('@@'):
                continue
            elif line.startswith('-') and not line.startswith('---'):
                # Removed line (vulnerable code)
                before_lines.append(line[1:])
            elif line.startswith('+') and not line.startswith('+++'):
                # Added line (fixed code)
                after_lines.append(line[1:])
            elif line.startswith(' '):
                # Context line (appears in both)
                before_lines.append(line[1:])
                after_lines.append(line[1:])

        before_code = '\n'.join(before_lines).strip()
        after_code = '\n'.join(after_lines).strip()

        # Only return if there are actual differences
        if before_code and after_code and before_code != after_code:
            return before_code, after_code

        return None, None

    def _enforce_rate_limit(self):
        """
        Enforce rate limiting (5 requests per 30 seconds).
        """
        current_time = time.time()

        # Remove timestamps older than the rate limit window
        self.last_request_times = [
            t for t in self.last_request_times
            if current_time - t < self.RATE_LIMIT_WINDOW
        ]

        # If we've hit the limit, wait
        if len(self.last_request_times) >= self.RATE_LIMIT_REQUESTS:
            oldest_request = self.last_request_times[0]
            wait_time = self.RATE_LIMIT_WINDOW - (current_time - oldest_request)

            if wait_time > 0:
                print(f"  Rate limit reached, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                current_time = time.time()

        # Record this request
        self.last_request_times.append(current_time)


def main():
    """Main entry point for CVE collection."""
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced CVE Collector for StreamGuard')
    parser.add_argument('--output-dir', type=str,
                       default='C:\\Users\\Vimal Sajan\\streamguard\\data\\raw\\cves',
                       help='Output directory for collected data')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching')
    parser.add_argument('--github-token', type=str,
                       help='GitHub API token for higher rate limits')

    args = parser.parse_args()

    collector = CVECollectorEnhanced(
        output_dir=args.output_dir,
        cache_enabled=not args.no_cache,
        github_token=args.github_token
    )

    samples = collector.collect()

    print(f"\nCollection complete! Collected {len(samples)} CVE samples.")
    print(f"Data saved to: {args.output_dir}\\cve_data.jsonl")


if __name__ == '__main__':
    main()
