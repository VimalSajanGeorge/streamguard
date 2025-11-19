"""Enhanced CVE Collector for StreamGuard data collection.

Collects CVE samples from NVD API 2.0, enriches them with GitHub code diffs,
and enforces strict quality filters so the downstream CodexGlue/GNN training
pipeline receives balanced, production-grade samples.
"""

from __future__ import annotations

import argparse
import math
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    # When executed via `python -m training.scripts.collection.cve_collector_enhanced`
    from .base_collector import BaseCollector
except ImportError:  # pragma: no cover - fallback for running as script
    from base_collector import BaseCollector


@dataclass(frozen=True)
class DateWindow:
    """Represents a contiguous date window when querying the NVD API."""

    start: datetime
    end: datetime

    def to_api_params(self) -> Tuple[str, str]:
        """Return the ISO strings expected by the NVD API."""
        return (
            self.start.strftime("%Y-%m-%dT00:00:00.000"),
            self.end.strftime("%Y-%m-%dT23:59:59.999"),
        )

    def label(self) -> str:
        """Readable label for logging/caching."""
        return f"{self.start.date()}_{self.end.date()}"


class CVECollectorEnhanced(BaseCollector):
    """Enhanced CVE collector with GitHub code extraction."""

    # API endpoints
    NVD_API_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    GITHUB_API_BASE = "https://api.github.com"

    # Default collection parameters (can be overridden via CLI)
    TARGET_SAMPLES = 15_000
    RESULTS_PER_PAGE = 2000
    START_YEAR = 2020
    END_YEAR = datetime.utcnow().year
    WINDOW_DAYS = 90  # Keep under NVD 120-day constraint while overlapping a bit

    # Vulnerability keywords and CWE hints
    KEYWORDS: Sequence[str] = (
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
        "insecure deserialization",
    )

    CWE_HINTS = {
        "SQL injection": "CWE-89",
        "cross-site scripting": "CWE-79",
        "path traversal": "CWE-22",
        "command injection": "CWE-77",
        "SSRF": "CWE-918",
        "XXE": "CWE-611",
    }

    # Rate limit configuration (NVD public limit is 5 req / 30 seconds)
    NVD_RATE = (5, 30)
    # GitHub: assume token available -> stay well below 5k/hr, else 60/hr
    GITHUB_RATE_AUTH = (75, 60)  # 75 requests per 60 seconds
    GITHUB_RATE_NOAUTH = (30, 60)

    CODE_EXTENSIONS = {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".c",
        ".cpp",
        ".cs",
        ".rb",
        ".php",
        ".go",
        ".rs",
        ".swift",
        ".kt",
        ".scala",
    }

    def __init__(
        self,
        output_dir: str,
        cache_enabled: bool = True,
        github_token: Optional[str] = None,
        start_year: int = START_YEAR,
        end_year: int = END_YEAR,
        target_samples: int = TARGET_SAMPLES,
        keywords: Optional[Sequence[str]] = None,
    ):
        """
        Initialize enhanced CVE collector.

        Args:
            output_dir: Directory to save collected data.
            cache_enabled: Enable/disable request caching.
            github_token: Optional GitHub API token.
            start_year: Lower bound for CVE published year.
            end_year: Upper bound (inclusive).
            target_samples: Total samples to collect.
            keywords: Optional override for keyword list.
        """
        super().__init__(output_dir, cache_enabled)

        self.github_token = github_token
        self.start_year = start_year
        self.end_year = end_year
        self.TARGET_SAMPLES = target_samples
        self.keywords = list(keywords) if keywords else list(self.KEYWORDS)

        # Rate trackers per API namespace
        self.request_history = {
            "nvd": deque(),
            "github": deque(),
        }

        self.keyword_stats = defaultdict(lambda: {"samples": 0, "windows": 0})

        # Configure resilient HTTP session
        retry_strategy = Retry(
            total=5,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        if self.github_token:
            self.session.headers.update({"Authorization": f"token {self.github_token}"})

    # ------------------------------------------------------------------ #
    # Top-level orchestration
    # ------------------------------------------------------------------ #

    def collect(self) -> List[Dict]:
        """Collect CVE samples across keywords and date windows."""
        date_windows = self._generate_date_windows()
        per_keyword_cap = math.ceil(self.TARGET_SAMPLES / len(self.keywords))

        print(f"\n{'=' * 80}")
        print("StreamGuard CVE Collection")
        print(f"{'=' * 80}")
        print(f"Target samples          : {self.TARGET_SAMPLES}")
        print(f"Keywords                : {len(self.keywords)}")
        print(f"Date range              : {self.start_year} - {self.end_year}")
        print(f"Windows per keyword     : {len(date_windows)} (each {self.WINDOW_DAYS} days)")
        print(f"Cache enabled           : {self.cache_enabled}")
        print(f"GitHub token provided   : {bool(self.github_token)}")
        print(f"{'=' * 80}\n")

        all_samples: List[Dict] = []
        for keyword in self.keywords:
            if len(all_samples) >= self.TARGET_SAMPLES:
                break

            target_for_keyword = min(
                per_keyword_cap, self.TARGET_SAMPLES - len(all_samples)
            )

            print(f"\n--- Keyword: {keyword} | Target: {target_for_keyword} ---")
            keyword_samples = self._collect_keyword(keyword, date_windows, target_for_keyword)
            all_samples.extend(keyword_samples)
            self.keyword_stats[keyword]["samples"] = len(keyword_samples)

        # Deduplicate by CVE ID so we keep the best copy per vulnerability
        deduped = self.deduplicate_samples(all_samples, key="cve_id")
        output_file = self.save_samples(deduped, "cve_data.jsonl")

        stats = self.get_stats()
        samples_with_code = sum(1 for s in deduped if s.get("vulnerable_code"))

        print(f"\n{'=' * 80}")
        print("CVE Collection Complete")
        print(f"{'=' * 80}")
        print(f"Total samples saved     : {len(deduped)}")
        print(f"Samples w/ code pairs   : {samples_with_code}")
        print(f"Errors encountered      : {stats['errors_count']}")
        print(f"Output file             : {output_file}")
        print("Per keyword breakdown   :")
        for keyword, data in self.keyword_stats.items():
            print(
                f"  - {keyword:<22} -> {data['samples']:>5} samples "
                f"across {data['windows']} windows"
            )
        print(f"{'=' * 80}\n")

        return deduped

    # ------------------------------------------------------------------ #
    # Keyword / window processing
    # ------------------------------------------------------------------ #

    def _collect_keyword(
        self,
        keyword: str,
        date_windows: Sequence[DateWindow],
        ceiling: int,
    ) -> List[Dict]:
        """Collect CVEs for a single keyword across all date windows."""
        collected: List[Dict] = []

        for window in date_windows:
            if len(collected) >= ceiling:
                break

            remaining = ceiling - len(collected)
            print(
                f"   Window {window.label()} | remaining target {remaining} | "
                f"samples so far {len(collected)}"
            )

            window_samples = self._collect_window(keyword, window, remaining)
            collected.extend(window_samples)
            self.keyword_stats[keyword]["windows"] += 1

            if not window_samples:
                print(f"     No CVEs found for keyword '{keyword}' in this window.")

        return collected

    def _collect_window(
        self,
        keyword: str,
        window: DateWindow,
        remaining: int,
    ) -> List[Dict]:
        """Collect CVEs for a keyword in a specific window."""
        samples: List[Dict] = []
        start_index = 0

        while len(samples) < remaining:
            page = self._fetch_cve_page(keyword, window, start_index, remaining - len(samples))
            if not page:
                break

            for cve_item in page:
                if len(samples) >= remaining:
                    break
                sample = self.extract_cve_data(cve_item, window)
                if sample:
                    samples.append(sample)
                    self.samples_collected += 1

            if len(page) < self.RESULTS_PER_PAGE:
                break

            start_index += len(page)

        return samples

    # ------------------------------------------------------------------ #
    # API interactions
    # ------------------------------------------------------------------ #

    def _fetch_cve_page(
        self,
        keyword: str,
        window: DateWindow,
        start_index: int,
        limit: int,
    ) -> List[Dict]:
        """Fetch a page of CVEs from the NVD API."""
        cache_key = self.make_cache_key(
            "nvd",
            keyword,
            window.label(),
            start_index,
            limit,
        )

        cached = self.load_cache(cache_key)
        if cached:
            print(f"     Cache hit for keyword '{keyword}' at index {start_index}")
            return cached.get("vulnerabilities", [])

        params = {
            "keywordSearch": keyword,
            "resultsPerPage": min(self.RESULTS_PER_PAGE, limit),
            "startIndex": start_index,
            "pubStartDate": window.to_api_params()[0],
            "pubEndDate": window.to_api_params()[1],
        }

        self._enforce_rate_limit("nvd")

        try:
            response = self.session.get(self.NVD_API_BASE, params=params, timeout=45)
            response.raise_for_status()
            data = response.json()
            self.save_cache(cache_key, data)
            vulnerabilities = data.get("vulnerabilities", [])
            print(
                f"     Pulled {len(vulnerabilities)} CVEs "
                f"(keyword='{keyword}', index={start_index})"
            )
            return vulnerabilities
        except Exception as exc:  # pragma: no cover - network failures
            self.log_error(
                f"Failed to fetch CVEs for keyword '{keyword}'",
                {
                    "error": str(exc),
                    "keyword": keyword,
                    "start_index": start_index,
                    "window": window.label(),
                },
            )
            return []

    # ------------------------------------------------------------------ #
    # CVE parsing & GitHub enrichment
    # ------------------------------------------------------------------ #

    def extract_cve_data(self, cve_item: Dict, window: DateWindow) -> Optional[Dict]:
        """
        Parse CVE record and extract relevant data.

        Args:
            cve_item: CVE item from NVD API response
            window: Date window metadata (for tracing)

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
                    if vuln and fixed and self.validate_code(vuln) and self.validate_code(fixed):
                        vulnerable_code = vuln
                        fixed_code = fixed
                        code_source = source
                        break
                except Exception as e:
                    self.log_error(
                        f"Failed to fetch code from GitHub for {cve_id}",
                        {"error": str(e), "ref": ref},
                    )

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
                'collection_window': window.label(),
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

        cache_key = self.make_cache_key('github', owner, repo, commit_hash)
        cached_data = self.load_cache(cache_key)

        if cached_data:
            return (cached_data.get('vulnerable_code'),
                    cached_data.get('fixed_code'),
                    cached_data.get('source'))

        self._enforce_rate_limit("github")

        try:
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
        return any(filename.endswith(ext) for ext in self.CODE_EXTENSIONS)

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

    # ------------------------------------------------------------------ #
    # Rate limiting helpers
    # ------------------------------------------------------------------ #

    def _enforce_rate_limit(self, namespace: str):
        """Throttle outgoing requests to respect API limits."""
        if namespace == "nvd":
            limit, window = self.NVD_RATE
        elif namespace == "github":
            limit, window = (
                self.GITHUB_RATE_AUTH if self.github_token else self.GITHUB_RATE_NOAUTH
            )
        else:
            raise ValueError(f"Unknown rate limit namespace: {namespace}")

        history = self.request_history[namespace]
        now = time.time()

        while history and now - history[0] > window:
            history.popleft()

        if len(history) >= limit:
            wait_time = window - (now - history[0])
            if wait_time > 0:
                print(
                    f"     Rate limit reached for {namespace} API, "
                    f"sleeping {wait_time:.2f}s"
                )
                time.sleep(wait_time)
                now = time.time()
                while history and now - history[0] > window:
                    history.popleft()

        history.append(time.time())

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    def _generate_date_windows(self) -> List[DateWindow]:
        """Generate inclusive date windows that satisfy the NVD limitation."""
        windows: List[DateWindow] = []
        start = datetime(self.start_year, 1, 1)
        final = datetime(self.end_year, 12, 31)

        current = start
        while current <= final:
            window_end = min(current + timedelta(days=self.WINDOW_DAYS - 1), final)
            windows.append(DateWindow(start=current, end=window_end))
            current = window_end + timedelta(days=1)

        return windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="StreamGuard Enhanced CVE Collector")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("data/raw/cves").resolve()),
        help="Directory to store collected CVE data (default: data/raw/cves)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable request caching",
    )
    parser.add_argument(
        "--github-token",
        type=str,
        default=None,
        help="GitHub token for higher rate limits",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=CVECollectorEnhanced.START_YEAR,
        help="Earliest CVE publication year to include",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=CVECollectorEnhanced.END_YEAR,
        help="Latest CVE publication year to include",
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=CVECollectorEnhanced.TARGET_SAMPLES,
        help="Total number of CVE samples to collect",
    )
    parser.add_argument(
        "--keywords-file",
        type=str,
        default=None,
        help="Optional file with newline-separated keywords to override defaults",
    )
    return parser.parse_args()


def load_keywords_from_file(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    keyword_path = Path(path)
    if not keyword_path.exists():
        raise FileNotFoundError(f"Keyword file not found: {keyword_path}")
    with open(keyword_path, "r", encoding="utf-8") as handle:
        keywords = [line.strip() for line in handle if line.strip()]
    return keywords or None


def main():
    args = parse_args()
    keywords_override = load_keywords_from_file(args.keywords_file)

    collector = CVECollectorEnhanced(
        output_dir=args.output_dir,
        cache_enabled=not args.no_cache,
        github_token=args.github_token,
        start_year=args.start_year,
        end_year=args.end_year,
        target_samples=args.target_samples,
        keywords=keywords_override,
    )

    samples = collector.collect()
    print(f"\nCollected {len(samples)} CVE samples. Data saved to {args.output_dir}")


if __name__ == "__main__":
    main()
