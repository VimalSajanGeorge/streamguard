"""Enhanced GitHub Security Advisory Collector for StreamGuard.

Collects vulnerability data from GitHub Security Advisories GraphQL API
with code extraction from version diffs.
"""

import os
import json
import time
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

import requests
from base_collector import BaseCollector


class GitHubAdvisoryCollectorEnhanced(BaseCollector):
    """Enhanced collector for GitHub Security Advisories with code extraction."""

    # GitHub GraphQL API endpoint
    GRAPHQL_API = "https://api.github.com/graphql"

    # REST API for repository and diff operations
    REST_API = "https://api.github.com"

    # Supported ecosystems
    ECOSYSTEMS = ["PIP", "NPM", "MAVEN", "RUBYGEMS", "GO", "COMPOSER", "NUGET", "CARGO"]

    # Severity levels
    SEVERITIES = ["LOW", "MODERATE", "HIGH", "CRITICAL"]

    # Rate limit configuration
    RATE_LIMIT_POINTS_PER_HOUR = 5000
    RATE_LIMIT_BUFFER = 100  # Keep buffer for safety

    # Package manager registry URLs
    PACKAGE_REGISTRIES = {
        "PIP": "https://pypi.org/pypi/{package}/json",
        "NPM": "https://registry.npmjs.org/{package}",
        "MAVEN": "https://search.maven.org/solrsearch/select?q=g:{group}+AND+a:{artifact}&rows=1&wt=json",
        "RUBYGEMS": "https://rubygems.org/api/v1/gems/{package}.json",
        "CARGO": "https://crates.io/api/v1/crates/{package}",
        "GO": "https://pkg.go.dev/{package}",
        "COMPOSER": "https://packagist.org/packages/{package}.json",
        "NUGET": "https://api.nuget.org/v3-flatcontainer/{package}/index.json"
    }

    def __init__(self, output_dir: str = "data/raw/github", cache_enabled: bool = True, github_token: Optional[str] = None):
        """
        Initialize GitHub Advisory Collector.

        Args:
            output_dir: Directory to save collected data
            cache_enabled: Whether to enable caching
            github_token: Optional GitHub token (if not provided, reads from GITHUB_TOKEN env var)
        """
        super().__init__(output_dir, cache_enabled)

        # Get GitHub token from parameter first, then environment
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        if not self.github_token:
            raise ValueError("GITHUB_TOKEN is required. Pass as parameter or set GITHUB_TOKEN environment variable")

        # Setup headers
        self.headers = {
            "Authorization": f"Bearer {self.github_token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.github.v4+json"
        }

        # Rate limiting tracking
        from datetime import timezone
        self.rate_limit_remaining = self.RATE_LIMIT_POINTS_PER_HOUR
        self.rate_limit_reset_time = datetime.now(timezone.utc) + timedelta(hours=1)

        # Collection statistics
        self.stats = {
            "total_advisories": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "by_ecosystem": defaultdict(int),
            "by_severity": defaultdict(int)
        }

    def collect(self) -> List[Dict]:
        """
        Main collection method - collects all advisories.

        Returns:
            List of collected vulnerability samples
        """
        print("Starting Enhanced GitHub Security Advisory Collection")
        print(f"Target: 10,000 samples across {len(self.ECOSYSTEMS)} ecosystems")
        print(f"Time range: Last 3 years ({datetime.now().year - 2}-{datetime.now().year})")

        return self.collect_all_advisories(target_samples=10000)

    def collect_all_advisories(self, target_samples: int = 10000) -> List[Dict]:
        """
        Collect advisories across all ecosystems and severities.

        Args:
            target_samples: Target number of samples to collect

        Returns:
            List of collected vulnerability samples
        """
        all_samples = []
        samples_per_combination = target_samples // (len(self.ECOSYSTEMS) * len(self.SEVERITIES))

        print(f"\nTarget per ecosystem/severity combination: {samples_per_combination}")
        print(f"DEBUG: Starting collection with {len(self.ECOSYSTEMS)} ecosystems x {len(self.SEVERITIES)} severities")

        for ecosystem in self.ECOSYSTEMS:
            for severity in self.SEVERITIES:
                print(f"\n{'='*60}")
                print(f"Collecting: {ecosystem} / {severity}")
                print(f"{'='*60}")

                try:
                    samples = self.collect_by_ecosystem_severity(
                        ecosystem=ecosystem,
                        severity=severity,
                        max_samples=samples_per_combination
                    )

                    print(f"DEBUG: collect_by_ecosystem_severity returned {len(samples)} samples")
                    all_samples.extend(samples)

                    print(f"Collected {len(samples)} samples for {ecosystem}/{severity}")
                    print(f"Total samples so far: {len(all_samples)}")
                    print(f"DEBUG: all_samples list now has {len(all_samples)} items")

                    # Save intermediate results
                    if len(all_samples) % 1000 == 0:
                        self._save_intermediate_results(all_samples)

                except Exception as e:
                    error_msg = f"Error collecting {ecosystem}/{severity}: {str(e)}"
                    print(f"ERROR: {error_msg}")
                    self.log_error(error_msg, {"ecosystem": ecosystem, "severity": severity})

                # Check rate limits
                self._check_rate_limits()

        # Deduplicate samples
        print(f"\nDEBUG: Before deduplication: {len(all_samples)} samples")
        unique_samples = self.deduplicate_samples(all_samples, key="advisory_id")
        print(f"DEBUG: After deduplication: {len(unique_samples)} samples")
        print(f"\nTotal unique samples collected: {len(unique_samples)}")

        # Save final results
        if len(unique_samples) > 0:
            output_file = self.save_samples(unique_samples, "github_advisories.jsonl")
            print(f"\nSaved to: {output_file}")
        else:
            print("\nWARNING: No samples to save! Check collection process and API responses.")

        # Print statistics
        self._print_statistics()

        return unique_samples

    def collect_by_ecosystem_severity(
        self,
        ecosystem: str,
        severity: str,
        max_samples: int = 1000
    ) -> List[Dict]:
        """
        Collect advisories for specific ecosystem and severity.

        Args:
            ecosystem: Package ecosystem (PIP, NPM, etc.)
            severity: Severity level (LOW, MODERATE, HIGH, CRITICAL)
            max_samples: Maximum samples to collect

        Returns:
            List of vulnerability samples
        """
        samples = []
        cursor = None
        has_next_page = True

        # Calculate date range (last 3 years) - make it timezone-aware
        from datetime import timezone
        start_date = datetime.now(timezone.utc) - timedelta(days=3*365)

        while has_next_page and len(samples) < max_samples:
            # Build and execute GraphQL query
            query_result = self._query_advisories(
                ecosystem=ecosystem,
                severity=severity,
                after_cursor=cursor,
                first=100  # Fetch 100 at a time
            )

            if not query_result:
                print(f"DEBUG: No query result for {ecosystem}/{severity}, breaking")
                break

            print(f"DEBUG: Query result keys: {query_result.keys()}")

            vulnerabilities = query_result.get("data", {}).get("securityVulnerabilities", {})
            nodes = vulnerabilities.get("nodes", [])
            page_info = vulnerabilities.get("pageInfo", {})

            print(f"DEBUG: Found {len(nodes)} vulnerability nodes in this page")

            # Check for errors in response
            if "errors" in query_result:
                print(f"WARNING: GraphQL errors: {query_result['errors']}")

            # Process each vulnerability
            for vuln_node in nodes:
                advisory = vuln_node.get("advisory", {})

                # Check date range
                published_at = advisory.get("publishedAt")
                if published_at:
                    pub_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                    if pub_date < start_date:
                        continue

                # Extract sample from vulnerability node
                sample = self._process_vulnerability_node(vuln_node, ecosystem, severity)
                if sample:
                    samples.append(sample)
                    self.samples_collected += 1
                else:
                    print(f"DEBUG: Failed to process vulnerability node (advisory_id: {advisory.get('ghsaId', 'unknown')})")

                if len(samples) >= max_samples:
                    print(f"DEBUG: Reached max_samples ({max_samples}), breaking")
                    break

            # Update pagination
            has_next_page = page_info.get("hasNextPage", False)
            cursor = page_info.get("endCursor")

            print(f"DEBUG: Pagination - hasNextPage: {has_next_page}, cursor: {cursor is not None}")

            # Rate limiting
            self.rate_limit(requests_per_second=0.5)  # 2 seconds per request

        print(f"DEBUG: Returning {len(samples)} samples from collect_by_ecosystem_severity")
        return samples

    def _query_advisories(
        self,
        ecosystem: str,
        severity: str,
        after_cursor: Optional[str] = None,
        first: int = 100
    ) -> Optional[Dict]:
        """
        Execute GraphQL query for security advisories.

        Args:
            ecosystem: Package ecosystem
            severity: Severity level
            after_cursor: Pagination cursor
            first: Number of results to fetch

        Returns:
            Query result or None if failed
        """
        # Build GraphQL query - use securityVulnerabilities instead of securityAdvisories
        query = """
        query($ecosystem: SecurityAdvisoryEcosystem!, $severity: [SecurityAdvisorySeverity!], $first: Int!, $after: String) {
          securityVulnerabilities(
            ecosystem: $ecosystem,
            severities: $severity,
            first: $first,
            after: $after,
            orderBy: {field: UPDATED_AT, direction: DESC}
          ) {
            nodes {
              advisory {
                ghsaId
                summary
                description
                severity
                publishedAt
                updatedAt
                withdrawnAt
                identifiers {
                  type
                  value
                }
                references {
                  url
                }
              }
              package {
                name
                ecosystem
              }
              vulnerableVersionRange
              firstPatchedVersion {
                identifier
              }
            }
            pageInfo {
              hasNextPage
              endCursor
            }
          }
          rateLimit {
            cost
            remaining
            resetAt
          }
        }
        """

        variables = {
            "ecosystem": ecosystem,
            "severity": [severity],
            "first": first,
            "after": after_cursor
        }

        # Check cache
        cache_key = self.make_cache_key("advisory", ecosystem, severity, after_cursor or "start")
        cached_result = self.load_cache(cache_key)
        if cached_result:
            return cached_result

        try:
            print(f"DEBUG: Querying GraphQL for {ecosystem}/{severity}, cursor: {after_cursor}")
            response = requests.post(
                self.GRAPHQL_API,
                headers=self.headers,
                json={"query": query, "variables": variables},
                timeout=30
            )

            print(f"DEBUG: Response status code: {response.status_code}")
            response.raise_for_status()

            result = response.json()

            # Debug: Check if we have data
            has_data = "data" in result
            has_errors = "errors" in result
            print(f"DEBUG: Response has data: {has_data}, has errors: {has_errors}")

            # Update rate limit info
            if "data" in result and "rateLimit" in result["data"]:
                rate_limit = result["data"]["rateLimit"]
                self.rate_limit_remaining = rate_limit.get("remaining", 0)

            # Save to cache
            self.save_cache(cache_key, result)

            return result

        except Exception as e:
            error_msg = f"GraphQL query failed: {str(e)}"
            self.log_error(error_msg, {"ecosystem": ecosystem, "severity": severity})
            return None

    def _process_vulnerability_node(self, vuln_node: Dict, ecosystem: str, severity: str) -> Optional[Dict]:
        """
        Process a single vulnerability node and extract code samples.

        Args:
            vuln_node: Vulnerability node data from GraphQL
            ecosystem: Package ecosystem
            severity: Severity level

        Returns:
            Processed sample or None
        """
        advisory = vuln_node.get("advisory", {})
        ghsa_id = advisory.get("ghsaId")
        summary = advisory.get("summary", "")
        description = advisory.get("description", "")
        published_at = advisory.get("publishedAt")

        # Get package information directly from vulnerability node
        package_info = vuln_node.get("package", {})
        package_name = package_info.get("name")

        if not package_name:
            return None

        # Get version information from vulnerability node
        vulnerable_range = vuln_node.get("vulnerableVersionRange")
        patched_version_obj = vuln_node.get("firstPatchedVersion")
        patched_version = patched_version_obj.get("identifier") if patched_version_obj else None

        # Extract references
        references = advisory.get("references", [])
        reference_urls = [ref.get("url") for ref in references if ref.get("url")]

        # Try to extract code with diff
        vulnerable_code, fixed_code = self.extract_code_with_diff(
            package_name=package_name,
            ecosystem=ecosystem,
            vulnerable_range=vulnerable_range,
            patched_version=patched_version,
            references=reference_urls
        )

        # Update statistics
        self.stats["total_advisories"] += 1
        self.stats["by_ecosystem"][ecosystem] += 1
        self.stats["by_severity"][severity] += 1

        if vulnerable_code and fixed_code:
            self.stats["successful_extractions"] += 1
        else:
            self.stats["failed_extractions"] += 1

        # Create sample
        sample = {
            "advisory_id": ghsa_id,
            "description": f"{summary}\n\n{description}".strip(),
            "vulnerable_code": vulnerable_code or f"# Vulnerable package: {package_name} {vulnerable_range}",
            "fixed_code": fixed_code or f"# Fixed package: {package_name} {patched_version or 'latest'}",
            "ecosystem": ecosystem,
            "severity": severity,
            "published_at": published_at,
            "source": "github_advisory",
            "metadata": {
                "package_name": package_name,
                "vulnerable_range": vulnerable_range,
                "patched_version": patched_version,
                "references": reference_urls,
                "vulnerability_type": self.extract_vulnerability_type(description)
            }
        }

        return sample

    def _process_advisory_DEPRECATED(self, advisory: Dict, ecosystem: str, severity: str) -> Optional[Dict]:
        """
        Process a single advisory and extract code samples.

        Args:
            advisory: Advisory data from GraphQL
            ecosystem: Package ecosystem
            severity: Severity level

        Returns:
            Processed sample or None
        """
        ghsa_id = advisory.get("ghsaId")
        summary = advisory.get("summary", "")
        description = advisory.get("description", "")
        published_at = advisory.get("publishedAt")

        # Get vulnerabilities
        vulnerabilities = advisory.get("vulnerabilities", {}).get("nodes", [])

        if not vulnerabilities:
            return None

        # Process first vulnerability (primary one)
        vulnerability = vulnerabilities[0]
        package_info = vulnerability.get("package", {})
        package_name = package_info.get("name")

        if not package_name:
            return None

        # Get version information
        vulnerable_range = vulnerability.get("vulnerableVersionRange")
        patched_version_obj = vulnerability.get("firstPatchedVersion")
        patched_version = patched_version_obj.get("identifier") if patched_version_obj else None

        # Extract references
        references = advisory.get("references", [])
        reference_urls = [ref.get("url") for ref in references if ref.get("url")]

        # Try to extract code with diff
        vulnerable_code, fixed_code = self.extract_code_with_diff(
            package_name=package_name,
            ecosystem=ecosystem,
            vulnerable_range=vulnerable_range,
            patched_version=patched_version,
            references=reference_urls
        )

        # Update statistics
        self.stats["total_advisories"] += 1
        self.stats["by_ecosystem"][ecosystem] += 1
        self.stats["by_severity"][severity] += 1

        if vulnerable_code and fixed_code:
            self.stats["successful_extractions"] += 1
        else:
            self.stats["failed_extractions"] += 1

        # Create sample
        sample = {
            "advisory_id": ghsa_id,
            "description": f"{summary}\n\n{description}".strip(),
            "vulnerable_code": vulnerable_code or f"# Vulnerable package: {package_name} {vulnerable_range}",
            "fixed_code": fixed_code or f"# Fixed package: {package_name} {patched_version or 'latest'}",
            "ecosystem": ecosystem,
            "severity": severity,
            "published_at": published_at,
            "source": "github_advisory",
            "metadata": {
                "package_name": package_name,
                "vulnerable_range": vulnerable_range,
                "patched_version": patched_version,
                "references": reference_urls,
                "vulnerability_type": self.extract_vulnerability_type(description)
            }
        }

        return sample

    def extract_code_with_diff(
        self,
        package_name: str,
        ecosystem: str,
        vulnerable_range: Optional[str],
        patched_version: Optional[str],
        references: List[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract vulnerable and fixed code from package version diffs.

        Args:
            package_name: Name of the package
            ecosystem: Package ecosystem
            vulnerable_range: Vulnerable version range
            patched_version: First patched version
            references: List of reference URLs

        Returns:
            Tuple of (vulnerable_code, fixed_code) or (None, None)
        """
        # Try to find repository from references
        repo_url = self._find_repo_from_references(references)

        if not repo_url:
            # Try to find repo from package
            repo_url = self.find_repo_for_package(package_name, ecosystem)

        if not repo_url:
            return None, None

        # Extract owner and repo name
        repo_match = re.search(r'github\.com[/:]([^/]+)/([^/\.]+)', repo_url)
        if not repo_match:
            return None, None

        owner, repo = repo_match.groups()

        # Try to fetch version diff
        vulnerable_code, fixed_code = self.fetch_version_diff(
            owner=owner,
            repo=repo,
            vulnerable_range=vulnerable_range,
            patched_version=patched_version,
            references=references
        )

        return vulnerable_code, fixed_code

    def find_repo_for_package(self, package_name: str, ecosystem: str) -> Optional[str]:
        """
        Find repository URL for a package.

        Args:
            package_name: Package name
            ecosystem: Package ecosystem

        Returns:
            Repository URL or None
        """
        # Check cache
        cache_key = self.make_cache_key("repo", package_name, ecosystem)
        cached_repo = self.load_cache(cache_key)
        if cached_repo:
            return cached_repo.get("repo_url")

        try:
            # Get registry URL template
            registry_template = self.PACKAGE_REGISTRIES.get(ecosystem)
            if not registry_template:
                return None

            # Build registry URL
            if ecosystem == "MAVEN":
                # Maven packages use group:artifact format
                parts = package_name.split(":")
                if len(parts) == 2:
                    registry_url = registry_template.format(group=parts[0], artifact=parts[1])
                else:
                    return None
            else:
                registry_url = registry_template.format(package=package_name)

            # Fetch package metadata
            response = requests.get(registry_url, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Extract repository URL based on ecosystem
            repo_url = None

            if ecosystem == "PIP":
                repo_url = data.get("info", {}).get("project_urls", {}).get("Source")
                if not repo_url:
                    repo_url = data.get("info", {}).get("home_page")

            elif ecosystem == "NPM":
                repo = data.get("repository", {})
                if isinstance(repo, dict):
                    repo_url = repo.get("url")
                elif isinstance(repo, str):
                    repo_url = repo

            elif ecosystem == "CARGO":
                repo_url = data.get("crate", {}).get("repository")

            elif ecosystem == "RUBYGEMS":
                repo_url = data.get("source_code_uri") or data.get("homepage_uri")

            elif ecosystem == "COMPOSER":
                repo_url = data.get("package", {}).get("repository")

            # Clean up repo URL
            if repo_url:
                # Remove git:// prefix and .git suffix
                repo_url = repo_url.replace("git://", "https://").replace("git+", "")
                repo_url = re.sub(r'\.git$', '', repo_url)

                # Only accept GitHub URLs
                if "github.com" not in repo_url:
                    repo_url = None

            # Cache result
            self.save_cache(cache_key, {"repo_url": repo_url})

            return repo_url

        except Exception as e:
            self.log_error(f"Failed to find repo for {package_name}: {str(e)}")
            return None

    def fetch_version_diff(
        self,
        owner: str,
        repo: str,
        vulnerable_range: Optional[str],
        patched_version: Optional[str],
        references: List[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Fetch code diff between vulnerable and fixed versions.

        Args:
            owner: Repository owner
            repo: Repository name
            vulnerable_range: Vulnerable version range
            patched_version: First patched version
            references: Reference URLs (may contain commit URLs)

        Returns:
            Tuple of (vulnerable_code, fixed_code)
        """
        # Try to extract commit SHA from references
        commit_sha = self._extract_commit_from_references(references)

        if commit_sha:
            return self._fetch_commit_diff(owner, repo, commit_sha)

        # Try to find tags/releases
        if patched_version:
            return self._fetch_tag_diff(owner, repo, vulnerable_range, patched_version)

        return None, None

    def _fetch_commit_diff(self, owner: str, repo: str, commit_sha: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Fetch diff for a specific commit.

        Args:
            owner: Repository owner
            repo: Repository name
            commit_sha: Commit SHA

        Returns:
            Tuple of (before_code, after_code)
        """
        # Check cache
        cache_key = self.make_cache_key("commit_diff", owner, repo, commit_sha)
        cached_diff = self.load_cache(cache_key)
        if cached_diff:
            return cached_diff.get("before"), cached_diff.get("after")

        try:
            # Fetch commit
            url = f"{self.REST_API}/repos/{owner}/{repo}/commits/{commit_sha}"
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            commit_data = response.json()
            files = commit_data.get("files", [])

            # Find relevant code changes (prefer main code files, not tests)
            for file_info in files:
                filename = file_info.get("filename", "")

                # Skip non-code files
                if not self._is_code_file(filename):
                    continue

                # Skip test files
                if "test" in filename.lower() or "spec" in filename.lower():
                    continue

                patch = file_info.get("patch")
                if not patch:
                    continue

                # Parse patch to extract before/after code
                before_code, after_code = self._parse_patch(patch)

                if before_code and after_code:
                    # Validate code quality
                    if self.validate_code(before_code) and self.validate_code(after_code):
                        result = {"before": before_code, "after": after_code}
                        self.save_cache(cache_key, result)
                        return before_code, after_code

            return None, None

        except Exception as e:
            self.log_error(f"Failed to fetch commit diff: {str(e)}", {
                "owner": owner, "repo": repo, "commit": commit_sha
            })
            return None, None

    def _fetch_tag_diff(
        self,
        owner: str,
        repo: str,
        vulnerable_range: Optional[str],
        patched_version: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Fetch diff between version tags.

        Args:
            owner: Repository owner
            repo: Repository name
            vulnerable_range: Vulnerable version range
            patched_version: Patched version

        Returns:
            Tuple of (vulnerable_code, fixed_code)
        """
        try:
            # Get tags
            url = f"{self.REST_API}/repos/{owner}/{repo}/tags"
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            tags = response.json()

            # Find patched version tag
            patched_tag = None
            vulnerable_tag = None

            for tag in tags:
                tag_name = tag.get("name", "")

                # Match patched version
                if patched_version in tag_name:
                    patched_tag = tag

                # Try to find previous version (simple heuristic)
                if vulnerable_range and self._version_in_range(tag_name, vulnerable_range):
                    vulnerable_tag = tag

            if not patched_tag:
                return None, None

            # Get comparison
            if vulnerable_tag:
                base = vulnerable_tag.get("commit", {}).get("sha")
                head = patched_tag.get("commit", {}).get("sha")

                if base and head:
                    return self._fetch_comparison_diff(owner, repo, base, head)

            return None, None

        except Exception as e:
            self.log_error(f"Failed to fetch tag diff: {str(e)}")
            return None, None

    def _fetch_comparison_diff(self, owner: str, repo: str, base: str, head: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Fetch comparison diff between two commits.

        Args:
            owner: Repository owner
            repo: Repository name
            base: Base commit SHA
            head: Head commit SHA

        Returns:
            Tuple of (base_code, head_code)
        """
        try:
            url = f"{self.REST_API}/repos/{owner}/{repo}/compare/{base}...{head}"
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            comparison = response.json()
            files = comparison.get("files", [])

            for file_info in files[:5]:  # Check first 5 files
                filename = file_info.get("filename", "")

                if not self._is_code_file(filename):
                    continue

                if "test" in filename.lower():
                    continue

                patch = file_info.get("patch")
                if patch:
                    before, after = self._parse_patch(patch)
                    if before and after:
                        if self.validate_code(before) and self.validate_code(after):
                            return before, after

            return None, None

        except Exception as e:
            self.log_error(f"Failed to fetch comparison: {str(e)}")
            return None, None

    def _parse_patch(self, patch: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse unified diff patch to extract before/after code.

        Args:
            patch: Unified diff patch string

        Returns:
            Tuple of (before_code, after_code)
        """
        before_lines = []
        after_lines = []

        for line in patch.split('\n'):
            if line.startswith('@@'):
                continue
            elif line.startswith('-') and not line.startswith('---'):
                before_lines.append(line[1:])
            elif line.startswith('+') and not line.startswith('+++'):
                after_lines.append(line[1:])
            elif not line.startswith('\\'):
                # Context line - add to both
                before_lines.append(line[1:] if line.startswith(' ') else line)
                after_lines.append(line[1:] if line.startswith(' ') else line)

        before_code = '\n'.join(before_lines).strip()
        after_code = '\n'.join(after_lines).strip()

        return before_code if before_code else None, after_code if after_code else None

    def _find_repo_from_references(self, references: List[str]) -> Optional[str]:
        """Find GitHub repository URL from reference URLs."""
        for url in references:
            if "github.com" in url and "/commit/" not in url and "/pull/" not in url:
                # Clean URL to get repo base
                match = re.search(r'(https?://github\.com/[^/]+/[^/]+)', url)
                if match:
                    return match.group(1)
        return None

    def _extract_commit_from_references(self, references: List[str]) -> Optional[str]:
        """Extract commit SHA from reference URLs."""
        for url in references:
            # Match commit URLs
            match = re.search(r'/commit/([a-f0-9]{40})', url)
            if match:
                return match.group(1)

            # Match PR URLs (we can fetch the PR to get commits)
            match = re.search(r'/pull/(\d+)', url)
            if match:
                # We could fetch PR details here, but skipping for simplicity
                pass

        return None

    def _is_code_file(self, filename: str) -> bool:
        """Check if file is a code file."""
        code_extensions = [
            '.py', '.js', '.ts', '.java', '.rb', '.go',
            '.php', '.cs', '.cpp', '.c', '.rs', '.jsx', '.tsx'
        ]
        return any(filename.endswith(ext) for ext in code_extensions)

    def _version_in_range(self, version: str, version_range: str) -> bool:
        """
        Check if version is in vulnerable range.

        This is a simplified version - proper implementation would need
        version parsing library for each ecosystem.
        """
        # Extract version numbers
        version_match = re.search(r'(\d+\.[\d\.]+)', version)
        if not version_match:
            return False

        return version_match.group(1) in version_range

    def _check_rate_limits(self):
        """Check and handle rate limits."""
        from datetime import timezone
        if self.rate_limit_remaining < self.RATE_LIMIT_BUFFER:
            wait_time = (self.rate_limit_reset_time - datetime.now(timezone.utc)).total_seconds()
            if wait_time > 0:
                print(f"\nRate limit approaching. Waiting {wait_time:.0f} seconds...")
                time.sleep(wait_time + 60)  # Add 1 minute buffer
                self.rate_limit_remaining = self.RATE_LIMIT_POINTS_PER_HOUR
                self.rate_limit_reset_time = datetime.now(timezone.utc) + timedelta(hours=1)

    def _save_intermediate_results(self, samples: List[Dict]):
        """Save intermediate results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"github_advisories_intermediate_{timestamp}.jsonl"
        self.save_samples(samples, filename)
        print(f"\nSaved intermediate results: {filename}")

    def _print_statistics(self):
        """Print collection statistics."""
        print("\n" + "="*60)
        print("COLLECTION STATISTICS")
        print("="*60)
        print(f"Total Advisories Processed: {self.stats['total_advisories']}")
        print(f"Successful Code Extractions: {self.stats['successful_extractions']}")
        print(f"Failed Code Extractions: {self.stats['failed_extractions']}")

        print("\nBy Ecosystem:")
        for ecosystem, count in sorted(self.stats['by_ecosystem'].items()):
            print(f"  {ecosystem}: {count}")

        print("\nBy Severity:")
        for severity, count in sorted(self.stats['by_severity'].items()):
            print(f"  {severity}: {count}")

        print("\nErrors:", len(self.errors))
        if self.errors:
            print("\nLast 5 errors:")
            for error in self.errors[-5:]:
                print(f"  - {error['error']}")


def main():
    """Main entry point for the collector."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced GitHub Security Advisory Collector"
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/github",
        help="Output directory for collected data"
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=10000,
        help="Target number of samples to collect"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching"
    )

    args = parser.parse_args()

    # Initialize collector
    collector = GitHubAdvisoryCollectorEnhanced(
        output_dir=args.output_dir,
        cache_enabled=not args.no_cache
    )

    # Collect data
    try:
        samples = collector.collect_all_advisories(target_samples=args.target_samples)
        print(f"\nCollection complete! Collected {len(samples)} samples.")

    except KeyboardInterrupt:
        print("\n\nCollection interrupted by user.")
        print(f"Collected {collector.samples_collected} samples before interruption.")

    except Exception as e:
        print(f"\n\nCollection failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # Print final statistics
        stats = collector.get_stats()
        print(f"\nFinal statistics: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
