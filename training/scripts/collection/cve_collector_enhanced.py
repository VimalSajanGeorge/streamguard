#!/usr/bin/env python3
"""
training/scripts/collection/cve_collector_enhanced.py

Two-phase CVE/NVD collector for StreamGuard.

Phase 1 (NVDIndexBuilder):
    Queries NVD API v2.0 for each of 12 target CWEs.
    Extracts CVE IDs with GitHub commit URLs from references.
    Output: data/raw/cve/cve_index.jsonl (18K-25K entries)

Phase 2 (GitHubDiffFetcher):
    Reads cve_index.jsonl, fetches commit diffs from GitHub API.
    Extracts before/after C code from modified .c/.h files.
    Output: data/raw/cve/cve_samples.jsonl (3K-5K pairs)

Production fixes incorporated:
    PI-01: 503 handling with sleep + jitter
    PI-03: NVD v2.0 endpoint (not v1.0)
    PI-06: SHA normalization (short SHA -> full SHA)
    PI-08: Jitter + GitHub 403 abuse detection

Source: docs/New Docs/DATA_PIPELINE.md §Collector 3: CVE/NVD
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import uuid
from collections import Counter
from pathlib import Path

from loguru import logger

try:
    from .base_collector import BaseCollector
    from .schema import VALID_CWE, make_sample_id, make_timestamp, validate_sample
except ImportError:
    from base_collector import BaseCollector
    from schema import VALID_CWE, make_sample_id, make_timestamp, validate_sample


# ── Constants ─────────────────────────────────────────────────────────────

NVD_API_V2_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

TARGET_CWES = sorted(VALID_CWE)  # All 12 target CWEs

# NVD rate limits
NVD_RATE_WITH_KEY = (50, 30)    # 50 requests per 30 seconds
NVD_RATE_WITHOUT_KEY = (5, 30)  # 5 requests per 30 seconds

NVD_MAX_RESULTS_PER_PAGE = 2000

# GitHub diff filter thresholds
MIN_CHANGED_LINES = 5
MAX_CHANGED_LINES = 150
VALID_DIFF_EXTENSIONS = {".c", ".h"}


# ── GitHub Token Rotator ─────────────────────────────────────────────────

class GitHubTokenRotator:
    """
    Round-robin through multiple GitHub tokens to maximize API throughput.

    Each token gets 5,000 requests/hour. With N tokens, throughput is N * 5,000/hr.
    Tracks per-token rate limits via X-RateLimit-Remaining/Reset headers.
    When a token is exhausted, rotates to next. When all exhausted, sleeps
    until the earliest reset time.
    """

    def __init__(self, tokens: list[str]):
        if not tokens:
            raise ValueError("At least one GitHub token is required for Phase 2")
        self.tokens = tokens
        self._idx = 0
        # Per-token state: {token_idx: {remaining: int, reset: float}}
        self._limits: dict[int, dict] = {}
        logger.info(f"GitHub token rotator initialized with {len(tokens)} token(s)")

    @property
    def current_token(self) -> str:
        return self.tokens[self._idx]

    @property
    def auth_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.current_token}",
            "Accept": "application/vnd.github.v3+json",
        }

    def update_limits(self, response_headers: dict) -> None:
        """Update rate limit state from GitHub response headers."""
        remaining = response_headers.get("X-RateLimit-Remaining")
        reset = response_headers.get("X-RateLimit-Reset")
        if remaining is not None and reset is not None:
            self._limits[self._idx] = {
                "remaining": int(remaining),
                "reset": float(reset),
            }

    def rotate_if_needed(self) -> None:
        """
        Rotate to a token with remaining capacity.
        If all tokens are near exhaustion (< 50 remaining), sleep until earliest reset.
        """
        # Check if current token still has capacity
        current_limit = self._limits.get(self._idx, {})
        if current_limit.get("remaining", 1000) > 50:
            return  # Current token still good

        # Try to find a token with remaining capacity
        for i in range(len(self.tokens)):
            candidate = (self._idx + 1 + i) % len(self.tokens)
            limit = self._limits.get(candidate, {})
            if limit.get("remaining", 1000) > 50:
                old_idx = self._idx
                self._idx = candidate
                logger.info(
                    f"Rotated from token #{old_idx} to #{self._idx} "
                    f"(remaining: {limit.get('remaining', '?')})"
                )
                return

        # All tokens exhausted — sleep until earliest reset
        resets = [
            info.get("reset", time.time() + 3600)
            for info in self._limits.values()
        ]
        if resets:
            earliest_reset = min(resets)
            wait_secs = max(earliest_reset - time.time() + 5, 1)
            logger.warning(
                f"All {len(self.tokens)} tokens exhausted. "
                f"Sleeping {wait_secs:.0f}s until reset."
            )
            time.sleep(wait_secs)
            # Reset limit tracking after sleep
            self._limits.clear()
            # Rotate to next token
            self._idx = (self._idx + 1) % len(self.tokens)


# ══════════════════════════════════════════════════════════════════════════
# Phase 1: NVD Index Builder
# ══════════════════════════════════════════════════════════════════════════

class NVDIndexBuilder(BaseCollector):
    """
    Phase 1: Query NVD API v2.0 for CVEs with GitHub commit URLs.
    Builds an index file — does NOT save training samples.
    """

    def __init__(
        self,
        output_dir: str,
        checkpoint_dir: str | None = None,
        nvd_api_key: str | None = None,
        dry_run: bool = False,
        max_samples: int = 0,
    ):
        super().__init__(
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            source_name="cve_index",
        )
        self.nvd_api_key = nvd_api_key or os.environ.get("NVD_API_KEY", "")
        self.dry_run = dry_run
        self.max_samples = max_samples

        self._has_key = bool(self.nvd_api_key)
        rate_limit, rate_window = (
            NVD_RATE_WITH_KEY if self._has_key else NVD_RATE_WITHOUT_KEY
        )
        self._rate_limit = rate_limit
        self._rate_window = rate_window
        self._request_times: list[float] = []

        # Counters
        self.cves_found = 0
        self.cves_with_github = 0
        self.cves_without_github = 0
        self.index_entries_saved = 0
        self.cwe_counts: Counter = Counter()

        if not self._has_key:
            logger.warning(
                "NVD_API_KEY not set. Rate limit: 5 req/30s (vs 50 req/30s with key). "
                "Phase 1 will take ~10x longer."
            )

    # ── NVD Rate Limiting ─────────────────────────────────────────────

    def _nvd_rate_wait(self) -> None:
        """Proactive NVD rate limiting. Track requests in sliding window."""
        now = time.time()
        # Remove requests outside the window
        self._request_times = [
            t for t in self._request_times
            if now - t < self._rate_window
        ]
        if len(self._request_times) >= self._rate_limit:
            # Wait until oldest request falls outside window
            oldest = self._request_times[0]
            wait_secs = self._rate_window - (now - oldest) + random.uniform(0.5, 2.0)
            if wait_secs > 0:
                logger.debug(f"NVD rate limit: sleeping {wait_secs:.1f}s")
                time.sleep(wait_secs)
        self._request_times.append(time.time())

    # ── NVD API Query ─────────────────────────────────────────────────

    def _query_nvd(self, cwe: str, start_index: int = 0) -> dict | None:
        """
        Query NVD API v2.0 for a specific CWE.
        Returns parsed JSON response or None on failure.
        """
        self._nvd_rate_wait()

        headers = {}
        if self.nvd_api_key:
            headers["apiKey"] = self.nvd_api_key

        params = {
            "cweId": cwe,
            "resultsPerPage": NVD_MAX_RESULTS_PER_PAGE,
            "startIndex": start_index,
        }

        try:
            resp = self.safe_get(NVD_API_V2_URL, headers=headers, params=params, timeout=120)
            if resp.status_code != 200:
                logger.warning(
                    f"NVD API returned {resp.status_code} for {cwe} "
                    f"startIndex={start_index}"
                )
                return None
            return resp.json()
        except Exception as e:
            logger.error(f"NVD query failed for {cwe}: {e}")
            return None

    # ── Reference Parsing ─────────────────────────────────────────────

    @staticmethod
    def extract_github_commit_url(references: list[dict]) -> str | None:
        """
        Extract first GitHub commit URL from NVD references list.
        Must contain 'github.com' AND '/commit/' (not /pull/ or /issues/).
        """
        for ref in references:
            url = ref.get("url", "")
            if "github.com" in url and "/commit/" in url:
                # Exclude pull requests and issues URLs that might contain /commit/
                if "/pull/" not in url and "/issues/" not in url:
                    return url
        return None

    @staticmethod
    def _extract_cvss(cve_data: dict) -> float:
        """Extract CVSS base score from NVD metrics. Returns 0.0 if not found."""
        metrics = cve_data.get("metrics", {})
        # Try CVSS v3.1 first, then v3.0, then v2.0
        for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            metric_list = metrics.get(key, [])
            if metric_list:
                cvss = metric_list[0].get("cvssData", {})
                score = cvss.get("baseScore", 0.0)
                if score:
                    return float(score)
        return 0.0

    @staticmethod
    def extract_cwe_from_weaknesses(weaknesses: list[dict]) -> str | None:
        """Extract CWE ID from NVD weaknesses field."""
        for w in weaknesses:
            descs = w.get("description", [])
            for d in descs:
                value = d.get("value", "")
                if value.startswith("CWE-") and value in VALID_CWE:
                    return value
        return None

    # ── Index Entry Saving ────────────────────────────────────────────

    def _save_index_entry(self, entry: dict) -> None:
        """Append an index entry to cve_index.jsonl."""
        if self.dry_run:
            self.index_entries_saved += 1
            return

        index_path = self.output_dir / "cve_index.jsonl"
        with open(index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.index_entries_saved += 1

    # ── Main Collection ───────────────────────────────────────────────

    def collect(self) -> dict:
        """
        Query NVD for all 12 CWEs. Build index of CVEs with GitHub commit URLs.
        Returns summary dict.
        """
        # R-06: Hard assert v2.0 URL — never silently use deprecated v1.0
        assert "/2.0" in NVD_API_V2_URL, (
            f"FATAL: NVD URL must be v2.0, got {NVD_API_V2_URL}"
        )

        if self.dry_run:
            logger.info("DRY RUN MODE — will process but not write index")

        checkpoint = self.load_checkpoint()
        cwe_progress = checkpoint.get("cwe_progress", {})

        # Track seen CVE IDs to deduplicate across CWEs
        seen_cves: set[str] = set()

        # Load existing index to avoid duplicates on resume
        index_path = self.output_dir / "cve_index.jsonl"
        if index_path.exists() and not self.dry_run:
            with open(index_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            seen_cves.add(entry.get("cve_id", ""))
                        except json.JSONDecodeError:
                            pass
            if seen_cves:
                logger.info(f"Loaded {len(seen_cves)} existing index entries for dedup")

        for cwe in TARGET_CWES:
            if self.max_samples > 0 and self.index_entries_saved >= self.max_samples:
                logger.info(f"Reached max_samples ({self.max_samples}). Stopping.")
                break

            # Check if this CWE is already completed
            if cwe_progress.get(cwe, {}).get("completed", False):
                logger.info(f"Skipping {cwe} — already completed in checkpoint")
                continue

            start_index = cwe_progress.get(cwe, {}).get("last_start_index", 0)
            self._process_cwe(cwe, start_index, seen_cves, cwe_progress)

        # Final checkpoint
        self.save_checkpoint({
            "cwe_progress": cwe_progress,
            "index_entries_saved": self.index_entries_saved,
            "completed": True,
        })

        return self._build_summary()

    def _process_cwe(
        self,
        cwe: str,
        start_index: int,
        seen_cves: set[str],
        cwe_progress: dict,
    ) -> None:
        """Process all pages for a single CWE."""
        logger.info(f"Querying NVD for {cwe} (startIndex={start_index})...")

        while True:
            if self.max_samples > 0 and self.index_entries_saved >= self.max_samples:
                break

            data = self._query_nvd(cwe, start_index)
            if data is None:
                logger.warning(f"NVD query returned None for {cwe} at {start_index}")
                break

            total_results = data.get("totalResults", 0)
            vulnerabilities = data.get("vulnerabilities", [])

            if not vulnerabilities:
                break

            for vuln_wrapper in vulnerabilities:
                if self.max_samples > 0 and self.index_entries_saved >= self.max_samples:
                    break

                cve_data = vuln_wrapper.get("cve", {})
                cve_id = cve_data.get("id", "")
                self.cves_found += 1

                if not cve_id or cve_id in seen_cves:
                    continue

                # Extract GitHub commit URL from references
                references = cve_data.get("references", [])
                github_url = self.extract_github_commit_url(references)

                if not github_url:
                    self.cves_without_github += 1
                    continue

                # Extract CWE from weaknesses (may differ from query CWE)
                # R-04: Track CWE confidence tier
                weaknesses = cve_data.get("weaknesses", [])
                extracted_cwe = self.extract_cwe_from_weaknesses(weaknesses)
                if extracted_cwe:
                    cwe_confidence = "tier1_direct"  # CWE from NVD source
                else:
                    extracted_cwe = cwe  # Fallback to query CWE
                    cwe_confidence = "tier3_inferred"  # CWE from query param

                # Extract published date
                published = cve_data.get("published", "")

                # Extract CVSS severity score (for L_severity head)
                severity_score = self._extract_cvss(cve_data)

                entry = {
                    "cve_id": cve_id,
                    "github_commit_url": github_url,
                    "cwe": extracted_cwe,
                    "cwe_confidence": cwe_confidence,
                    "severity_score": severity_score,
                    "nvd_published_date": published,
                }

                self._save_index_entry(entry)
                seen_cves.add(cve_id)
                self.cves_with_github += 1
                self.cwe_counts[extracted_cwe] += 1

            # Pagination
            start_index += NVD_MAX_RESULTS_PER_PAGE

            # Checkpoint after each page
            cwe_progress[cwe] = {"last_start_index": start_index}
            self.save_checkpoint({
                "cwe_progress": cwe_progress,
                "index_entries_saved": self.index_entries_saved,
            })

            if start_index >= total_results:
                cwe_progress[cwe] = {"completed": True}
                logger.info(
                    f"Completed {cwe}: {total_results} total CVEs, "
                    f"{self.cves_with_github} with GitHub URLs so far"
                )
                break

            self.jitter_sleep(0.3)

    def _build_summary(self) -> dict:
        return {
            "cves_found": self.cves_found,
            "cves_with_github": self.cves_with_github,
            "cves_without_github": self.cves_without_github,
            "index_entries_saved": self.index_entries_saved,
            "cwe_distribution": dict(self.cwe_counts),
        }

    def print_summary(self) -> None:
        summary = self._build_summary()
        print("\n" + "=" * 60)
        print("NVD INDEX BUILDER SUMMARY (Phase 1)")
        print("=" * 60)
        print(f"CVEs found:           {summary['cves_found']}")
        print(f"With GitHub commit:   {summary['cves_with_github']}")
        print(f"Without GitHub:       {summary['cves_without_github']}")
        print(f"Index entries saved:  {summary['index_entries_saved']}")
        if summary["cwe_distribution"]:
            print("\nCWE Distribution:")
            for cwe, count in sorted(
                summary["cwe_distribution"].items(), key=lambda x: -x[1]
            ):
                print(f"  {cwe}: {count}")
        print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════
# Phase 2: GitHub Diff Fetcher
# ══════════════════════════════════════════════════════════════════════════

class GitHubDiffFetcher(BaseCollector):
    """
    Phase 2: Fetch commit diffs from GitHub for CVE index entries.
    Extracts before/after C code and saves as paired training samples.
    Supports multi-token rotation for faster collection.
    """

    GITHUB_API_BASE = "https://api.github.com"

    def __init__(
        self,
        output_dir: str,
        checkpoint_dir: str | None = None,
        github_tokens: list[str] | None = None,
        dry_run: bool = False,
        max_samples: int = 0,
        index_path: str | None = None,
    ):
        super().__init__(
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            source_name="cve",
        )
        # Resolve tokens from args or env
        # Priority: args > GITHUB_TOKENS env > GITHUB_TOKEN env
        # All sources support comma-separated values for multi-token rotation
        tokens = github_tokens or []
        if not tokens:
            multi = os.environ.get("GITHUB_TOKENS", "")
            if multi:
                tokens = [t.strip() for t in multi.split(",") if t.strip()]
        if not tokens:
            single = os.environ.get("GITHUB_TOKEN", "")
            if single:
                tokens = [t.strip() for t in single.split(",") if t.strip()]

        if not tokens:
            raise ValueError(
                "No GitHub token provided. Set GITHUB_TOKEN or GITHUB_TOKENS env var, "
                "or use --github-token / --github-tokens CLI flag."
            )

        self.rotator = GitHubTokenRotator(tokens)
        self.dry_run = dry_run
        self.max_samples = max_samples

        # Index file path
        self._index_path = Path(index_path) if index_path else self.output_dir / "cve_index.jsonl"

        # Counters
        self.commits_processed = 0
        self.commits_skipped = 0
        self.commits_stale_404 = 0
        self.files_qualifying = 0
        self.files_rejected_ext = 0
        self.files_rejected_status = 0
        self.files_rejected_size = 0
        self.pairs_saved = 0
        self.cwe_counts: Counter = Counter()
        self.label_counts: Counter = Counter()

    # ── URL Parsing ───────────────────────────────────────────────────

    @staticmethod
    def parse_github_commit_url(url: str) -> tuple[str, str, str] | None:
        """
        Parse owner, repo, sha from a GitHub commit URL.
        Examples:
            https://github.com/owner/repo/commit/abc123
            https://github.com/owner/repo/commit/abc123def456...
        Returns (owner, repo, sha) or None if unparseable.
        """
        m = re.match(
            r"https?://github\.com/([^/]+)/([^/]+)/commit/([0-9a-fA-F]+)",
            url,
        )
        if m:
            return m.group(1), m.group(2), m.group(3)
        return None

    # ── SHA Normalization (PI-06) ─────────────────────────────────────

    def _resolve_full_sha(self, owner: str, repo: str, short_sha: str) -> str | None:
        """
        If SHA is less than 40 chars, resolve to full SHA via GitHub API.
        Returns full 40-char SHA or None on failure.
        """
        if len(short_sha) >= 40:
            return short_sha

        self.rotator.rotate_if_needed()
        url = f"{self.GITHUB_API_BASE}/repos/{owner}/{repo}/commits/{short_sha}"

        try:
            resp = self.safe_get(url, headers=self.rotator.auth_headers)
            self.rotator.update_limits(dict(resp.headers))

            if resp.status_code == 200:
                return resp.json().get("sha", short_sha)
            return None
        except Exception as e:
            logger.debug(f"SHA resolution failed for {short_sha}: {e}")
            return None

    # ── Commit Fetching ───────────────────────────────────────────────

    def _fetch_commit(self, owner: str, repo: str, sha: str) -> dict | None:
        """
        Fetch a commit's details (including file diffs) from GitHub API.
        Returns parsed JSON or None.
        """
        self.rotator.rotate_if_needed()
        url = f"{self.GITHUB_API_BASE}/repos/{owner}/{repo}/commits/{sha}"

        try:
            resp = self.safe_get(url, headers=self.rotator.auth_headers)
            self.rotator.update_limits(dict(resp.headers))

            if resp.status_code == 404:
                return None  # Stale commit URL
            if resp.status_code != 200:
                logger.warning(f"GitHub returned {resp.status_code} for {url}")
                return None
            return resp.json()
        except Exception as e:
            logger.error(f"GitHub fetch failed for {owner}/{repo}/{sha}: {e}")
            return None

    # ── Diff Parsing ──────────────────────────────────────────────────

    @staticmethod
    def extract_before_after(patch: str) -> tuple[str, str]:
        """
        Extract before_code and after_code from a unified diff patch.

        before_code = lines starting with '-' (minus additions) + context lines
        after_code  = lines starting with '+' (minus deletions) + context lines

        Strips the leading +/- character. Skips @@ hunk headers.
        """
        before_lines = []
        after_lines = []

        for line in patch.splitlines():
            if line.startswith("@@"):
                # Hunk header — add to both as context marker
                continue
            elif line.startswith("-"):
                before_lines.append(line[1:])  # Strip leading -
            elif line.startswith("+"):
                after_lines.append(line[1:])   # Strip leading +
            else:
                # Context line (no prefix or space prefix)
                content = line[1:] if line.startswith(" ") else line
                before_lines.append(content)
                after_lines.append(content)

        return "\n".join(before_lines), "\n".join(after_lines)

    @staticmethod
    def count_changed_lines(patch: str) -> int:
        """Count number of changed lines (additions + deletions) in a patch."""
        count = 0
        for line in patch.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                count += 1
            elif line.startswith("-") and not line.startswith("---"):
                count += 1
        return count

    @staticmethod
    def is_qualifying_file(file_info: dict) -> tuple[bool, str]:
        """
        Check if a diff file qualifies for extraction.
        Returns (qualifies, rejection_reason).
        """
        filename = file_info.get("filename", "")
        status = file_info.get("status", "")
        patch = file_info.get("patch", "")

        # Extension check
        ext = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""
        if ext.lower() not in VALID_DIFF_EXTENSIONS:
            return False, "extension"

        # Status check — only modified files
        if status != "modified":
            return False, "status"

        # Changed lines check
        if not patch:
            return False, "no_patch"

        changed = GitHubDiffFetcher.count_changed_lines(patch)
        if changed < MIN_CHANGED_LINES or changed > MAX_CHANGED_LINES:
            return False, "size"

        return True, ""

    # ── Stale URL Logging ─────────────────────────────────────────────

    def _save_stale(self, entry: dict) -> None:
        """Log stale (404) commit URL to cve_stale.jsonl."""
        stale_path = self.output_dir / "cve_stale.jsonl"
        with open(stale_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── Main Collection ───────────────────────────────────────────────

    def collect(self) -> dict:
        """
        Read index, fetch diffs, extract before/after code, save pairs.
        Returns summary dict.
        """
        if self.dry_run:
            logger.info("DRY RUN MODE — will process but not save samples")

        if not self._index_path.exists():
            raise FileNotFoundError(
                f"CVE index not found at {self._index_path}. Run Phase 1 first."
            )

        # Load index entries
        entries = []
        with open(self._index_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        logger.info(f"Loaded {len(entries)} index entries from {self._index_path}")

        # Load checkpoint — set of already-processed commit SHAs
        checkpoint = self.load_checkpoint()
        done_shas: set[str] = set(checkpoint.get("done_shas", []))
        if done_shas:
            logger.info(f"Checkpoint: {len(done_shas)} commits already processed")

        for idx, entry in enumerate(entries):
            if self.max_samples > 0 and self._samples_saved >= self.max_samples:
                logger.info(f"Reached max_samples ({self.max_samples}). Stopping.")
                break

            self._process_index_entry(entry, done_shas)

            # Checkpoint every 100 entries
            if (idx + 1) % 100 == 0:
                self.save_checkpoint({
                    "done_shas": list(done_shas),
                    "samples_saved": self._samples_saved,
                })

            # Jitter between requests
            self.jitter_sleep(0.3)

        # Final checkpoint
        self.save_checkpoint({
            "done_shas": list(done_shas),
            "samples_saved": self._samples_saved,
            "completed": True,
        })

        # Finalize orphan pairs
        if not self.dry_run:
            orphans = self.finalize_pairs()
            if orphans:
                logger.info(f"Cleared {orphans} orphan pair_ids")

        return self._build_summary()

    def _process_index_entry(self, entry: dict, done_shas: set[str]) -> None:
        """Process a single index entry: fetch commit, extract diffs, save pairs."""
        github_url = entry.get("github_commit_url", "")
        cve_id = entry.get("cve_id", "")
        cwe = entry.get("cwe", "")
        severity_score = entry.get("severity_score", 0.0)
        cwe_confidence = entry.get("cwe_confidence", "tier1_direct")

        parsed = self.parse_github_commit_url(github_url)
        if not parsed:
            logger.debug(f"Unparseable GitHub URL: {github_url}")
            self.commits_skipped += 1
            return

        owner, repo, sha = parsed

        # SHA normalization (PI-06)
        if len(sha) < 40:
            full_sha = self._resolve_full_sha(owner, repo, sha)
            if not full_sha:
                self.commits_skipped += 1
                return
            sha = full_sha

        # Skip already-processed commits
        if sha in done_shas:
            return

        # Fetch commit
        commit_data = self._fetch_commit(owner, repo, sha)
        if commit_data is None:
            self.commits_stale_404 += 1
            if not self.dry_run:
                self._save_stale(entry)
            done_shas.add(sha)
            return

        self.commits_processed += 1
        done_shas.add(sha)

        # Process each file in the commit
        files = commit_data.get("files", [])
        for file_info in files:
            if self.max_samples > 0 and self._samples_saved >= self.max_samples:
                break

            qualifies, reason = self.is_qualifying_file(file_info)
            if not qualifies:
                if reason == "extension":
                    self.files_rejected_ext += 1
                elif reason == "status":
                    self.files_rejected_status += 1
                elif reason == "size":
                    self.files_rejected_size += 1
                continue

            self.files_qualifying += 1
            patch = file_info.get("patch", "")
            filename = file_info.get("filename", "")

            before_code, after_code = self.extract_before_after(patch)

            # Create paired samples
            timestamp = make_timestamp()
            pair_id = str(uuid.uuid4())

            vuln_sample = {
                "id": make_sample_id(),
                "source": "cve",
                "code": before_code,
                "label": 1,
                "cwe": cwe,
                "language": "c",
                "collected_at": timestamp,
                "pair_id": pair_id,
                "commit_sha": sha,
                "cve_id": cve_id,
                "severity_score": severity_score,
                "file_path": filename,
                "cfa_origin": "native",
                "metadata": {
                    "github_url": github_url,
                    "owner": owner,
                    "repo": repo,
                    "cwe_confidence": cwe_confidence,
                },
            }

            fixed_sample = {
                "id": make_sample_id(),
                "source": "cve",
                "code": after_code,
                "label": 0,
                "cwe": cwe,
                "language": "c",
                "collected_at": timestamp,
                "pair_id": pair_id,
                "commit_sha": sha,
                "cve_id": cve_id,
                "severity_score": severity_score,
                "file_path": filename,
                "cfa_origin": "native",
                "metadata": {
                    "github_url": github_url,
                    "owner": owner,
                    "repo": repo,
                    "cwe_confidence": cwe_confidence,
                },
            }

            # Validate both samples
            vuln_valid, vuln_errors = validate_sample(vuln_sample)
            fixed_valid, fixed_errors = validate_sample(fixed_sample)

            if not vuln_valid or not fixed_valid:
                self._samples_failed += 2
                if not self.dry_run:
                    if not vuln_valid:
                        self.save_failed_item(vuln_sample, reason=str(vuln_errors))
                    if not fixed_valid:
                        self.save_failed_item(fixed_sample, reason=str(fixed_errors))
                continue

            # Save pair
            if self.dry_run:
                self._samples_saved += 2
                self.pairs_saved += 1
                self.cwe_counts[cwe] += 2
                self.label_counts[1] += 1
                self.label_counts[0] += 1
            else:
                saved_vuln = self.save_sample(vuln_sample)
                saved_fixed = self.save_sample(fixed_sample)
                if saved_vuln and saved_fixed:
                    self.pairs_saved += 1
                if saved_vuln:
                    self.cwe_counts[cwe] += 1
                    self.label_counts[1] += 1
                if saved_fixed:
                    self.cwe_counts[cwe] += 1
                    self.label_counts[0] += 1

    def _build_summary(self) -> dict:
        return {
            "commits_processed": self.commits_processed,
            "commits_skipped": self.commits_skipped,
            "commits_stale_404": self.commits_stale_404,
            "files_qualifying": self.files_qualifying,
            "files_rejected_ext": self.files_rejected_ext,
            "files_rejected_status": self.files_rejected_status,
            "files_rejected_size": self.files_rejected_size,
            "samples_saved": self._samples_saved,
            "samples_failed": self._samples_failed,
            "pairs_saved": self.pairs_saved,
            "label_distribution": dict(self.label_counts),
            "cwe_distribution": dict(self.cwe_counts),
            "tokens_available": len(self.rotator.tokens),
        }

    def print_summary(self) -> None:
        summary = self._build_summary()
        print("\n" + "=" * 60)
        print("GITHUB DIFF FETCHER SUMMARY (Phase 2)")
        print("=" * 60)
        print(f"Commits processed:    {summary['commits_processed']}")
        print(f"Commits skipped:      {summary['commits_skipped']}")
        print(f"Commits stale (404):  {summary['commits_stale_404']}")
        print(f"Files qualifying:     {summary['files_qualifying']}")
        print(f"Files rejected (ext): {summary['files_rejected_ext']}")
        print(f"Files rejected (st):  {summary['files_rejected_status']}")
        print(f"Files rejected (sz):  {summary['files_rejected_size']}")
        print(f"Samples saved:        {summary['samples_saved']}")
        print(f"Samples failed:       {summary['samples_failed']}")
        print(f"Pairs saved:          {summary['pairs_saved']}")
        print(f"GitHub tokens:        {summary['tokens_available']}")
        if summary["label_distribution"]:
            print("\nLabel Distribution:")
            for label, count in sorted(summary["label_distribution"].items()):
                label_str = "vulnerable" if label == 1 else "safe"
                print(f"  {label_str} (label={label}): {count}")
        if summary["cwe_distribution"]:
            print("\nCWE Distribution:")
            for cwe, count in sorted(
                summary["cwe_distribution"].items(), key=lambda x: -x[1]
            ):
                print(f"  {cwe}: {count}")
        print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def main():
    # Load .env for API keys (NVD_API_KEY, GITHUB_TOKEN, GITHUB_TOKENS)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        description="CVE/NVD Two-Phase Collector for StreamGuard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Phase 1: Build NVD index
  python cve_collector_enhanced.py --phase 1 --dry-run --max-samples 50

  # Phase 1: Full index build (requires NVD_API_KEY in .env)
  python cve_collector_enhanced.py --phase 1

  # Phase 2: Fetch GitHub diffs (requires GITHUB_TOKEN in .env)
  python cve_collector_enhanced.py --phase 2 --dry-run --max-samples 20

  # Phase 2: Multi-token collection (3x faster)
  python cve_collector_enhanced.py --phase 2 --github-tokens "tok1,tok2,tok3"
        """,
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        required=True,
        help="Phase 1: NVD index builder, Phase 2: GitHub diff fetcher",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/cve",
        help="Output directory (default: data/raw/cve)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Checkpoint directory (default: <output-dir>/checkpoints)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without writing output files",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Stop after saving N samples/entries (0=unlimited)",
    )
    parser.add_argument(
        "--nvd-api-key",
        default=None,
        help="NVD API key (fallback: NVD_API_KEY env var)",
    )
    parser.add_argument(
        "--github-token",
        default=None,
        help="Single GitHub token (fallback: GITHUB_TOKEN env var)",
    )
    parser.add_argument(
        "--github-tokens",
        default=None,
        help="Comma-separated GitHub tokens for multi-token rotation "
             "(fallback: GITHUB_TOKENS env var). 3 tokens = 15K req/hr.",
    )
    parser.add_argument(
        "--index-path",
        default=None,
        help="Path to cve_index.jsonl (Phase 2 only, default: <output-dir>/cve_index.jsonl)",
    )

    args = parser.parse_args()

    if args.phase == 1:
        logger.info("Starting Phase 1: NVD Index Builder...")
        collector = NVDIndexBuilder(
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
            nvd_api_key=args.nvd_api_key,
            dry_run=args.dry_run,
            max_samples=args.max_samples,
        )
    else:
        logger.info("Starting Phase 2: GitHub Diff Fetcher...")
        # Resolve tokens
        tokens = []
        if args.github_tokens:
            tokens = [t.strip() for t in args.github_tokens.split(",") if t.strip()]
        elif args.github_token:
            tokens = [args.github_token]

        collector = GitHubDiffFetcher(
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
            github_tokens=tokens or None,
            dry_run=args.dry_run,
            max_samples=args.max_samples,
            index_path=args.index_path,
        )

    try:
        collector.collect()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Checkpoint saved.")
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        raise
    finally:
        collector.print_summary()


if __name__ == "__main__":
    main()
