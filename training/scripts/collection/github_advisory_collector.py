#!/usr/bin/env python3
"""
training/scripts/collection/github_advisory_collector.py

GitHub Advisory GraphQL collector for StreamGuard.

Uses the `securityAdvisories` GraphQL query to scan ALL advisories.
Filters by valid CWE first, then follows GitHub commit URLs and relies on
is_qualifying_file() (.c/.h extension check) as the language gate.
No package-name pre-filter — the file extension is ground truth.

Source: docs/New Docs/DATA_PIPELINE.md §Collector 4: GitHub Advisory
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from collections import Counter
from pathlib import Path

from loguru import logger

try:
    from .base_collector import BaseCollector
    from .schema import VALID_CWE, make_sample_id, make_timestamp, validate_sample
    from .cve_collector_enhanced import GitHubTokenRotator, GitHubDiffFetcher
    from .exploitdb_collector import CWE_KEYWORD_MAP
except ImportError:
    from base_collector import BaseCollector
    from schema import VALID_CWE, make_sample_id, make_timestamp, validate_sample
    from cve_collector_enhanced import GitHubTokenRotator, GitHubDiffFetcher
    from exploitdb_collector import CWE_KEYWORD_MAP


# ── Constants ─────────────────────────────────────────────────────────────

GRAPHQL_ENDPOINT = "https://api.github.com/graphql"

# Known C/C++ project names that appear as package names in GitHub advisories.
# Substring match (case-insensitive) against advisory package names.
C_PROJECT_KEYWORDS = {
    "linux", "kernel", "openssl", "curl", "libcurl",
    "glibc", "libc", "zlib", "libpng", "libtiff",
    "ffmpeg", "imagemagick", "sqlite", "busybox",
    "qemu", "wireshark", "nginx", "apache",
}

RATE_CHECK_INTERVAL = 50    # Check rate limit every N advisories
RATE_LIMIT_FLOOR = 400      # Sleep if remaining < this


# ── GraphQL query ─────────────────────────────────────────────────────────

GRAPHQL_QUERY = """
query($first: Int!, $after: String) {
  securityAdvisories(
    first: $first,
    after: $after,
    orderBy: {field: PUBLISHED_AT, direction: DESC}
  ) {
    nodes {
      ghsaId
      summary
      description
      publishedAt
      vulnerabilities(first: 10) {
        nodes {
          package {
            name
            ecosystem
          }
        }
      }
      references {
        url
      }
      cvss {
        score
      }
      cwes(first: 5) {
        nodes {
          cweId
          name
        }
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


class GitHubAdvisoryCollector(BaseCollector):
    """
    Collects C/C++ vulnerability samples from GitHub Security Advisories.
    Inherits checkpoint/resume, dedup, and validation from BaseCollector.
    """

    def __init__(
        self,
        output_dir: str,
        checkpoint_dir: str | None = None,
        github_tokens: list[str] | None = None,
        dry_run: bool = False,
        max_samples: int = 0,
    ):
        super().__init__(
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            source_name="github_advisory",
        )

        # Resolve tokens: CLI args > GITHUB_TOKENS env > GITHUB_TOKEN env
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
                "or use --github-tokens CLI flag."
            )

        self.rotator = GitHubTokenRotator(tokens)
        self.dry_run = dry_run
        self.max_samples = max_samples

        # Counters
        self.advisories_seen = 0
        self.advisories_no_cwe = 0
        self.advisories_no_commit = 0
        self.advisories_with_commit = 0
        self.commits_processed = 0
        self.commits_no_c_files = 0
        self.commits_stale_404 = 0
        self.files_qualifying = 0
        self.pairs_saved = 0
        self.cwe_counts: Counter = Counter()
        self.label_counts: Counter = Counter()
        self.page_count = 0

    # ── C project filter ──────────────────────────────────────────────

    @staticmethod
    def _is_c_project(advisory: dict) -> bool:
        """Check if any package name matches C_PROJECT_KEYWORDS (substring, case-insensitive)."""
        vulns = advisory.get("vulnerabilities", {}).get("nodes", [])
        for v in vulns:
            pkg = v.get("package")
            if not pkg:
                continue
            name = (pkg.get("name") or "").lower()
            for kw in C_PROJECT_KEYWORDS:
                if kw in name:
                    return True
        return False

    # ── CWE extraction ────────────────────────────────────────────────

    @staticmethod
    def _extract_cwe(advisory: dict) -> str | None:
        """
        Extract CWE from advisory cwes.nodes.
        Falls back to keyword inference from summary+description.
        Returns canonical CWE ID or None.
        """
        # Primary: from cwes.nodes
        cwes_nodes = advisory.get("cwes", {}).get("nodes", [])
        for node in cwes_nodes:
            cwe_id = node.get("cweId", "")
            if cwe_id in VALID_CWE:
                return cwe_id

        # Fallback: keyword inference from description
        summary = advisory.get("summary", "") or ""
        description = advisory.get("description", "") or ""
        combined = (summary + " " + description).lower()
        for cwe, keywords in CWE_KEYWORD_MAP:
            for kw in keywords:
                if kw.lower() in combined:
                    return cwe
        return None

    # ── Commit URL extraction ─────────────────────────────────────────

    @staticmethod
    def _extract_commit_urls(references: list[dict]) -> list[str]:
        """Extract GitHub commit URLs from advisory references."""
        urls = []
        for ref in references:
            url = ref.get("url", "")
            if "github.com" in url and "/commit/" in url:
                if "/pull/" not in url and "/issues/" not in url:
                    urls.append(url)
        return urls

    # ── GraphQL query ─────────────────────────────────────────────────

    def _query_advisories(self, after_cursor: str | None = None) -> dict | None:
        """Execute GraphQL query for securityAdvisories. Returns parsed JSON or None."""
        self.rotator.rotate_if_needed()

        variables = {"first": 100}
        if after_cursor:
            variables["after"] = after_cursor

        try:
            resp = self.safe_post(
                GRAPHQL_ENDPOINT,
                headers={
                    **self.rotator.auth_headers,
                    "Content-Type": "application/json",
                },
                json={"query": GRAPHQL_QUERY, "variables": variables},
                timeout=45,
            )
            self.rotator.update_limits(dict(resp.headers))

            if resp.status_code != 200:
                logger.warning(f"GraphQL returned {resp.status_code}: {resp.text[:200]}")
                return None

            data = resp.json()
            if "errors" in data:
                error_msg = str(data["errors"])
                logger.warning(f"GraphQL errors: {error_msg[:300]}")
                # PI-04: cursor recovery
                if "cursor" in error_msg.lower():
                    logger.warning("Cursor error detected — will restart from page 1")
                    return {"_cursor_error": True}
            return data
        except Exception as e:
            logger.error(f"GraphQL query failed: {e}")
            return None

    # ── Rate limit check ──────────────────────────────────────────────

    def _check_graphql_rate_limit(self, rate_limit_data: dict) -> None:
        """Sleep if GraphQL rate limit remaining < RATE_LIMIT_FLOOR."""
        remaining = rate_limit_data.get("remaining", 5000)
        reset_at = rate_limit_data.get("resetAt", "")

        if remaining < RATE_LIMIT_FLOOR:
            # Parse resetAt ISO timestamp
            try:
                import datetime as dt
                reset_time = dt.datetime.fromisoformat(reset_at.replace("Z", "+00:00"))
                now = dt.datetime.now(dt.timezone.utc)
                wait_secs = max((reset_time - now).total_seconds() + 5, 1)
            except (ValueError, AttributeError):
                wait_secs = 60

            logger.warning(
                f"GraphQL rate limit low: {remaining} remaining. "
                f"Sleeping {wait_secs:.0f}s."
            )
            time.sleep(wait_secs)

    # ── Commit fetching ───────────────────────────────────────────────

    def _fetch_commit(self, owner: str, repo: str, sha: str) -> dict | None:
        """Fetch commit details from GitHub REST API."""
        self.rotator.rotate_if_needed()
        url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"

        try:
            resp = self.safe_get(url, headers=self.rotator.auth_headers)
            self.rotator.update_limits(dict(resp.headers))

            if resp.status_code == 404:
                return None
            if resp.status_code != 200:
                logger.warning(f"GitHub returned {resp.status_code} for {url}")
                return None
            return resp.json()
        except Exception as e:
            logger.error(f"GitHub fetch failed for {owner}/{repo}/{sha}: {e}")
            return None

    # ── Process one advisory ──────────────────────────────────────────

    def _process_advisory(self, advisory: dict) -> None:
        """Full processing pipeline for one advisory."""
        ghsa_id = advisory.get("ghsaId", "")

        # Extract CWE first — skip early before any API calls
        cwe = self._extract_cwe(advisory)
        if not cwe:
            self.advisories_no_cwe += 1
            return

        # Find commit URLs in references — skip before API calls
        references = advisory.get("references", [])
        commit_urls = self._extract_commit_urls(references)
        if not commit_urls:
            self.advisories_no_commit += 1
            return

        self.advisories_with_commit += 1

        # Extract CVSS
        cvss_data = advisory.get("cvss") or {}
        severity_score = float(cvss_data.get("score", 0.0))

        # Process each commit URL
        for commit_url in commit_urls:
            if self.max_samples > 0 and self._samples_saved >= self.max_samples:
                break

            parsed = GitHubDiffFetcher.parse_github_commit_url(commit_url)
            if not parsed:
                continue

            owner, repo, sha = parsed

            commit_data = self._fetch_commit(owner, repo, sha)
            if commit_data is None:
                self.commits_stale_404 += 1
                continue

            self.commits_processed += 1

            # Quick check: does this commit touch any .c/.h files?
            files = commit_data.get("files", [])
            has_c_file = any(
                f.get("filename", "").lower().endswith((".c", ".h"))
                for f in files
            )
            if not has_c_file:
                self.commits_no_c_files += 1
                continue
            for file_info in files:
                if self.max_samples > 0 and self._samples_saved >= self.max_samples:
                    break

                qualifies, _ = GitHubDiffFetcher.is_qualifying_file(file_info)
                if not qualifies:
                    continue

                self.files_qualifying += 1
                patch = file_info.get("patch", "")
                filename = file_info.get("filename", "")

                before_code, after_code = GitHubDiffFetcher.extract_before_after(patch)

                timestamp = make_timestamp()
                pair_id = str(uuid.uuid4())

                vuln_sample = {
                    "id": make_sample_id(),
                    "source": "github_advisory",
                    "code": before_code,
                    "label": 1,
                    "cwe": cwe,
                    "language": "c",
                    "collected_at": timestamp,
                    "pair_id": pair_id,
                    "commit_sha": sha,
                    "severity_score": severity_score,
                    "file_path": filename,
                    "cfa_origin": "native",
                    "metadata": {
                        "ghsa_id": ghsa_id,
                        "github_url": commit_url,
                        "owner": owner,
                        "repo": repo,
                    },
                }

                fixed_sample = {
                    "id": make_sample_id(),
                    "source": "github_advisory",
                    "code": after_code,
                    "label": 0,
                    "cwe": cwe,
                    "language": "c",
                    "collected_at": timestamp,
                    "pair_id": pair_id,
                    "commit_sha": sha,
                    "severity_score": severity_score,
                    "file_path": filename,
                    "cfa_origin": "native",
                    "metadata": {
                        "ghsa_id": ghsa_id,
                        "github_url": commit_url,
                        "owner": owner,
                        "repo": repo,
                    },
                }

                # Validate both
                vuln_valid, _ = validate_sample(vuln_sample)
                fixed_valid, _ = validate_sample(fixed_sample)

                if not vuln_valid or not fixed_valid:
                    self._samples_failed += 2
                    if not self.dry_run:
                        if not vuln_valid:
                            self.save_failed_item(vuln_sample, reason="schema validation")
                        if not fixed_valid:
                            self.save_failed_item(fixed_sample, reason="schema validation")
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

            self.jitter_sleep(0.3)

    # ── Main collection loop ──────────────────────────────────────────

    def collect(self) -> dict:
        """
        Paginate securityAdvisories, filter for C projects, process diffs.
        Returns summary dict.
        """
        if self.dry_run:
            logger.info("DRY RUN MODE — will process but not save samples")

        # Load checkpoint
        checkpoint = self.load_checkpoint()
        last_cursor = checkpoint.get("last_cursor")
        seen_advisory_ids: set[str] = set(checkpoint.get("seen_advisory_ids", []))
        self.page_count = checkpoint.get("page_count", 0)

        if seen_advisory_ids:
            logger.info(
                f"Checkpoint: {len(seen_advisory_ids)} advisories already seen, "
                f"page_count={self.page_count}"
            )

        cursor_error_count = 0
        query_fail_count = 0
        MAX_CURSOR_ERRORS = 3
        MAX_QUERY_FAILURES = 5

        while True:
            if self.max_samples > 0 and self._samples_saved >= self.max_samples:
                logger.info(f"Reached max_samples ({self.max_samples}). Stopping.")
                break

            result = self._query_advisories(after_cursor=last_cursor)
            if result is None:
                query_fail_count += 1
                if query_fail_count >= MAX_QUERY_FAILURES:
                    logger.error(
                        f"GraphQL query failed {MAX_QUERY_FAILURES} consecutive times. Stopping."
                    )
                    break
                logger.warning(
                    f"GraphQL query returned None ({query_fail_count}/{MAX_QUERY_FAILURES}). "
                    f"Retrying in 30s..."
                )
                time.sleep(30)
                continue
            query_fail_count = 0  # Reset on success

            # PI-04: cursor recovery
            if result.get("_cursor_error"):
                cursor_error_count += 1
                if cursor_error_count >= MAX_CURSOR_ERRORS:
                    logger.error(
                        f"Cursor error repeated {MAX_CURSOR_ERRORS} times. Stopping."
                    )
                    break
                logger.warning(
                    f"Cursor error ({cursor_error_count}/{MAX_CURSOR_ERRORS}) "
                    f"— clearing cursor and restarting from page 1"
                )
                last_cursor = None
                self.save_checkpoint({
                    "last_cursor": None,
                    "seen_advisory_ids": list(seen_advisory_ids),
                    "page_count": self.page_count,
                    "samples_saved": self._samples_saved,
                })
                continue

            advisories_data = result.get("data", {}).get("securityAdvisories", {})
            nodes = advisories_data.get("nodes", [])
            page_info = advisories_data.get("pageInfo", {})

            if not nodes:
                logger.info("No more advisories. Stopping.")
                break

            # Save cursor immediately after receiving page
            last_cursor = page_info.get("endCursor")
            self.page_count += 1

            self.save_checkpoint({
                "last_cursor": last_cursor,
                "seen_advisory_ids": list(seen_advisory_ids),
                "page_count": self.page_count,
                "samples_saved": self._samples_saved,
            })

            # Check rate limit
            rate_limit = result.get("data", {}).get("rateLimit", {})
            if rate_limit:
                self._check_graphql_rate_limit(rate_limit)

            # Process advisories
            for advisory in nodes:
                if self.max_samples > 0 and self._samples_saved >= self.max_samples:
                    break

                ghsa_id = advisory.get("ghsaId", "")
                self.advisories_seen += 1

                # Skip already-seen advisories (for cursor recovery restart)
                if ghsa_id in seen_advisory_ids:
                    continue
                seen_advisory_ids.add(ghsa_id)

                if len(seen_advisory_ids) % 10000 == 0:
                    logger.info(f"seen_advisory_ids set size: {len(seen_advisory_ids)}")

                # No package-name pre-filter — rely on .c/.h file extension
                # check in _process_advisory after fetching commit diff.
                self._process_advisory(advisory)

            # Check pagination
            if not page_info.get("hasNextPage", False):
                logger.info("No more pages. Collection complete.")
                break

        # Final checkpoint
        self.save_checkpoint({
            "last_cursor": last_cursor,
            "seen_advisory_ids": list(seen_advisory_ids),
            "page_count": self.page_count,
            "samples_saved": self._samples_saved,
            "completed": True,
        })

        # Finalize orphan pairs
        if not self.dry_run:
            orphans = self.finalize_pairs()
            if orphans:
                logger.info(f"Cleared {orphans} orphan pair_ids")

        return self._build_summary()

    # ── Summary ───────────────────────────────────────────────────────

    def _build_summary(self) -> dict:
        return {
            "advisories_seen": self.advisories_seen,
            "advisories_no_cwe": self.advisories_no_cwe,
            "advisories_no_commit": self.advisories_no_commit,
            "advisories_with_commit": self.advisories_with_commit,
            "commits_processed": self.commits_processed,
            "commits_no_c_files": self.commits_no_c_files,
            "commits_stale_404": self.commits_stale_404,
            "files_qualifying": self.files_qualifying,
            "samples_saved": self._samples_saved,
            "samples_failed": self._samples_failed,
            "pairs_saved": self.pairs_saved,
            "pages_fetched": self.page_count,
            "label_distribution": dict(self.label_counts),
            "cwe_distribution": dict(self.cwe_counts),
            "tokens_available": len(self.rotator.tokens),
        }

    def print_summary(self) -> None:
        summary = self._build_summary()
        print("\n" + "=" * 60)
        print("GITHUB ADVISORY COLLECTOR SUMMARY")
        print("=" * 60)
        print(f"Advisories seen:       {summary['advisories_seen']}")
        print(f"No CWE:                {summary['advisories_no_cwe']}")
        print(f"No commit URL:         {summary['advisories_no_commit']}")
        print(f"With commit URL:       {summary['advisories_with_commit']}")
        print(f"Commits processed:     {summary['commits_processed']}")
        print(f"Commits no .c/.h:      {summary['commits_no_c_files']}")
        print(f"Commits stale (404):   {summary['commits_stale_404']}")
        print(f"Files qualifying:      {summary['files_qualifying']}")
        print(f"Samples saved:         {summary['samples_saved']}")
        print(f"Samples failed:        {summary['samples_failed']}")
        print(f"Pairs saved:           {summary['pairs_saved']}")
        print(f"Pages fetched:         {summary['pages_fetched']}")
        print(f"GitHub tokens:         {summary['tokens_available']}")
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


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        description="GitHub Advisory GraphQL Collector for StreamGuard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run with 10 samples
  python -m training.scripts.collection.github_advisory_collector --dry-run --max-samples 10

  # Full collection with multi-token rotation
  python -m training.scripts.collection.github_advisory_collector --github-tokens "tok1,tok2"
        """,
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/github_advisory",
        help="Output directory (default: data/raw/github_advisory)",
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
        help="Stop after saving N samples (0=unlimited)",
    )
    parser.add_argument(
        "--github-tokens",
        default=None,
        help="Comma-separated GitHub tokens for multi-token rotation "
             "(fallback: GITHUB_TOKENS / GITHUB_TOKEN env vars)",
    )

    args = parser.parse_args()

    tokens = []
    if args.github_tokens:
        tokens = [t.strip() for t in args.github_tokens.split(",") if t.strip()]

    logger.info("Starting GitHub Advisory collector...")
    collector = GitHubAdvisoryCollector(
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        github_tokens=tokens or None,
        dry_run=args.dry_run,
        max_samples=args.max_samples,
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
