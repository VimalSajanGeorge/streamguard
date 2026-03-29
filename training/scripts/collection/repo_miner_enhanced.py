#!/usr/bin/env python3
"""
training/scripts/collection/repo_miner_enhanced.py

API-only repo miner for StreamGuard.
Mines security-related commits from 15 high-value C repositories via
the GitHub Commits API. NEVER uses git clone/fetch.

Steps:
  1. Paginate through commits for each target repo
  2. Score commit messages using weighted keyword filter (PI-05)
  3. Fetch full diff only for qualifying commits (score >= 2, no exclusions)
  4. Extract before/after C code from modified .c/.h files
  5. Save paired samples with shared pair_id
  6. Cross-dedup against CVE, OSV, and GitHub Advisory datasets

Source: docs/New Docs/DATA_PIPELINE.md §Collector 6: Repo Miner
"""
from __future__ import annotations

import argparse
import json
import os
import re
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


# ── Target Repositories (process in this exact order) ─────────────────────
REPOS = [
    "openssl/openssl",
    "curl/curl",
    "nginx/nginx",
    "FFmpeg/FFmpeg",
    "php/php-src",
    "sqlite/sqlite",
    "libpng/libpng",
    "madler/zlib",
    "git/git",
    "redis/redis",
    "libuv/libuv",
    "libevent/libevent",
    "openldap/openldap",
    "antirez/redis",
    "torvalds/linux",  # ~3000 security commits — process LAST
]

# ── Scored Keyword Filter (PI-05) ─────────────────────────────────────────
HIGH_KEYWORDS = ["cve-", "security", "vulnerability", "vuln", "exploit"]
MED_KEYWORDS = [
    "overflow", "buffer", "heap", "stack", "use-after-free",
    "null deref", "injection", "out of bounds", "oob", "sanitize",
]
LOW_KEYWORDS = ["fix", "patch", "bug", "issue", "crash", "error"]
EXCLUSION_KEYWORDS = [
    "typo", "whitespace", "formatting", "style", "copyright",
    "license", "documentation", "docs", "comment only",
]

HIGH_WEIGHT = 3
MED_WEIGHT = 2
LOW_WEIGHT = 1

# ── Diff Filter Thresholds ────────────────────────────────────────────────
MIN_CHANGED_LINES = 5
MAX_CHANGED_LINES = 150
VALID_DIFF_EXTENSIONS = {".c", ".h"}

# ── Checkpoint Interval ───────────────────────────────────────────────────
CHECKPOINT_PAGE_INTERVAL = 10

GITHUB_API_BASE = "https://api.github.com"


class RepoMiner(BaseCollector):
    """
    API-only repo miner. Paginate commits, score messages, fetch diffs.
    NEVER uses git clone/fetch — GitHub Commits API only.
    """

    def __init__(
        self,
        output_dir: str,
        checkpoint_dir: str | None = None,
        github_tokens: list[str] | None = None,
        dry_run: bool = False,
        max_samples: int = 0,
        repo_filter: str | None = None,
        cve_samples_path: str | None = None,
        osv_samples_path: str | None = None,
        github_advisory_samples_path: str | None = None,
    ):
        super().__init__(
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            source_name="repo",
        )

        # Resolve tokens from args or env
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
                "or use --github-token CLI flag."
            )

        self.rotator = GitHubTokenRotator(tokens)
        self.dry_run = dry_run
        self.max_samples = max_samples
        self.repo_filter = repo_filter

        # Cross-dedup SHA sets
        self._dedup_shas: set[str] = set()
        self._load_dedup_shas(cve_samples_path, osv_samples_path, github_advisory_samples_path)

        # SHA normalization cache (short -> full)
        self._sha_cache: dict[str, str] = {}

        # Counters
        self.repos_completed = 0
        self.commits_scored = 0
        self.commits_fetched = 0
        self.commits_skipped_score = 0
        self.commits_skipped_exclusion = 0
        self.commits_skipped_dedup = 0
        self.commits_stale_404 = 0
        self.files_qualifying = 0
        self.files_rejected_ext = 0
        self.files_rejected_status = 0
        self.files_rejected_size = 0
        self.pairs_saved = 0
        self.cwe_counts: Counter = Counter()
        self.label_counts: Counter = Counter()
        self.repo_sample_counts: Counter = Counter()

    # ── Cross-Dedup Loading ───────────────────────────────────────────

    def _load_dedup_shas(
        self,
        cve_path: str | None,
        osv_path: str | None,
        advisory_path: str | None,
    ) -> None:
        """Load commit_sha sets from CVE, OSV, and GitHub Advisory samples for cross-dedup."""
        default_base = Path(__file__).parent / "data" / "raw"
        paths = {
            "cve": Path(cve_path) if cve_path else default_base / "cve" / "cve_samples.jsonl",
            "osv": Path(osv_path) if osv_path else default_base / "osv" / "osv_samples.jsonl",
            "github_advisory": (
                Path(advisory_path) if advisory_path
                else default_base / "github_advisory" / "github_advisory_samples.jsonl"
            ),
        }

        for source_name, path in paths.items():
            if not path.exists():
                logger.debug(f"Cross-dedup: {source_name} file not found at {path}")
                continue
            count = 0
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sha = json.loads(line).get("commit_sha", "")
                        if sha and len(sha) >= 7:
                            self._dedup_shas.add(sha)
                            count += 1
                    except (json.JSONDecodeError, KeyError):
                        pass
            logger.info(f"Cross-dedup: loaded {count} SHAs from {source_name}")

        logger.info(f"Cross-dedup: {len(self._dedup_shas)} unique SHAs total")

    # ── Scored Keyword Filter (PI-05) ─────────────────────────────────

    @staticmethod
    def score_commit_message(message: str) -> tuple[int, bool]:
        """
        Score a commit message for security relevance.
        Returns (score, has_exclusion).
        """
        msg_lower = message.lower()

        # Check exclusions first
        for excl in EXCLUSION_KEYWORDS:
            if excl in msg_lower:
                return 0, True

        score = 0
        for kw in HIGH_KEYWORDS:
            if kw in msg_lower:
                score += HIGH_WEIGHT
        for kw in MED_KEYWORDS:
            if kw in msg_lower:
                score += MED_WEIGHT
        for kw in LOW_KEYWORDS:
            if kw in msg_lower:
                score += LOW_WEIGHT

        return score, False

    # ── CWE Inference ─────────────────────────────────────────────────

    @staticmethod
    def infer_cwe(message: str) -> str | None:
        """
        Infer CWE from commit message using same priority-ordered map as ExploitDB.
        Also checks for explicit CWE mentions.
        """
        msg_lower = message.lower()

        # Check for explicit CWE mention first (e.g., "CWE-121")
        cwe_match = re.search(r"cwe-(\d+)", msg_lower)
        if cwe_match:
            cwe_id = f"CWE-{cwe_match.group(1)}"
            if cwe_id in VALID_CWE:
                return cwe_id

        # Fall back to keyword inference (reuses ExploitDB map)
        for cwe, keywords in CWE_KEYWORD_MAP:
            for kw in keywords:
                if kw.lower() in msg_lower:
                    return cwe

        # Additional repo-miner-specific patterns
        extra_patterns = [
            ("CWE-119", ["memory corruption", "out-of-bounds write", "oob write"]),
            ("CWE-125", ["out-of-bounds read", "oob read", "read overflow"]),
        ]
        for cwe, patterns in extra_patterns:
            for p in patterns:
                if p in msg_lower:
                    return cwe

        return None

    # ── SHA Normalization (PI-06) ─────────────────────────────────────

    def _resolve_full_sha(self, owner: str, repo: str, short_sha: str) -> str | None:
        """Resolve short SHA to full 40-char via GitHub API. Caches results."""
        if len(short_sha) >= 40:
            return short_sha

        cache_key = f"{owner}/{repo}/{short_sha}"
        if cache_key in self._sha_cache:
            return self._sha_cache[cache_key]

        self.rotator.rotate_if_needed()
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits/{short_sha}"

        try:
            resp = self.safe_get(url, headers=self.rotator.auth_headers)
            self.rotator.update_limits(dict(resp.headers))

            if resp.status_code == 200:
                full_sha = resp.json().get("sha", short_sha)
                self._sha_cache[cache_key] = full_sha
                return full_sha
            return None
        except Exception as e:
            logger.debug(f"SHA resolution failed for {short_sha}: {e}")
            return None

    # ── Commit Listing (Paginated) ────────────────────────────────────

    def _list_commits(self, owner: str, repo: str, page: int) -> list[dict] | None:
        """
        Fetch one page of commits from GitHub Commits API.
        GET /repos/{owner}/{repo}/commits?per_page=100&page={n}
        Returns list of commit dicts or None on failure.
        """
        self.rotator.rotate_if_needed()
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits"
        params = {"per_page": 100, "page": page}

        try:
            resp = self.safe_get(url, headers=self.rotator.auth_headers, params=params)
            self.rotator.update_limits(dict(resp.headers))

            if resp.status_code == 409:
                # Empty repository
                logger.warning(f"{owner}/{repo}: empty repository (409)")
                return []
            if resp.status_code != 200:
                logger.warning(f"{owner}/{repo} page {page}: HTTP {resp.status_code}")
                return None
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to list commits for {owner}/{repo} page {page}: {e}")
            return None

    # ── Commit Detail Fetch ───────────────────────────────────────────

    def _fetch_commit(self, owner: str, repo: str, sha: str) -> dict | None:
        """Fetch a single commit's detail (including file diffs)."""
        self.rotator.rotate_if_needed()
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits/{sha}"

        try:
            resp = self.safe_get(url, headers=self.rotator.auth_headers)
            self.rotator.update_limits(dict(resp.headers))

            if resp.status_code == 404:
                return None
            if resp.status_code != 200:
                logger.warning(f"GitHub returned {resp.status_code} for {owner}/{repo}/{sha}")
                return None
            return resp.json()
        except Exception as e:
            logger.error(f"GitHub fetch failed for {owner}/{repo}/{sha}: {e}")
            return None

    # ── File Qualification ────────────────────────────────────────────

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

        # Patch must exist
        if not patch:
            return False, "no_patch"

        # Changed lines check
        changed = GitHubDiffFetcher.count_changed_lines(patch)
        if changed < MIN_CHANGED_LINES or changed > MAX_CHANGED_LINES:
            return False, "size"

        return True, ""

    # ── Main Collection ───────────────────────────────────────────────

    def collect(self) -> dict:
        """
        Mine all target repos in order. Returns summary dict.
        """
        if self.dry_run:
            logger.info("DRY RUN MODE — will process but not save samples")

        # Determine repos to process
        repos = REPOS
        if self.repo_filter:
            if self.repo_filter not in REPOS:
                logger.warning(
                    f"--repo-filter '{self.repo_filter}' not in REPOS list. "
                    f"Processing it anyway."
                )
            repos = [self.repo_filter]

        # Load checkpoint
        checkpoint = self.load_checkpoint()
        repo_status: dict = checkpoint.get("repo_status", {})
        repo_page: dict = checkpoint.get("repo_page", {})
        done_shas: set[str] = set(checkpoint.get("done_shas", []))

        if done_shas:
            logger.info(f"Checkpoint: {len(done_shas)} commits already processed")

        for repo_name in repos:
            if self.max_samples > 0 and self._samples_saved >= self.max_samples:
                logger.info(f"Reached max_samples ({self.max_samples}). Stopping.")
                break

            # Skip completed repos
            if repo_status.get(repo_name) == "complete":
                logger.info(f"Skipping {repo_name} (already complete)")
                self.repos_completed += 1
                continue

            logger.info(f"Starting repo: {repo_name}")
            repo_status[repo_name] = "in_progress"

            owner, repo = repo_name.split("/")
            start_page = repo_page.get(repo_name, 1)

            self._mine_repo(
                owner, repo, repo_name, start_page,
                done_shas, repo_status, repo_page,
            )

            repo_status[repo_name] = "complete"
            self.repos_completed += 1

            # Checkpoint after each repo
            self._save_full_checkpoint(repo_status, repo_page, done_shas)
            logger.info(
                f"Completed {repo_name}. "
                f"Total samples: {self._samples_saved}, pairs: {self.pairs_saved}"
            )

        # Final checkpoint
        self._save_full_checkpoint(repo_status, repo_page, done_shas)

        # Finalize orphan pairs
        if not self.dry_run:
            orphans = self.finalize_pairs()
            if orphans:
                logger.info(f"Cleared {orphans} orphan pair_ids")

        return self._build_summary()

    def _mine_repo(
        self,
        owner: str,
        repo: str,
        repo_name: str,
        start_page: int,
        done_shas: set[str],
        repo_status: dict,
        repo_page: dict,
    ) -> None:
        """Paginate through commits for a single repo."""
        page = start_page
        consecutive_empty = 0

        while True:
            if self.max_samples > 0 and self._samples_saved >= self.max_samples:
                break

            commits = self._list_commits(owner, repo, page)
            if commits is None:
                # API error — skip page, try next
                logger.warning(f"{repo_name} page {page}: API error, skipping")
                page += 1
                consecutive_empty += 1
                if consecutive_empty >= 5:
                    logger.warning(f"{repo_name}: 5 consecutive empty/error pages, stopping")
                    break
                continue

            if not commits:
                # No more commits
                break

            consecutive_empty = 0

            for commit_entry in commits:
                if self.max_samples > 0 and self._samples_saved >= self.max_samples:
                    break

                sha = commit_entry.get("sha", "")
                message = commit_entry.get("commit", {}).get("message", "")

                # SHA normalization
                if sha and len(sha) < 40:
                    full_sha = self._resolve_full_sha(owner, repo, sha)
                    if full_sha:
                        sha = full_sha

                # Skip already-processed commits
                if sha in done_shas:
                    continue

                # Skip cross-dedup
                if sha in self._dedup_shas:
                    self.commits_skipped_dedup += 1
                    done_shas.add(sha)
                    continue

                # Score the commit message
                score, has_exclusion = self.score_commit_message(message)
                self.commits_scored += 1

                if has_exclusion:
                    self.commits_skipped_exclusion += 1
                    done_shas.add(sha)
                    continue

                if score < 2:
                    self.commits_skipped_score += 1
                    done_shas.add(sha)
                    continue

                # Qualifying commit — fetch full diff
                self._process_commit(owner, repo, repo_name, sha, message, done_shas)
                done_shas.add(sha)

                # Jitter between commit fetches
                self.jitter_sleep(0.05)

            page += 1

            # Checkpoint every CHECKPOINT_PAGE_INTERVAL pages
            if page % CHECKPOINT_PAGE_INTERVAL == 0:
                repo_page[repo_name] = page
                self._save_full_checkpoint(repo_status, repo_page, done_shas)
                logger.info(
                    f"{repo_name} page {page}: "
                    f"scored={self.commits_scored}, "
                    f"fetched={self.commits_fetched}, "
                    f"saved={self._samples_saved}"
                )

        # Record last page for this repo
        repo_page[repo_name] = page

    def _process_commit(
        self,
        owner: str,
        repo: str,
        repo_name: str,
        sha: str,
        message: str,
        done_shas: set[str],
    ) -> None:
        """Fetch commit diff, extract C code, save paired samples."""
        commit_data = self._fetch_commit(owner, repo, sha)
        if commit_data is None:
            self.commits_stale_404 += 1
            return

        self.commits_fetched += 1

        # Infer CWE from commit message
        cwe = self.infer_cwe(message)
        if not cwe:
            # No CWE inferable — skip this commit
            return

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

            before_code, after_code = GitHubDiffFetcher.extract_before_after(patch)

            # Create paired samples
            timestamp = make_timestamp()
            pair_id = str(uuid.uuid4())

            vuln_sample = {
                "id": make_sample_id(),
                "source": "repo",
                "code": before_code,
                "label": 1,
                "cwe": cwe,
                "language": "c",
                "collected_at": timestamp,
                "pair_id": pair_id,
                "commit_sha": sha,
                "file_path": filename,
                "cfa_origin": "native",
                "metadata": {
                    "repo_name": repo_name,
                    "commit_message": message[:500],
                    "cwe_confidence": "tier3_inferred",
                },
            }

            fixed_sample = {
                "id": make_sample_id(),
                "source": "repo",
                "code": after_code,
                "label": 0,
                "cwe": cwe,
                "language": "c",
                "collected_at": timestamp,
                "pair_id": pair_id,
                "commit_sha": sha,
                "file_path": filename,
                "cfa_origin": "native",
                "metadata": {
                    "repo_name": repo_name,
                    "commit_message": message[:500],
                    "cwe_confidence": "tier3_inferred",
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
                    self.repo_sample_counts[repo_name] += 1
                if saved_fixed:
                    self.cwe_counts[cwe] += 1
                    self.label_counts[0] += 1
                    self.repo_sample_counts[repo_name] += 1

    # ── Checkpoint Helpers ────────────────────────────────────────────

    def _save_full_checkpoint(
        self,
        repo_status: dict,
        repo_page: dict,
        done_shas: set[str],
    ) -> None:
        """Save full checkpoint state."""
        self.save_checkpoint({
            "repo_status": repo_status,
            "repo_page": repo_page,
            "done_shas": list(done_shas),
            "total_saved": self._samples_saved,
        })

    # ── Summary ───────────────────────────────────────────────────────

    def _build_summary(self) -> dict:
        return {
            "repos_completed": self.repos_completed,
            "commits_scored": self.commits_scored,
            "commits_fetched": self.commits_fetched,
            "commits_skipped_score": self.commits_skipped_score,
            "commits_skipped_exclusion": self.commits_skipped_exclusion,
            "commits_skipped_dedup": self.commits_skipped_dedup,
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
            "repo_distribution": dict(self.repo_sample_counts),
            "tokens_available": len(self.rotator.tokens),
        }

    def print_summary(self) -> None:
        summary = self._build_summary()
        print("\n" + "=" * 60)
        print("REPO MINER SUMMARY")
        print("=" * 60)
        print(f"Repos completed:         {summary['repos_completed']}/{len(REPOS)}")
        print(f"Commits scored:          {summary['commits_scored']}")
        print(f"Commits fetched (diff):  {summary['commits_fetched']}")
        print(f"Commits skipped (score): {summary['commits_skipped_score']}")
        print(f"Commits skipped (excl):  {summary['commits_skipped_exclusion']}")
        print(f"Commits skipped (dedup): {summary['commits_skipped_dedup']}")
        print(f"Commits stale (404):     {summary['commits_stale_404']}")
        print(f"Files qualifying:        {summary['files_qualifying']}")
        print(f"Samples saved:           {summary['samples_saved']}")
        print(f"Samples failed:          {summary['samples_failed']}")
        print(f"Pairs saved:             {summary['pairs_saved']}")
        print(f"GitHub tokens:           {summary['tokens_available']}")
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
        if summary["repo_distribution"]:
            print("\nPer-Repo Distribution:")
            for repo_name, count in sorted(
                summary["repo_distribution"].items(), key=lambda x: -x[1]
            ):
                print(f"  {repo_name}: {count}")
        print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        description="StreamGuard Repo Miner — API-only commit mining"
    )
    parser.add_argument(
        "--output-dir",
        default="training/scripts/collection/data/raw/repo",
        help="Output directory for samples JSONL",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="training/scripts/collection/data/raw/checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--github-token",
        help="GitHub token(s), comma-separated for rotation",
    )
    parser.add_argument(
        "--repo-filter",
        help="Process only this repo (e.g., 'openssl/openssl')",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Stop after saving N samples (0 = unlimited)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process but don't save samples",
    )
    parser.add_argument(
        "--cve-samples-path",
        help="Path to cve_samples.jsonl for cross-dedup",
    )
    parser.add_argument(
        "--osv-samples-path",
        help="Path to osv_samples.jsonl for cross-dedup",
    )
    parser.add_argument(
        "--github-advisory-samples-path",
        help="Path to github_advisory_samples.jsonl for cross-dedup",
    )

    args = parser.parse_args()

    tokens = None
    if args.github_token:
        tokens = [t.strip() for t in args.github_token.split(",") if t.strip()]

    miner = RepoMiner(
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        github_tokens=tokens,
        dry_run=args.dry_run,
        max_samples=args.max_samples,
        repo_filter=args.repo_filter,
        cve_samples_path=args.cve_samples_path,
        osv_samples_path=args.osv_samples_path,
        github_advisory_samples_path=args.github_advisory_samples_path,
    )

    logger.info("Starting repo miner collection...")
    summary = miner.collect()
    miner.print_summary()

    # Save summary
    summary_path = Path(args.output_dir) / "repo_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
