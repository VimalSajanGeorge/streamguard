#!/usr/bin/env python3
"""
training/scripts/collection/osv_collector.py

OSV (Open Source Vulnerabilities) collector for StreamGuard.

Queries Linux and OSS-Fuzz ecosystems via GCS bulk ZIP download.
Extracts commit SHAs from ranges[].events[].fixed (type=GIT).
Fetches commit diffs from GitHub and creates paired C vulnerability samples.

Cross-deduplicates against CVE collector output to prevent test set contamination.

Source: docs/New Docs/DATA_PIPELINE.md §Collector 5: OSV
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import time
import uuid
import zipfile
from collections import Counter
from pathlib import Path

import requests
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

OSV_GCS_BASE = "https://osv-vulnerabilities.storage.googleapis.com"

# Per spec: Linux kernel vulns (C) and OSS-Fuzz (C/C++ memory bugs)
ECOSYSTEMS = ["Linux", "OSS-Fuzz"]

# Checkpoint frequency
CHECKPOINT_INTERVAL = 50  # Save after every N commits processed


class OSVCollector(BaseCollector):
    """
    Collects C vulnerability samples from OSV (Linux + OSS-Fuzz ecosystems).
    Downloads bulk ZIP from GCS, extracts commit SHAs, fetches GitHub diffs.
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
            source_name="osv",
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

        # Cross-dedup set: commit SHAs already in CVE collector output
        self._cve_dedup_shas: set[str] = set()

        # Counters
        self.vulns_scanned = 0
        self.vulns_no_git_range = 0
        self.vulns_no_cwe = 0
        self.vulns_deduped_cve = 0
        self.commits_processed = 0
        self.commits_stale_404 = 0
        self.files_qualifying = 0
        self.pairs_saved = 0
        self.cwe_counts: Counter = Counter()
        self.label_counts: Counter = Counter()
        self.ecosystem_counts: Counter = Counter()

    # ── CVE Cross-Dedup ──────────────────────────────────────────────

    def _load_cve_dedup_set(self) -> set[str]:
        """
        Load all commit_sha values from CVE collector output.
        Used to skip OSV vulns whose fixed commit already exists in CVE data.
        Returns set of commit SHAs.
        """
        cve_path = self.output_dir.parent / "cve" / "cve_samples.jsonl"
        shas: set[str] = set()

        if not cve_path.exists():
            logger.info("No CVE samples found for cross-dedup — skipping")
            return shas

        try:
            with open(cve_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                        sha = sample.get("commit_sha", "")
                        if sha:
                            shas.add(sha)
                    except json.JSONDecodeError:
                        pass
            logger.info(f"Loaded {len(shas)} commit SHAs from CVE data for cross-dedup")
        except Exception as e:
            logger.warning(f"Failed to load CVE dedup set: {e}")

        return shas

    # ── GCS Bulk Download ────────────────────────────────────────────

    def _download_ecosystem_zip(self, ecosystem: str) -> list[dict]:
        """
        Download all.zip from GCS for an ecosystem and parse all JSON vulns.
        Returns list of parsed vulnerability dicts.
        """
        url = f"{OSV_GCS_BASE}/{ecosystem}/all.zip"
        logger.info(f"Downloading {url} ...")

        try:
            resp = requests.get(url, timeout=300, stream=True)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 404:
                logger.warning(f"No GCS data found for {ecosystem} (404)")
            else:
                logger.error(f"HTTP error downloading {ecosystem}: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to download {ecosystem} ZIP: {e}")
            return []

        vulns = []
        try:
            zip_data = io.BytesIO(resp.content)
            with zipfile.ZipFile(zip_data) as zf:
                json_files = [f for f in zf.namelist() if f.endswith(".json")]
                logger.info(f"Parsing {len(json_files)} JSON files from {ecosystem}/all.zip")

                for json_file in json_files:
                    try:
                        with zf.open(json_file) as jf:
                            vuln = json.load(jf)
                            vulns.append(vuln)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.debug(f"Failed to parse {json_file}: {e}")
                        continue
        except zipfile.BadZipFile:
            logger.error(f"Corrupt ZIP file for {ecosystem}")
            return []

        logger.info(f"Extracted {len(vulns)} vulnerabilities from {ecosystem}")
        return vulns

    # ── Vulnerability Parsing ────────────────────────────────────────

    @staticmethod
    def _extract_commit_shas(vuln: dict) -> list[tuple[str, str, str]]:
        """
        Extract GitHub commit SHAs from ranges[].events[].fixed where type=GIT.
        Returns list of (owner, repo, sha) tuples.
        """
        results = []
        for affected in vuln.get("affected", []):
            for rng in affected.get("ranges", []):
                if rng.get("type") != "GIT":
                    continue
                repo_url = rng.get("repo", "")
                # Parse GitHub repo URL
                m = re.match(
                    r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?$",
                    repo_url,
                )
                if not m:
                    continue
                owner, repo = m.group(1), m.group(2)

                for event in rng.get("events", []):
                    sha = event.get("fixed", "")
                    if sha and len(sha) >= 7:
                        results.append((owner, repo, sha))
        return results

    @staticmethod
    def _extract_cwe(vuln: dict) -> tuple[str | None, str]:
        """
        Extract CWE from OSV vulnerability data.
        Priority:
          1. database_specific.cwe_ids (array) → tier1_direct
          2. Keyword inference from summary/details → tier3_inferred
        Returns (canonical CWE ID or None, confidence tier).
        R-04: CWE confidence tiers for label smoothing during training.
        """
        # 1. database_specific.cwe_ids
        db_specific = vuln.get("database_specific", {})
        cwe_ids = db_specific.get("cwe_ids", [])
        for cwe_id in cwe_ids:
            if cwe_id in VALID_CWE:
                return cwe_id, "tier1_direct"

        # 2. Keyword inference from summary/details
        summary = vuln.get("summary", "") or ""
        details = vuln.get("details", "") or ""
        combined = (summary + " " + details).lower()
        for cwe, keywords in CWE_KEYWORD_MAP:
            for kw in keywords:
                if kw.lower() in combined:
                    return cwe, "tier3_inferred"

        return None, ""

    @staticmethod
    def _extract_severity(vuln: dict) -> float:
        """
        Extract CVSS v3 base score from OSV severity field.
        Returns 0.0 if not found.
        """
        severity_list = vuln.get("severity", [])
        for sev in severity_list:
            if sev.get("type") == "CVSS_V3":
                score_str = sev.get("score", "")
                if isinstance(score_str, (int, float)):
                    return float(score_str)
                # CVSS vector string: extract baseScore
                # Format: CVSS:3.1/AV:N/AC:L/...
                # Need to parse or just use 0.0
                if isinstance(score_str, str) and score_str.startswith("CVSS:"):
                    # Parse CVSS vector to extract base score
                    # OSV stores the vector, not the score directly
                    # We'll return 0.0 and let downstream handle it
                    return 0.0
        return 0.0

    @staticmethod
    def _extract_aliases(vuln: dict) -> dict:
        """Extract alias IDs (CVE, GHSA) from aliases field."""
        aliases: dict[str, str] = {}
        for alias in vuln.get("aliases", []):
            if alias.startswith("CVE-"):
                aliases["cve"] = alias
            elif alias.startswith("GHSA-"):
                aliases["ghsa"] = alias
        return aliases

    # ── Commit Fetching ──────────────────────────────────────────────

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

    # ── Process One Vulnerability ────────────────────────────────────

    def _process_vulnerability(
        self,
        vuln: dict,
        ecosystem: str,
        done_shas: set[str],
    ) -> None:
        """Full processing pipeline for one OSV vulnerability."""
        vuln_id = vuln.get("id", "")
        self.vulns_scanned += 1
        self.ecosystem_counts[ecosystem] += 1

        # Extract CWE (R-04: with confidence tier)
        cwe, cwe_confidence = self._extract_cwe(vuln)
        if not cwe:
            self.vulns_no_cwe += 1
            return

        # Extract severity
        severity_score = self._extract_severity(vuln)

        # Extract aliases
        aliases = self._extract_aliases(vuln)

        # Extract commit SHAs from GIT ranges
        commit_tuples = self._extract_commit_shas(vuln)
        if not commit_tuples:
            self.vulns_no_git_range += 1
            return

        # Process each commit
        for owner, repo, sha in commit_tuples:
            if self.max_samples > 0 and self._samples_saved >= self.max_samples:
                break

            # Cross-dedup against CVE data
            if sha in self._cve_dedup_shas:
                self.vulns_deduped_cve += 1
                continue

            # Skip already-processed commits
            if sha in done_shas:
                continue

            # Fetch commit from GitHub
            commit_data = self._fetch_commit(owner, repo, sha)
            if commit_data is None:
                self.commits_stale_404 += 1
                done_shas.add(sha)
                continue

            # Resolve full SHA
            full_sha = commit_data.get("sha", sha)
            if full_sha in self._cve_dedup_shas:
                self.vulns_deduped_cve += 1
                done_shas.add(sha)
                continue

            self.commits_processed += 1
            done_shas.add(sha)
            if full_sha != sha:
                done_shas.add(full_sha)

            # Process each file in the commit
            files = commit_data.get("files", [])
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

                commit_url = f"https://github.com/{owner}/{repo}/commit/{full_sha}"

                vuln_sample = {
                    "id": make_sample_id(),
                    "source": "osv",
                    "code": before_code,
                    "label": 1,
                    "cwe": cwe,
                    "language": "c",
                    "collected_at": timestamp,
                    "pair_id": pair_id,
                    "commit_sha": full_sha,
                    "severity_score": severity_score,
                    "file_path": filename,
                    "cfa_origin": "native",
                    "aliases": aliases,
                    "metadata": {
                        "osv_id": vuln_id,
                        "ecosystem": ecosystem,
                        "github_url": commit_url,
                        "owner": owner,
                        "repo": repo,
                        "cwe_confidence": cwe_confidence,
                    },
                }

                fixed_sample = {
                    "id": make_sample_id(),
                    "source": "osv",
                    "code": after_code,
                    "label": 0,
                    "cwe": cwe,
                    "language": "c",
                    "collected_at": timestamp,
                    "pair_id": pair_id,
                    "commit_sha": full_sha,
                    "severity_score": severity_score,
                    "file_path": filename,
                    "cfa_origin": "native",
                    "aliases": aliases,
                    "metadata": {
                        "osv_id": vuln_id,
                        "ecosystem": ecosystem,
                        "github_url": commit_url,
                        "owner": owner,
                        "repo": repo,
                        "cwe_confidence": cwe_confidence,
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

    # ── Main Collection Loop ─────────────────────────────────────────

    def collect(self) -> dict:
        """
        Download OSV bulk data, extract commit SHAs, fetch diffs, save pairs.
        Returns summary dict.
        """
        if self.dry_run:
            logger.info("DRY RUN MODE — will process but not save samples")

        # Load CVE cross-dedup set
        self._cve_dedup_shas = self._load_cve_dedup_set()

        # Load checkpoint
        checkpoint = self.load_checkpoint()
        done_shas: set[str] = set(checkpoint.get("done_shas", []))
        processed_vulns: set[str] = set(checkpoint.get("processed_vulns", []))
        completed_ecosystems: list[str] = checkpoint.get("completed_ecosystems", [])

        if done_shas:
            logger.info(
                f"Checkpoint: {len(done_shas)} commits processed, "
                f"{len(processed_vulns)} vulns processed"
            )

        commits_since_checkpoint = 0

        for ecosystem in ECOSYSTEMS:
            if self.max_samples > 0 and self._samples_saved >= self.max_samples:
                logger.info(f"Reached max_samples ({self.max_samples}). Stopping.")
                break

            if ecosystem in completed_ecosystems:
                logger.info(f"Skipping {ecosystem} — already completed")
                continue

            logger.info(f"Processing ecosystem: {ecosystem}")

            # Download bulk ZIP
            vulns = self._download_ecosystem_zip(ecosystem)
            if not vulns:
                completed_ecosystems.append(ecosystem)
                continue

            for vuln in vulns:
                if self.max_samples > 0 and self._samples_saved >= self.max_samples:
                    break

                vuln_id = vuln.get("id", "")
                if vuln_id in processed_vulns:
                    continue

                self._process_vulnerability(vuln, ecosystem, done_shas)
                processed_vulns.add(vuln_id)

                # Checkpoint periodically
                commits_since_checkpoint += 1
                if commits_since_checkpoint >= CHECKPOINT_INTERVAL:
                    self.save_checkpoint({
                        "done_shas": list(done_shas),
                        "processed_vulns": list(processed_vulns),
                        "completed_ecosystems": completed_ecosystems,
                        "samples_saved": self._samples_saved,
                    })
                    commits_since_checkpoint = 0

            completed_ecosystems.append(ecosystem)

            # Checkpoint after each ecosystem
            self.save_checkpoint({
                "done_shas": list(done_shas),
                "processed_vulns": list(processed_vulns),
                "completed_ecosystems": completed_ecosystems,
                "samples_saved": self._samples_saved,
            })

        # Final checkpoint
        self.save_checkpoint({
            "done_shas": list(done_shas),
            "processed_vulns": list(processed_vulns),
            "completed_ecosystems": completed_ecosystems,
            "samples_saved": self._samples_saved,
            "completed": True,
        })

        # Finalize orphan pairs
        if not self.dry_run:
            orphans = self.finalize_pairs()
            if orphans:
                logger.info(f"Cleared {orphans} orphan pair_ids")

        return self._build_summary()

    # ── Summary ──────────────────────────────────────────────────────

    def _build_summary(self) -> dict:
        return {
            "vulns_scanned": self.vulns_scanned,
            "vulns_no_git_range": self.vulns_no_git_range,
            "vulns_no_cwe": self.vulns_no_cwe,
            "vulns_deduped_cve": self.vulns_deduped_cve,
            "commits_processed": self.commits_processed,
            "commits_stale_404": self.commits_stale_404,
            "files_qualifying": self.files_qualifying,
            "samples_saved": self._samples_saved,
            "samples_failed": self._samples_failed,
            "pairs_saved": self.pairs_saved,
            "label_distribution": dict(self.label_counts),
            "cwe_distribution": dict(self.cwe_counts),
            "ecosystem_distribution": dict(self.ecosystem_counts),
            "tokens_available": len(self.rotator.tokens),
        }

    def print_summary(self) -> None:
        summary = self._build_summary()
        print("\n" + "=" * 60)
        print("OSV COLLECTOR SUMMARY")
        print("=" * 60)
        print(f"Vulns scanned:         {summary['vulns_scanned']}")
        print(f"No GIT range:          {summary['vulns_no_git_range']}")
        print(f"No CWE:                {summary['vulns_no_cwe']}")
        print(f"Deduped vs CVE:        {summary['vulns_deduped_cve']}")
        print(f"Commits processed:     {summary['commits_processed']}")
        print(f"Commits stale (404):   {summary['commits_stale_404']}")
        print(f"Files qualifying:      {summary['files_qualifying']}")
        print(f"Samples saved:         {summary['samples_saved']}")
        print(f"Samples failed:        {summary['samples_failed']}")
        print(f"Pairs saved:           {summary['pairs_saved']}")
        print(f"GitHub tokens:         {summary['tokens_available']}")
        if summary["ecosystem_distribution"]:
            print("\nEcosystem Distribution:")
            for eco, count in sorted(
                summary["ecosystem_distribution"].items(), key=lambda x: -x[1]
            ):
                print(f"  {eco}: {count}")
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
        description="OSV Collector for StreamGuard (Linux + OSS-Fuzz ecosystems)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run with 10 samples
  python -m training.scripts.collection.osv_collector --dry-run --max-samples 10

  # Full collection with multi-token rotation
  python -m training.scripts.collection.osv_collector --github-tokens "tok1,tok2"
        """,
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/osv",
        help="Output directory (default: data/raw/osv)",
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

    logger.info("Starting OSV collector...")
    collector = OSVCollector(
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
