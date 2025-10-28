"""Enhanced Repository Miner for StreamGuard.

Mines security-related commits from open-source repositories to extract
vulnerable/fixed code pairs for training data.
"""

import json
import re
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

import git
from git import Repo, GitCommandError

from base_collector import BaseCollector


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedRepoMiner(BaseCollector):
    """Enhanced repository miner for security vulnerabilities."""

    # Security-related keywords to identify relevant commits
    SECURITY_KEYWORDS = [
        "security", "vulnerability", "CVE", "SQL injection", "XSS", "CSRF",
        "command injection", "RCE", "path traversal", "SSRF", "XXE",
        "fix security", "authentication bypass", "deserialization",
        "unsafe", "sanitize", "exploit", "malicious", "injection",
        "buffer overflow", "privilege escalation", "information disclosure"
    ]

    # Repository configurations with target sample counts
    REPOSITORIES = {
        # Python repositories
        "django/django": {"language": "python", "target": 3500},
        "pallets/flask": {"language": "python", "target": 3000},
        "sqlalchemy/sqlalchemy": {"language": "python", "target": 3000},
        "psf/requests": {"language": "python", "target": 2500},
        "tiangolo/fastapi": {"language": "python", "target": 2500},
        "Pylons/pyramid": {"language": "python", "target": 2000},

        # JavaScript repositories
        "expressjs/express": {"language": "javascript", "target": 3500},
        "nodejs/node": {"language": "javascript", "target": 3500},
        "koajs/koa": {"language": "javascript", "target": 2500},
        "fastify/fastify": {"language": "javascript", "target": 2500},
        "nestjs/nest": {"language": "javascript", "target": 2500},
        "hapijs/hapi": {"language": "javascript", "target": 2000},
    }

    def __init__(self, output_dir: str = "data/raw/opensource", cache_enabled: bool = True):
        """
        Initialize Enhanced Repository Miner.

        Args:
            output_dir: Directory to save collected data
            cache_enabled: Whether to cache cloned repositories
        """
        super().__init__(output_dir, cache_enabled)

        # Create repos cache directory
        self.repos_dir = Path(output_dir) / "repos"
        self.repos_dir.mkdir(parents=True, exist_ok=True)

        self.since_date = datetime.now() - timedelta(days=3*365)  # Last 3 years
        self.all_samples = []

    def collect(self) -> List[Dict]:
        """
        Collect vulnerability samples from all repositories.

        Returns:
            List of collected samples
        """
        logger.info(f"Starting repository mining for {len(self.REPOSITORIES)} repositories")
        logger.info(f"Target: ~20,000 samples total")
        logger.info(f"Mining commits since: {self.since_date.strftime('%Y-%m-%d')}")

        samples = self.mine_all_repositories()

        logger.info(f"Total samples collected: {len(samples)}")

        return samples

    def mine_all_repositories(self) -> List[Dict]:
        """
        Mine all repositories using multiprocessing.

        Returns:
            List of all collected samples
        """
        all_samples = []

        # Use ProcessPoolExecutor for parallel mining
        with ProcessPoolExecutor(max_workers=4) as executor:
            future_to_repo = {
                executor.submit(
                    self._mine_repository_wrapper,
                    repo_name,
                    config
                ): repo_name
                for repo_name, config in self.REPOSITORIES.items()
            }

            for future in as_completed(future_to_repo):
                repo_name = future_to_repo[future]
                try:
                    samples = future.result()
                    all_samples.extend(samples)
                    logger.info(
                        f"Completed {repo_name}: {len(samples)} samples "
                        f"(Total: {len(all_samples)})"
                    )
                except Exception as e:
                    logger.error(f"Error mining {repo_name}: {str(e)}")
                    self.log_error(f"Failed to mine {repo_name}", {"error": str(e)})

        # Deduplicate samples
        logger.info("Deduplicating samples...")
        all_samples = self.deduplicate_samples(all_samples, key="vulnerable_code")

        logger.info(f"Final sample count after deduplication: {len(all_samples)}")

        return all_samples

    def _mine_repository_wrapper(self, repo_name: str, config: Dict) -> List[Dict]:
        """
        Wrapper for mine_repository to handle multiprocessing.

        Args:
            repo_name: Repository name (org/repo)
            config: Repository configuration

        Returns:
            List of samples from this repository
        """
        try:
            return self.mine_repository(repo_name, config)
        except Exception as e:
            logger.error(f"Error in wrapper for {repo_name}: {str(e)}")
            return []

    def mine_repository(self, repo_name: str, config: Dict) -> List[Dict]:
        """
        Mine a single repository for security-related commits.

        Args:
            repo_name: Repository name (org/repo)
            config: Repository configuration with language and target count

        Returns:
            List of samples from this repository
        """
        logger.info(f"Mining repository: {repo_name}")

        samples = []
        target = config["target"]
        language = config["language"]

        try:
            # Clone or open repository
            repo = self._get_repository(repo_name)

            # Find security-related commits
            security_commits = self.find_security_commits(repo)
            logger.info(f"{repo_name}: Found {len(security_commits)} security commits")

            # Extract samples from commits
            for commit in security_commits:
                if len(samples) >= target:
                    break

                try:
                    commit_samples = self.extract_from_commit(
                        repo, commit, repo_name, language
                    )
                    samples.extend(commit_samples)

                    if len(samples) % 100 == 0:
                        logger.info(f"{repo_name}: Extracted {len(samples)}/{target} samples")

                except Exception as e:
                    logger.warning(f"Error extracting from commit {commit.hexsha[:8]}: {str(e)}")
                    continue

            logger.info(f"{repo_name}: Completed with {len(samples)} samples")

        except Exception as e:
            logger.error(f"Error mining repository {repo_name}: {str(e)}")
            self.log_error(f"Failed to mine {repo_name}", {"error": str(e)})

        return samples[:target]  # Limit to target count

    def _get_repository(self, repo_name: str) -> Repo:
        """
        Get repository instance, cloning if necessary.

        Args:
            repo_name: Repository name (org/repo)

        Returns:
            GitPython Repo object
        """
        repo_path = self.repos_dir / repo_name.replace("/", "_")

        if repo_path.exists():
            logger.info(f"Using cached repository: {repo_path}")
            try:
                repo = Repo(repo_path)
                # Try to pull latest changes
                try:
                    origin = repo.remotes.origin
                    origin.pull()
                    logger.info(f"Pulled latest changes for {repo_name}")
                except Exception as e:
                    logger.warning(f"Could not pull updates for {repo_name}: {str(e)}")
                return repo
            except Exception as e:
                logger.warning(f"Could not open cached repo, re-cloning: {str(e)}")
                # Remove corrupted repo and re-clone
                import shutil
                shutil.rmtree(repo_path)

        # Clone repository
        logger.info(f"Cloning repository: {repo_name}")
        repo_url = f"https://github.com/{repo_name}.git"

        try:
            repo = Repo.clone_from(repo_url, repo_path, depth=None)
            logger.info(f"Successfully cloned {repo_name}")
            return repo
        except GitCommandError as e:
            logger.error(f"Failed to clone {repo_name}: {str(e)}")
            raise

    def find_security_commits(self, repo: Repo) -> List:
        """
        Find commits with security-related keywords in commit message.

        Args:
            repo: GitPython Repo object

        Returns:
            List of security-related commits
        """
        security_commits = []

        try:
            # Get all commits since the specified date
            commits = list(repo.iter_commits(
                'HEAD',
                since=self.since_date.strftime('%Y-%m-%d')
            ))

            logger.info(f"Analyzing {len(commits)} commits for security keywords")

            for commit in commits:
                if self.is_security_commit(commit):
                    security_commits.append(commit)

        except Exception as e:
            logger.error(f"Error finding security commits: {str(e)}")
            raise

        return security_commits

    def is_security_commit(self, commit) -> bool:
        """
        Determine if a commit is security-related based on commit message.

        Args:
            commit: GitPython commit object

        Returns:
            True if commit appears to be security-related
        """
        commit_message = commit.message.lower()

        # Check for security keywords
        for keyword in self.SECURITY_KEYWORDS:
            if keyword.lower() in commit_message:
                return True

        return False

    def extract_from_commit(
        self,
        repo: Repo,
        commit,
        repo_name: str,
        language: str
    ) -> List[Dict]:
        """
        Extract vulnerable/fixed code pairs from a commit diff.

        Args:
            repo: GitPython Repo object
            commit: Commit object
            repo_name: Repository name
            language: Programming language

        Returns:
            List of sample dictionaries
        """
        samples = []

        try:
            # Get parent commit for comparison
            if not commit.parents:
                return samples  # Skip initial commits

            parent = commit.parents[0]

            # Get diff between parent and current commit
            diffs = parent.diff(commit, create_patch=True)

            for diff in diffs:
                # Skip if not a relevant file type
                if not self._is_relevant_file(diff.a_path or diff.b_path, language):
                    continue

                # Skip binary files
                if diff.diff is None or len(diff.diff) == 0:
                    continue

                try:
                    # Extract vulnerable and fixed code
                    vulnerable_code, fixed_code = self._extract_code_from_diff(diff)

                    if not vulnerable_code or not fixed_code:
                        continue

                    # Validate code quality
                    if not self.validate_code(vulnerable_code) or not self.validate_code(fixed_code):
                        continue

                    # Extract vulnerability type
                    vuln_type = self.extract_vulnerability_type(commit.message)

                    # Create sample
                    sample = {
                        "vulnerable_code": vulnerable_code,
                        "fixed_code": fixed_code,
                        "commit_sha": commit.hexsha,
                        "commit_message": commit.message.strip(),
                        "repository": repo_name,
                        "file_path": diff.b_path or diff.a_path,
                        "vulnerability_type": vuln_type,
                        "committed_date": commit.committed_datetime.isoformat(),
                        "source": "opensource_repo",
                        "language": language
                    }

                    samples.append(sample)
                    self.samples_collected += 1

                except Exception as e:
                    logger.debug(f"Error extracting code from diff: {str(e)}")
                    continue

        except Exception as e:
            logger.warning(f"Error processing commit {commit.hexsha[:8]}: {str(e)}")

        return samples

    def _is_relevant_file(self, file_path: Optional[str], language: str) -> bool:
        """
        Check if file is relevant for the given language.

        Args:
            file_path: Path to file
            language: Programming language

        Returns:
            True if file is relevant
        """
        if not file_path:
            return False

        file_path = file_path.lower()

        if language == "python":
            return file_path.endswith('.py')
        elif language == "javascript":
            return file_path.endswith(('.js', '.ts', '.jsx', '.tsx'))

        return False

    def _extract_code_from_diff(self, diff) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract vulnerable and fixed code from a diff.

        Args:
            diff: GitPython diff object

        Returns:
            Tuple of (vulnerable_code, fixed_code)
        """
        try:
            diff_text = diff.diff.decode('utf-8', errors='ignore')
        except Exception:
            return None, None

        # Parse diff to extract removed and added lines
        removed_lines = []
        added_lines = []
        context_lines = []

        lines = diff_text.split('\n')
        for line in lines:
            if line.startswith('---') or line.startswith('+++'):
                continue
            elif line.startswith('@@'):
                continue
            elif line.startswith('-'):
                removed_lines.append(line[1:])
            elif line.startswith('+'):
                added_lines.append(line[1:])
            elif line.startswith(' '):
                context_lines.append(line[1:])

        # Build vulnerable code (with removed lines)
        vulnerable_code = self._build_code_snippet(context_lines, removed_lines, added_lines, use_removed=True)

        # Build fixed code (with added lines)
        fixed_code = self._build_code_snippet(context_lines, removed_lines, added_lines, use_removed=False)

        return vulnerable_code, fixed_code

    def _build_code_snippet(
        self,
        context_lines: List[str],
        removed_lines: List[str],
        added_lines: List[str],
        use_removed: bool
    ) -> Optional[str]:
        """
        Build code snippet from diff components.

        Args:
            context_lines: Context lines from diff
            removed_lines: Removed lines
            added_lines: Added lines
            use_removed: If True, use removed lines; else use added lines

        Returns:
            Code snippet or None
        """
        if use_removed:
            target_lines = removed_lines
        else:
            target_lines = added_lines

        if not target_lines:
            return None

        # Combine context and target lines
        # Take some context before and after
        code_lines = []

        # Add context
        if context_lines:
            code_lines.extend(context_lines[:5])  # First 5 context lines

        # Add target lines
        code_lines.extend(target_lines)

        # Add more context
        if context_lines:
            code_lines.extend(context_lines[-5:])  # Last 5 context lines

        code = '\n'.join(code_lines).strip()

        return code if code else None

    def save_samples_to_file(self, samples: List[Dict], filename: str = "mined_samples.jsonl"):
        """
        Save collected samples to JSONL file.

        Args:
            samples: List of sample dictionaries
            filename: Output filename
        """
        output_path = self.save_samples(samples, filename)
        logger.info(f"Saved {len(samples)} samples to {output_path}")

        # Also save statistics
        stats = self.get_stats()
        stats["total_samples"] = len(samples)
        stats["repositories"] = len(self.REPOSITORIES)

        # Count samples by repository
        repo_counts = {}
        for sample in samples:
            repo = sample.get("repository", "unknown")
            repo_counts[repo] = repo_counts.get(repo, 0) + 1
        stats["samples_per_repo"] = repo_counts

        # Count samples by vulnerability type
        vuln_counts = {}
        for sample in samples:
            vuln = sample.get("vulnerability_type", "unknown")
            vuln_counts[vuln] = vuln_counts.get(vuln, 0) + 1
        stats["samples_per_vulnerability"] = vuln_counts

        stats_file = self.output_dir / f"{Path(filename).stem}_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved statistics to {stats_file}")


def main():
    """Main entry point for repository mining."""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Repository Miner")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/opensource",
        help="Output directory for collected samples"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching of cloned repositories"
    )

    args = parser.parse_args()

    # Initialize miner
    miner = EnhancedRepoMiner(
        output_dir=args.output_dir,
        cache_enabled=not args.no_cache
    )

    # Collect samples
    logger.info("Starting enhanced repository mining...")
    samples = miner.collect()

    # Save samples
    miner.save_samples_to_file(samples)

    # Print statistics
    stats = miner.get_stats()
    logger.info("\n" + "="*50)
    logger.info("Mining Complete!")
    logger.info("="*50)
    logger.info(f"Total samples collected: {stats['total_samples']}")
    logger.info(f"Errors encountered: {stats['errors_count']}")
    logger.info("\nSamples per repository:")
    for repo, count in stats['samples_per_repo'].items():
        logger.info(f"  {repo}: {count}")
    logger.info("\nSamples per vulnerability type:")
    for vuln, count in stats['samples_per_vulnerability'].items():
        logger.info(f"  {vuln}: {count}")


if __name__ == "__main__":
    main()
