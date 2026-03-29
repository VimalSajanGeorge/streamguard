"""
Dynamic Repository Discoverer for StreamGuard

Discovers repositories dynamically from OSV vulnerability data to:
- Increase repository diversity (reduce overfitting)
- Target repos with known vulnerabilities
- Build dynamic mining list instead of hardcoded repos
"""

import requests
import json
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
import re

try:
    from .github_api_manager import GitHubAPIManager
except ImportError:
    from github_api_manager import GitHubAPIManager


class DynamicRepoDiscoverer:
    """Discover repositories dynamically from OSV vulnerability data."""
    
    OSV_GCS_BASE = "https://osv-vulnerabilities.storage.googleapis.com"
    
    def __init__(self, api_manager: GitHubAPIManager):
        """
        Initialize dynamic repo discoverer.
        
        Args:
            api_manager: GitHub API manager for repo lookups
        """
        self.api_manager = api_manager
        self.discovered_repos = {}  # repo_url -> vuln_count
        self.package_to_repo = {}  # package_name -> repo_url (cache)
    
    def discover_from_osv(
        self,
        ecosystems: List[str] = ['PyPI', 'npm'],
        min_vulns: int = 3,
        max_repos: int = 200
    ) -> List[Dict]:
        """
        Discover repos with vulnerabilities from OSV data.
        
        Args:
            ecosystems: List of ecosystems to query
            min_vulns: Minimum vulnerabilities per repo
            max_repos: Maximum repos to return
            
        Returns:
            List of repo dicts sorted by vulnerability count
        """
        print(f"\n{'='*60}")
        print("Dynamic Repository Discovery")
        print(f"{'='*60}")
        print(f"Ecosystems: {', '.join(ecosystems)}")
        print(f"Min vulnerabilities: {min_vulns}")
        print(f"Max repos: {max_repos}")
        print()
        
        for ecosystem in ecosystems:
            print(f"Processing {ecosystem}...")
            self._process_ecosystem(ecosystem)
        
        # Filter and sort by vulnerability count
        repos = [
            {
                'repo': repo,
                'vuln_count': count,
                'language': self._detect_language(repo),
                'owner_repo': self._extract_owner_repo(repo)
            }
            for repo, count in self.discovered_repos.items()
            if count >= min_vulns
        ]
        
        repos = sorted(repos, key=lambda x: x['vuln_count'], reverse=True)
        
        print(f"\n{'='*60}")
        print(f"Discovery Complete")
        print(f"{'='*60}")
        print(f"Total repos discovered: {len(repos)}")
        print(f"Returning top {min(max_repos, len(repos))} repos")
        
        if repos:
            print(f"\nTop 10 repos by vulnerability count:")
            for i, repo in enumerate(repos[:10], 1):
                print(f"  {i}. {repo['owner_repo']}: {repo['vuln_count']} vulns ({repo['language']})")
        
        return repos[:max_repos]
    
    def _process_ecosystem(self, ecosystem: str):
        """
        Process an ecosystem to discover repos.
        
        Args:
            ecosystem: Ecosystem name (PyPI, npm, etc.)
        """
        # Download vulnerability list from OSV GCS
        vuln_ids = self._download_ecosystem_vulns(ecosystem)
        
        if not vuln_ids:
            print(f"  No vulnerabilities found for {ecosystem}")
            return
        
        print(f"  Found {len(vuln_ids)} vulnerabilities")
        print(f"  Resolving packages to repos...")
        
        # Sample vulnerabilities (process max 1000 for efficiency)
        sample_size = min(1000, len(vuln_ids))
        sampled_ids = vuln_ids[:sample_size]
        
        resolved = 0
        for i, vuln_id in enumerate(sampled_ids):
            if i % 100 == 0 and i > 0:
                print(f"    Processed {i}/{sample_size} ({resolved} repos found)")
            
            # Fetch vulnerability details
            vuln_data = self._fetch_vulnerability(vuln_id)
            if not vuln_data:
                continue
            
            # Extract package name
            affected = vuln_data.get('affected', [])
            if not affected:
                continue
            
            package_info = affected[0].get('package', {})
            package_name = package_info.get('name')
            
            if not package_name:
                continue
            
            # Resolve to GitHub repo
            repo_url = self._resolve_to_github(package_name, ecosystem)
            if repo_url:
                self.discovered_repos[repo_url] = \
                    self.discovered_repos.get(repo_url, 0) + 1
                resolved += 1
        
        print(f"  Resolved {resolved} packages to GitHub repos")
    
    def _download_ecosystem_vulns(self, ecosystem: str) -> List[str]:
        """
        Download vulnerability IDs for an ecosystem from OSV GCS.
        
        Args:
            ecosystem: Ecosystem name
            
        Returns:
            List of vulnerability IDs
        """
        try:
            url = f"{self.OSV_GCS_BASE}/{ecosystem}/all.zip"
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            import zipfile
            import io
            
            vuln_ids = []
            zip_data = io.BytesIO(response.content)
            
            with zipfile.ZipFile(zip_data) as zip_file:
                json_files = [f for f in zip_file.namelist() if f.endswith('.json')]
                
                for json_file in json_files:
                    try:
                        with zip_file.open(json_file) as f:
                            vuln_data = json.load(f)
                            vuln_id = vuln_data.get("id")
                            if vuln_id:
                                vuln_ids.append(vuln_id)
                    except Exception:
                        continue
            
            return vuln_ids
        
        except Exception as e:
            print(f"  Error downloading {ecosystem}: {e}")
            return []
    
    def _fetch_vulnerability(self, vuln_id: str) -> Optional[Dict]:
        """
        Fetch vulnerability details from OSV API.
        
        Args:
            vuln_id: Vulnerability ID
            
        Returns:
            Vulnerability data or None
        """
        try:
            url = f"https://api.osv.dev/v1/vulns/{vuln_id}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return None
    
    def _resolve_to_github(
        self,
        package_name: str,
        ecosystem: str
    ) -> Optional[str]:
        """
        Resolve package name to GitHub repository URL.
        
        Args:
            package_name: Package name
            ecosystem: Ecosystem (PyPI, npm)
            
        Returns:
            GitHub repository URL or None
        """
        # Check cache
        cache_key = f"{ecosystem}:{package_name}"
        if cache_key in self.package_to_repo:
            return self.package_to_repo[cache_key]
        
        repo_url = None
        
        try:
            if ecosystem == "PyPI":
                # Query PyPI API
                url = f"https://pypi.org/pypi/{package_name}/json"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    info = data.get('info', {})
                    
                    # Try multiple fields
                    project_urls = info.get('project_urls', {})
                    if isinstance(project_urls, dict):
                        for key, value in project_urls.items():
                            if value and 'github.com' in value:
                                repo_url = self._clean_github_url(value)
                                break
                    
                    if not repo_url:
                        home_page = info.get('home_page', '')
                        if 'github.com' in home_page:
                            repo_url = self._clean_github_url(home_page)
            
            elif ecosystem == "npm":
                # Query npm registry
                url = f"https://registry.npmjs.org/{package_name}"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    repo = data.get('repository', {})
                    
                    if isinstance(repo, dict):
                        repo_url_raw = repo.get('url', '')
                    else:
                        repo_url_raw = str(repo)
                    
                    if 'github.com' in repo_url_raw:
                        repo_url = self._clean_github_url(repo_url_raw)
        
        except Exception:
            pass
        
        # Cache result
        if repo_url:
            self.package_to_repo[cache_key] = repo_url
        
        return repo_url
    
    def _clean_github_url(self, url: str) -> str:
        """
        Clean and normalize GitHub URL.
        
        Args:
            url: Raw GitHub URL
            
        Returns:
            Cleaned URL
        """
        # Remove git:// or git+ prefixes
        url = url.replace('git://', 'https://')
        url = url.replace('git+https://', 'https://')
        url = url.replace('git+ssh://git@', 'https://')
        
        # Remove .git suffix
        url = url.rstrip('.git')
        
        # Extract just the repo URL
        match = re.search(r'github\.com/([^/]+/[^/]+)', url)
        if match:
            return f"https://github.com/{match.group(1)}"
        
        return url
    
    def _detect_language(self, repo_url: str) -> str:
        """
        Detect language from repository.
        
        Args:
            repo_url: GitHub repository URL
            
        Returns:
            Language name
        """
        parsed = self.api_manager.parse_github_url(repo_url)
        if not parsed:
            return "unknown"
        
        owner, repo, _ = parsed
        
        # Get repo info from GitHub
        repo_info = self.api_manager.get_repo_info(owner, repo)
        if repo_info:
            language = repo_info.get('language', 'unknown')
            return language.lower() if language else "unknown"
        
        return "unknown"
    
    def _extract_owner_repo(self, repo_url: str) -> str:
        """
        Extract owner/repo from URL.
        
        Args:
            repo_url: GitHub repository URL
            
        Returns:
            owner/repo string
        """
        match = re.search(r'github\.com/([^/]+/[^/]+)', repo_url)
        if match:
            return match.group(1)
        return repo_url
    
    def save_discovered_repos(self, repos: List[Dict], output_file: str):
        """
        Save discovered repos to JSON file.
        
        Args:
            repos: List of repo dictionaries
            output_file: Output file path
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(repos, f, indent=2)
        
        print(f"\nSaved discovered repos to: {output_path}")


def main():
    """Test the dynamic repo discoverer."""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Get GitHub tokens
    tokens_str = os.getenv('GITHUB_TOKENS', os.getenv('GITHUB_TOKEN', ''))
    tokens = [t.strip() for t in tokens_str.split(',') if t.strip()]
    
    if not tokens:
        print("Warning: No GitHub tokens found. Set GITHUB_TOKENS in .env")
        tokens = []
    
    # Initialize API manager
    cache_dir = Path("data/raw/.cache")
    api_manager = GitHubAPIManager(tokens, cache_dir)
    
    # Discover repos
    discoverer = DynamicRepoDiscoverer(api_manager)
    repos = discoverer.discover_from_osv(
        ecosystems=['PyPI', 'npm'],
        min_vulns=3,
        max_repos=100
    )
    
    # Save results
    discoverer.save_discovered_repos(repos, "data/raw/discovered_repos.json")
    
    print(f"\nDiscovered {len(repos)} repositories")


if __name__ == '__main__':
    main()
