"""
GitHub API Manager with Token Bucket Rate Limiting and Caching

Manages GitHub API calls with:
- Token bucket rate limiting (5000 requests/hour per token)
- Multiple token rotation
- Aggressive response caching
- Repository structure caching
"""

import time
import json
import hashlib
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import deque
from datetime import datetime
import threading


class TokenBucket:
    """Token bucket algorithm for rate limiting."""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Consume tokens from bucket, blocking if necessary.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True when tokens are consumed
        """
        with self.lock:
            while True:
                # Refill tokens based on time elapsed
                now = time.time()
                elapsed = now - self.last_refill
                self.tokens = min(
                    self.capacity,
                    self.tokens + (elapsed * self.refill_rate)
                )
                self.last_refill = now
                
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
                
                # Wait for tokens to refill
                wait_time = (tokens - self.tokens) / self.refill_rate
                time.sleep(min(wait_time, 1.0))  # Sleep max 1 second at a time


class GitHubAPIManager:
    """Manages GitHub API calls with rate limiting and caching."""
    
    def __init__(
        self,
        tokens: List[str],
        cache_dir: Path,
        requests_per_hour: int = 5000
    ):
        """
        Initialize GitHub API Manager.
        
        Args:
            tokens: List of GitHub tokens for rotation
            cache_dir: Directory for caching responses
            requests_per_hour: Rate limit per token
        """
        self.tokens = tokens if tokens else []
        self.current_token_idx = 0
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Token bucket: requests per hour converted to per second
        refill_rate = requests_per_hour / 3600.0
        self.bucket = TokenBucket(
            capacity=requests_per_hour,
            refill_rate=refill_rate
        )
        
        # Repository structure cache (in-memory)
        self.repo_cache = {}
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'rate_limit_waits': 0,
            'token_rotations': 0,
            'errors': 0
        }
        
        # Session for connection pooling
        self.session = requests.Session()
        if self.tokens:
            self._update_session_token()
    
    def _update_session_token(self):
        """Update session with current token."""
        if self.tokens:
            token = self.tokens[self.current_token_idx]
            self.session.headers.update({
                'Authorization': f'token {token}',
                'Accept': 'application/vnd.github.v3+json'
            })
    
    def _get_next_token(self) -> str:
        """
        Rotate to next token in pool.
        
        Returns:
            Next token
        """
        if not self.tokens:
            return ""
        
        self.current_token_idx = (self.current_token_idx + 1) % len(self.tokens)
        self.stats['token_rotations'] += 1
        self._update_session_token()
        return self.tokens[self.current_token_idx]
    
    def _make_cache_key(self, *args) -> str:
        """Generate cache key from arguments."""
        key_str = "_".join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _load_cache(self, cache_key: str) -> Optional[Dict]:
        """Load data from cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.stats['cache_hits'] += 1
                    return json.load(f)
            except Exception:
                return None
        return None
    
    def _save_cache(self, cache_key: str, data: Dict):
        """Save data to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass  # Silently fail cache writes
    
    def _make_request(
        self,
        url: str,
        method: str = 'GET',
        **kwargs
    ) -> Optional[Dict]:
        """
        Make HTTP request with error handling.
        
        Args:
            url: Request URL
            method: HTTP method
            **kwargs: Additional request arguments
            
        Returns:
            Response JSON or None
        """
        try:
            response = self.session.request(method, url, timeout=30, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                # Rate limit hit, try rotating token
                self._get_next_token()
                self.stats['rate_limit_waits'] += 1
            self.stats['errors'] += 1
            return None
        except Exception:
            self.stats['errors'] += 1
            return None
    
    def get_commit(
        self,
        owner: str,
        repo: str,
        sha: str
    ) -> Optional[Dict]:
        """
        Fetch commit with caching and rate limiting.
        
        Args:
            owner: Repository owner
            repo: Repository name
            sha: Commit SHA
            
        Returns:
            Commit data or None
        """
        cache_key = self._make_cache_key('commit', owner, repo, sha)
        
        # Check cache first
        cached = self._load_cache(cache_key)
        if cached:
            return cached
        
        self.stats['cache_misses'] += 1
        
        # Wait for token bucket
        self.bucket.consume(1)
        self.stats['total_requests'] += 1
        
        # Make request
        url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
        data = self._make_request(url)
        
        if data:
            self._save_cache(cache_key, data)
        
        return data
    
    def get_commit_diff(
        self,
        owner: str,
        repo: str,
        sha: str
    ) -> Optional[str]:
        """
        Fetch commit diff/patch.
        
        Args:
            owner: Repository owner
            repo: Repository name
            sha: Commit SHA
            
        Returns:
            Diff text or None
        """
        cache_key = self._make_cache_key('diff', owner, repo, sha)
        
        # Check cache
        cached = self._load_cache(cache_key)
        if cached:
            return cached.get('diff')
        
        self.stats['cache_misses'] += 1
        
        # Wait for token bucket
        self.bucket.consume(1)
        self.stats['total_requests'] += 1
        
        # Request with diff accept header
        url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
        try:
            response = self.session.get(
                url,
                headers={'Accept': 'application/vnd.github.v3.diff'},
                timeout=30
            )
            response.raise_for_status()
            diff = response.text
            
            # Cache the diff
            self._save_cache(cache_key, {'diff': diff})
            return diff
        except Exception:
            self.stats['errors'] += 1
            return None
    
    def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: str = 'main'
    ) -> Optional[str]:
        """
        Fetch file content from repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            path: File path in repo
            ref: Git reference (branch/tag/commit)
            
        Returns:
            File content or None
        """
        cache_key = self._make_cache_key('file', owner, repo, path, ref)
        
        # Check cache
        cached = self._load_cache(cache_key)
        if cached:
            return cached.get('content')
        
        self.stats['cache_misses'] += 1
        
        # Wait for token bucket
        self.bucket.consume(1)
        self.stats['total_requests'] += 1
        
        # Make request
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        data = self._make_request(url, params={'ref': ref})
        
        if data and data.get('content'):
            import base64
            try:
                content = base64.b64decode(data['content']).decode('utf-8')
                self._save_cache(cache_key, {'content': content})
                return content
            except Exception:
                return None
        
        return None
    
    def search_commits(
        self,
        query: str,
        owner: str,
        repo: str,
        max_results: int = 30
    ) -> List[Dict]:
        """
        Search commits in repository.
        
        Args:
            query: Search query
            owner: Repository owner
            repo: Repository name
            max_results: Maximum results to return
            
        Returns:
            List of commit data
        """
        cache_key = self._make_cache_key('search', owner, repo, query, max_results)
        
        # Check cache
        cached = self._load_cache(cache_key)
        if cached:
            return cached.get('commits', [])
        
        self.stats['cache_misses'] += 1
        
        # Wait for token bucket
        self.bucket.consume(1)
        self.stats['total_requests'] += 1
        
        # Make request
        search_query = f"{query} repo:{owner}/{repo}"
        url = "https://api.github.com/search/commits"
        data = self._make_request(
            url,
            params={'q': search_query, 'per_page': max_results},
            headers={'Accept': 'application/vnd.github.cloak-preview+json'}
        )
        
        commits = []
        if data and 'items' in data:
            commits = data['items']
            self._save_cache(cache_key, {'commits': commits})
        
        return commits
    
    def get_repo_info(
        self,
        owner: str,
        repo: str
    ) -> Optional[Dict]:
        """
        Get repository information.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            Repository data or None
        """
        repo_key = f"{owner}/{repo}"
        
        # Check in-memory cache
        if repo_key in self.repo_cache:
            return self.repo_cache[repo_key]
        
        cache_key = self._make_cache_key('repo', owner, repo)
        
        # Check disk cache
        cached = self._load_cache(cache_key)
        if cached:
            self.repo_cache[repo_key] = cached
            return cached
        
        self.stats['cache_misses'] += 1
        
        # Wait for token bucket
        self.bucket.consume(1)
        self.stats['total_requests'] += 1
        
        # Make request
        url = f"https://api.github.com/repos/{owner}/{repo}"
        data = self._make_request(url)
        
        if data:
            self._save_cache(cache_key, data)
            self.repo_cache[repo_key] = data
        
        return data
    
    def parse_github_url(self, url: str) -> Optional[Tuple[str, str, Optional[str]]]:
        """
        Parse GitHub URL to extract owner, repo, and optional commit SHA.
        
        Args:
            url: GitHub URL
            
        Returns:
            Tuple of (owner, repo, sha) or None
        """
        import re
        
        # Match commit URL: github.com/owner/repo/commit/sha
        commit_pattern = r'github\.com/([^/]+)/([^/]+)/commit/([a-f0-9]+)'
        match = re.search(commit_pattern, url)
        if match:
            return match.group(1), match.group(2), match.group(3)
        
        # Match repo URL: github.com/owner/repo
        repo_pattern = r'github\.com/([^/]+)/([^/]+?)(?:/|$)'
        match = re.search(repo_pattern, url)
        if match:
            return match.group(1), match.group(2), None
        
        return None
    
    def get_stats(self) -> Dict:
        """Get API manager statistics."""
        total = self.stats['cache_hits'] + self.stats['cache_misses']
        cache_hit_rate = (self.stats['cache_hits'] / total * 100) if total > 0 else 0
        
        return {
            **self.stats,
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'tokens_available': len(self.tokens)
        }
    
    def print_stats(self):
        """Print API manager statistics."""
        stats = self.get_stats()
        print("\n" + "="*60)
        print("GitHub API Manager Statistics")
        print("="*60)
        print(f"Total requests:      {stats['total_requests']}")
        print(f"Cache hits:          {stats['cache_hits']}")
        print(f"Cache misses:        {stats['cache_misses']}")
        print(f"Cache hit rate:      {stats['cache_hit_rate']}")
        print(f"Rate limit waits:    {stats['rate_limit_waits']}")
        print(f"Token rotations:     {stats['token_rotations']}")
        print(f"Errors:              {stats['errors']}")
        print(f"Tokens available:    {stats['tokens_available']}")
        print("="*60)
